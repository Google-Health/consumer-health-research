"""PHA Portal API - FastAPI backend for the PHA user study portal.

This provides REST endpoints for:
- Configuration (models, personas, baselines)
- Chat sessions with different baselines
- Health data access

Run with:
    cd personal_health_agent
    uvicorn api.main:app --reload --port 8000
"""

import os
import json
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Settings
from pha.utils.model_discovery import (
    get_available_models,
    get_default_model,
    get_recommended_models,
    get_openai_models,
    get_gemini_models,
    get_anthropic_models,
    KNOWN_GEMINI_MODELS,
    KNOWN_OPENAI_MODELS,
    KNOWN_ANTHROPIC_MODELS,
)
from pha.utils.personas import (
    list_personas,
    get_default_persona,
    load_persona_data,
)
from .utils.chat_logger import ChatSessionLogger
from pha.utils.request_stats import request_stats

GLOBAL_USER_STUDY_FLAG = False

# Path to user study defaults (gitignored; copy from user_study_defaults.example.json)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
USER_STUDY_DEFAULTS_PATH = PROJECT_ROOT / "config" / "user_study_defaults.json"


def _load_user_study_defaults() -> Optional[Dict[str, Any]]:
    """Load user study defaults from config file. Returns None if not found or invalid."""
    if not USER_STUDY_DEFAULTS_PATH.exists():
        return None
    try:
        with open(USER_STUDY_DEFAULTS_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


# =============================================================================
# Application Setup
# =============================================================================

# Store active sessions
sessions: Dict[str, "ChatSession"] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    print("🚀 PHA Portal API starting...")
    yield
    print("👋 PHA Portal API shutting down...")
    # Close all session loggers before clearing
    for session in sessions.values():
        if hasattr(session, 'logger') and session.logger:
            try:
                session.logger.close()
            except Exception:
                pass
    sessions.clear()


app = FastAPI(
    title="PHA Portal API",
    description="Backend API for the Personal Health Agent system",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Pydantic Models
# =============================================================================

class ModelOption(BaseModel):
    """Available model option."""
    id: str
    name: str
    provider: str
    description: Optional[str] = None
    available: bool = True


class PersonaOption(BaseModel):
    """Available persona option."""
    id: str
    name: str
    description: str = ""
    demographics: Optional[Dict[str, Any]] = None
    data_summary: Optional[Dict[str, Any]] = None


class BaselineOption(BaseModel):
    """Available baseline option."""
    id: str
    name: str
    description: str
    architecture: str


class ConfigResponse(BaseModel):
    """Configuration options response."""
    providers: List[str]
    models: Dict[str, List[ModelOption]]
    personas: List[PersonaOption]
    baselines: List[BaselineOption]
    defaults: Dict[str, str]
    user_study_mode: bool = False


class SessionConfig(BaseModel):
    """Configuration for creating a new session."""
    provider: Literal["gemini", "openai", "anthropic"] = "gemini"
    model_id: str = "models/gemini-3.1-flash-lite-preview"
    baseline: Literal["pha", "parallel", "phia"] = "pha"
    persona_id: str = "sample"
    user_id: str = "test"
    
    # API keys (optional, can use env vars)
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None


class SessionResponse(BaseModel):
    """Response when creating a session."""
    session_id: str
    config: SessionConfig
    created_at: str
    status: str = "ready"


class ChatMessage(BaseModel):
    """A chat message."""
    role: Literal["user", "assistant"]
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    """Request to send a message."""
    message: str


class SaveConversationLogRequest(BaseModel):
    """Request to save a conversation log."""
    filename: str
    content: str


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""
    message: ChatMessage
    session_id: str
    processing_time_ms: Optional[float] = None


class SessionState(BaseModel):
    """Current state of a session."""
    session_id: str
    config: SessionConfig
    message_count: int
    created_at: str
    last_activity: str
    status: str


# =============================================================================
# Chat Session Manager
# =============================================================================

class ChatSession:
    """Manages a chat session with a specific baseline."""
    
    def __init__(self, session_id: str, config: SessionConfig):
        self.session_id = session_id
        self.config = config
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.messages: List[ChatMessage] = []
        self.agent = None
        self._initialized = False
        
        # Initialize structured logger
        from pathlib import Path
        logs_root = Path(__file__).parent.parent / "chat_logs"
        self.logger = ChatSessionLogger(
            user_id=config.user_id,
            session_id=session_id,
            baseline=config.baseline,
            provider=config.provider,
            model_id=config.model_id,
            persona_id=config.persona_id,
            logs_root=logs_root,
        )
    
    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for provider from config or environment."""
        if provider == "gemini":
            return self.config.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        elif provider == "openai":
            return self.config.openai_api_key or os.environ.get("OPENAI_API_KEY")
        elif provider == "anthropic":
            return self.config.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        return None
    
    def _create_user_context(self, profile_df, summary_df) -> str:
        """Create user context string from profile and summary data.
        
        This context is passed to agents (like Health Coach) so they know
        about the user without having to ask.
        """
        context_parts = ["[USER PROFILE]"]
        
        # Extract profile information
        if profile_df is not None and not profile_df.empty:
            row = profile_df.iloc[0]
            if 'age' in profile_df.columns:
                context_parts.append(f"Age: {row['age']} years old")
            if 'gender' in profile_df.columns:
                context_parts.append(f"Sex: {row['gender']}")
            if 'height_cm' in profile_df.columns:
                context_parts.append(f"Height: {row['height_cm']} cm")
            if 'weight_kg' in profile_df.columns:
                context_parts.append(f"Weight: {row['weight_kg']} kg")
            if 'averageDailySteps' in profile_df.columns:
                context_parts.append(f"Average daily steps: {row['averageDailySteps']}")
            if 'cluster' in profile_df.columns:
                context_parts.append(f"Activity profile: {row['cluster']}")
        
        # Add summary of available data
        if summary_df is not None and not summary_df.empty:
            date_range = ""
            if 'datetime' in summary_df.columns or summary_df.index.name == 'datetime':
                try:
                    if summary_df.index.name == 'datetime':
                        dates = summary_df.index
                    else:
                        dates = summary_df['datetime']
                    min_date = dates.min().strftime('%Y-%m-%d')
                    max_date = dates.max().strftime('%Y-%m-%d')
                    date_range = f" from {min_date} to {max_date}"
                except:
                    pass
            context_parts.append(f"\n[AVAILABLE DATA]")
            context_parts.append(f"Health data: {len(summary_df)} days of records{date_range}")
        
        return "\n".join(context_parts)
    
    def _initialize_agent(self):
        """Initialize the appropriate agent based on baseline selection."""
        if self._initialized:
            return
        
        import traceback
        
        try:
            api_key = self._get_api_key(self.config.provider)
            if not api_key:
                raise ValueError(f"No API key found for {self.config.provider}")
            
            tavily_key = self.config.tavily_api_key or os.environ.get("TAVILY_API_KEY")
            
            # Get persona data directory
            from pathlib import Path
            data_dir = Path(__file__).parent.parent / "data" / self.config.persona_id
            print(f"Loading data from {data_dir}")
            
            # Create settings with correct data directory
            settings = Settings(
                data_dir=data_dir,
                llm_backend=self.config.provider,
                model_name=self.config.model_id,
                gemini_api_key=api_key if self.config.provider == "gemini" else None,
                openai_api_key=api_key if self.config.provider == "openai" else None,
                anthropic_api_key=api_key if self.config.provider == "anthropic" else None,
                tavily_api_key=tavily_key,
            )
            
            # Initialize the appropriate baseline
            if self.config.baseline == "pha":
                from pha.agents import create_orchestrator
                from pha.agents import DataScienceAgent, DomainExpertAgent, HealthCoachAgent
                from pha.utils import load_persona
                
                # Load persona data for context
                summary_df, activities_df, profile_df, population_df = load_persona(settings=settings)
                
                # Create user context string from profile
                user_context = self._create_user_context(profile_df, summary_df)
                
                # Get provider and model info
                provider = self.config.provider
                model_id = self.config.model_id
                
                # Create individual agents with multi-provider support
                ds_agent = DataScienceAgent(settings=settings)
                ds_agent.configure(
                    api_key=api_key,
                    provider=provider,
                    model_name=model_id,
                )
                
                # DomainExpertAgent ReAct now supports Gemini, OpenAI, and
                # Anthropic. The Anthropic path uses our OneTwo-compatible
                # AnthropicAPI wrapper (pha/llm/onetwo_anthropic.py).
                de_agent = DomainExpertAgent(tavily_api_key=tavily_key)
                de_agent.get_agent(
                    api_key=api_key,
                    provider=provider,
                    model_name=model_id,
                )
                # Pass user health data to domain expert
                # Combine profile and summary data for comprehensive context
                health_data_md = f"## User Profile\n{profile_df.to_string()}\n\n## Recent Health Summary\n{summary_df.head(30).to_string()}"
                de_agent.set_user_health_data(health_data_md)
                
                coach_agent = HealthCoachAgent(simple_mode=True)
                coach_agent.configure(
                    api_key=api_key,
                    provider=provider,
                    model_name=model_id,
                )
                # Initialize coach with user context so it knows about the user
                coach_agent.initialize_context(user_context)
                
                # Create orchestrator with agents
                self.agent = create_orchestrator(
                    api_key=api_key,
                    provider=provider,
                    model_name=model_id,
                    data_science_agent=ds_agent,
                    domain_expert_agent=de_agent,
                    health_coach_agent=coach_agent,
                )
                # Log version for debugging
                try:
                    from pha.agents.orchestrator import __version__ as orch_version
                    print(f"[API] PHA orchestrator version: {orch_version}")
                except ImportError:
                    pass
            elif self.config.baseline == "parallel":
                from pha.agents import create_parallel_baseline
                self.agent = create_parallel_baseline(
                    api_key=api_key,
                    provider=self.config.provider,
                    model_name=self.config.model_id,
                    tavily_api_key=tavily_key,
                    settings=settings,
                )
            elif self.config.baseline == "phia":
                from pha.agents import (
                    create_phia_baseline, 
                    is_onetwo_available, 
                    is_onetwo_openai_available,
                    get_onetwo_import_error,
                )
                
                # PHIA baseline uses OneTwo which supports Gemini and OpenAI
                if not is_onetwo_available():
                    import_error = get_onetwo_import_error() or "Unknown error"
                    raise ValueError(
                        f"PHIA baseline requires onetwo. Import error: {import_error}\n"
                        "Run setup.sh or install with: pip install --no-deps git+https://github.com/google-deepmind/onetwo"
                    )
                
                # Check provider support
                # PHIA now supports OpenAI with tool call format transformation
                # (see _patch_openai_for_compatibility in phia_baseline.py)
                if self.config.provider == "anthropic":
                    raise ValueError(
                        f"PHIA baseline does not support Anthropic models (no OneTwo backend). "
                        f"Please select Gemini or OpenAI, or use the 'pha' or 'parallel' baseline."
                    )
                elif self.config.provider not in ("gemini", "openai"):
                    raise ValueError(
                        f"PHIA baseline only supports Gemini and OpenAI models. "
                        f"Please select a supported provider or use the 'pha' or 'parallel' baseline."
                    )
                
                self.agent = create_phia_baseline(
                    api_key=api_key,
                    provider=self.config.provider,
                    tavily_api_key=tavily_key,
                    settings=settings,
                    few_shot_dir="few_shots/phia",
                    model_name=self.config.model_id,
                )
                # Log version for debugging
                try:
                    from pha.agents.phia_baseline import __version__ as phia_version
                    print(f"[API] PHIA baseline version: {phia_version}")
                except ImportError:
                    pass
            else:
                raise ValueError(f"Unknown baseline: {self.config.baseline}")
            
            self._initialized = True
            
        except Exception as e:
            print("=" * 60)
            print("ERROR in _initialize_agent:")
            traceback.print_exc()
            print("=" * 60)
            raise
    
    async def send_message(self, user_message: str) -> str:
        """Send a message and get a response."""
        import traceback
        
        try:
            self._initialize_agent()
        except Exception as e:
            traceback.print_exc()
            raise
        
        self.last_activity = datetime.now()
        
        # Add user message to history
        self.messages.append(ChatMessage(
            role="user",
            content=user_message,
            timestamp=datetime.now().isoformat(),
        ))
        
        # Begin logging this turn — capture all stdout/stderr
        capture = self.logger.begin_turn(user_message)
        
        # Get response from agent (run in thread pool to not block)
        start_time = datetime.now()
        try:
            with capture:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    self.agent.respond,
                    user_message,
                )
        except Exception as e:
            processing_ms = (datetime.now() - start_time).total_seconds() * 1000
            # Still log the failed turn
            self.logger.end_turn(
                assistant_response=f"[ERROR] {e}",
                processing_time_ms=processing_ms,
                agents_called=list(request_stats.agents_called),
                tools_used=dict(request_stats.tools_used),
                errors=request_stats.errors + 1,
            )
            print("=" * 60)
            print("ERROR in agent.respond:")
            traceback.print_exc()
            print("=" * 60)
            raise
        
        processing_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Finalize the turn log with stats from request_stats
        self.logger.end_turn(
            assistant_response=response,
            processing_time_ms=processing_ms,
            agents_called=list(request_stats.agents_called),
            tools_used=dict(request_stats.tools_used),
            errors=request_stats.errors,
        )
        
        # Add assistant message to history
        self.messages.append(ChatMessage(
            role="assistant",
            content=response,
            timestamp=datetime.now().isoformat(),
        ))
        
        return response
    
    def get_state(self) -> SessionState:
        """Get current session state."""
        return SessionState(
            session_id=self.session_id,
            config=self.config,
            message_count=len(self.messages),
            created_at=self.created_at.isoformat(),
            last_activity=self.last_activity.isoformat(),
            status="ready" if self._initialized else "pending",
        )


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "PHA Portal API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get all configuration options for the portal splash screen."""
    
    # Get models for each provider
    models = {
        "gemini": [
            ModelOption(
                id=m.id,
                name=m.name,
                provider=m.provider,
                description=m.description,
            )
            for m in KNOWN_GEMINI_MODELS
        ],
        "openai": [
            ModelOption(
                id=m.id,
                name=m.name,
                provider=m.provider,
                description=m.description,
            )
            for m in KNOWN_OPENAI_MODELS
        ],
        "anthropic": [
            ModelOption(
                id=m.id,
                name=m.name,
                provider=m.provider,
                description=m.description,
            )
            for m in KNOWN_ANTHROPIC_MODELS
        ],
    }
    
    # Get personas
    personas_raw = list_personas()
    personas = [
        PersonaOption(
            id=p["id"],
            name=p["name"],
            description=p.get("description", ""),
            demographics=p.get("demographics"),
            data_summary=p.get("data_summary"),
        )
        for p in personas_raw
    ]
    
    # Define baselines

    if GLOBAL_USER_STUDY_FLAG:
        baselines = [
            BaselineOption(
                id="pha",
                name="Agent A",
                description="Health insights assistant",
                architecture="System A",
            ),
            BaselineOption(
                id="parallel",
                name="Agent B",
                description="Health insights assistant",
                architecture="System B",
            ),
            BaselineOption(
                id="phia",
                name="Agent C",
                description="Health insights assistant",
                architecture="System C",
            ),
        ]
    else:
        baselines = [
            BaselineOption(
                id="pha",
                name="PHA (Orchestrated)",
                description="Full multi-agent system with intelligent orchestration",
                architecture="3 specialists + orchestrator",
            ),
            BaselineOption(
                id="parallel",
                name="Parallel (No Orchestration)",
                description="All agents run in parallel, responses synthesized",
                architecture="3 specialists, no routing",
            ),
            BaselineOption(
                id="phia",
                name="PHIA (Single Agent)",
                description="Original single ReAct agent baseline",
                architecture="1 generalist agent",
            ),
        ]
    
    return ConfigResponse(
        providers=["gemini", "openai", "anthropic"],
        models=models,
        personas=personas,
        baselines=baselines,
        defaults={
            "provider": "gemini",
            "model": get_default_model("gemini"),
            "baseline": "pha",
            "persona": get_default_persona(),
        },
        user_study_mode=GLOBAL_USER_STUDY_FLAG,
    )


@app.get("/user-study-defaults")
async def get_user_study_defaults():
    """Return provider and model defaults for user study mode. API keys are never exposed."""
    if not GLOBAL_USER_STUDY_FLAG:
        return {}
    data = _load_user_study_defaults()
    if not data:
        return {"provider": "gemini", "model": "models/gemini-3.1-flash-lite-preview"}
    return {
        "provider": data.get("provider", "gemini"),
        "model": data.get("model", "models/gemini-3.1-flash-lite-preview"),
    }


class ModelsRequest(BaseModel):
    """Request for fetching dynamic models."""
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None


class ModelsResponse(BaseModel):
    """Response with dynamically fetched models."""
    gemini: List[ModelOption]
    openai: List[ModelOption]
    anthropic: List[ModelOption]


@app.post("/models", response_model=ModelsResponse)
async def get_models(request: ModelsRequest):
    """Fetch available models dynamically using provided API keys.
    
    This endpoint discovers actual available models from each provider,
    returning more accurate model lists than the static defaults.
    """
    # Get curated lists annotated with live availability per the supplied keys.
    gemini_models = get_gemini_models(request.gemini_api_key)
    openai_models = get_openai_models(request.openai_api_key)
    anthropic_models = get_anthropic_models(request.anthropic_api_key)

    def _to_option(m):
        return ModelOption(
            id=m.id,
            name=m.name,
            provider=m.provider,
            description=m.description,
            available=m.available,
        )

    return ModelsResponse(
        gemini=[_to_option(m) for m in gemini_models],
        openai=[_to_option(m) for m in openai_models],
        anthropic=[_to_option(m) for m in anthropic_models],
    )


@app.post("/sessions", response_model=SessionResponse)
async def create_session(config: SessionConfig):
    """Create a new chat session with the specified configuration."""
    session_id = str(uuid.uuid4())[:8]
    
    # In user study mode, inject provider/model/api_keys from config file
    if GLOBAL_USER_STUDY_FLAG:
        defaults = _load_user_study_defaults()
        if not defaults:
            raise HTTPException(
                status_code=500,
                detail="config/user_study_defaults.json not found. Copy from user_study_defaults.example.json and fill in api_keys.",
            )
        updates = {
            "provider": defaults.get("provider", config.provider),
            "model_id": defaults.get("model", config.model_id),
        }
        api_keys = defaults.get("api_keys", {}) or {}
        provider = updates["provider"]
        key = api_keys.get(provider) or api_keys.get("gemini")
        if not key:
            raise HTTPException(
                status_code=500,
                detail=f"API key for provider '{provider}' not set in config/user_study_defaults.json",
            )
        if provider == "gemini":
            updates["gemini_api_key"] = key
        elif provider == "openai":
            updates["openai_api_key"] = key
        elif provider == "anthropic":
            updates["anthropic_api_key"] = key
        if api_keys.get("tavily"):
            updates["tavily_api_key"] = api_keys["tavily"]
        config = config.model_copy(update=updates)
    
    # Validate persona exists
    try:
        # Just check it exists, don't load yet
        from pha.utils.personas import get_persona_ids
        if config.persona_id not in get_persona_ids():
            raise HTTPException(
                status_code=400,
                detail=f"Persona '{config.persona_id}' not found",
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Create session
    session = ChatSession(session_id, config)
    sessions[session_id] = session
    
    return SessionResponse(
        session_id=session_id,
        config=config,
        created_at=session.created_at.isoformat(),
        status="ready",
    )


@app.get("/sessions/{session_id}", response_model=SessionState)
async def get_session(session_id: str):
    """Get the current state of a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return sessions[session_id].get_state()


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Close the session logger (writes final summary)
    session = sessions[session_id]
    if hasattr(session, 'logger') and session.logger:
        session.logger.close()
    
    del sessions[session_id]
    return {"status": "deleted", "session_id": session_id}


@app.get("/sessions/{session_id}/messages", response_model=List[ChatMessage])
async def get_messages(session_id: str):
    """Get all messages in a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return sessions[session_id].messages


@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def send_chat_message(session_id: str, request: ChatRequest):
    """Send a message and get a response."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    start_time = datetime.now()
    try:
        response_text = await session.send_message(request.message)
    except Exception as e:
        import traceback
        print("=" * 60)
        print("ERROR in send_chat_message:")
        traceback.print_exc()
        print("=" * 60)
        raise HTTPException(status_code=500, detail=str(e))
    
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return ChatResponse(
        message=ChatMessage(
            role="assistant",
            content=response_text,
            timestamp=datetime.now().isoformat(),
        ),
        session_id=session_id,
        processing_time_ms=processing_time,
    )


@app.post("/sessions/{session_id}/chat/stream")
async def stream_chat_message(session_id: str, request: ChatRequest):
    """Send a message and stream intermediate steps + final response via SSE.
    
    Events:
      event: step     — intermediate progress update
      event: result   — final response (terminal event)
      event: error    — error (terminal event)
    """
    import json
    import threading
    from fastapi.responses import StreamingResponse
    from pha.streaming import (
        create_emitter, destroy_emitter, bind_thread, unbind_thread,
        emit_result, emit_error, get_queue,
    )
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    emitter_id = create_emitter()
    
    # Agent work runs in a background thread
    def run_agent():
        bind_thread(emitter_id)
        start = datetime.now()
        
        # Begin logging this turn — capture all stdout/stderr from agent
        capture = session.logger.begin_turn(request.message)
        
        try:
            with capture:
                session._initialize_agent()
                session.last_activity = datetime.now()
                session.messages.append(ChatMessage(
                    role="user",
                    content=request.message,
                    timestamp=datetime.now().isoformat(),
                ))
                
                response_text = session.agent.respond(request.message)
                
                session.messages.append(ChatMessage(
                    role="assistant",
                    content=response_text,
                    timestamp=datetime.now().isoformat(),
                ))
            
            elapsed_ms = (datetime.now() - start).total_seconds() * 1000
            
            # Finalize the turn log with stats
            session.logger.end_turn(
                assistant_response=response_text,
                processing_time_ms=elapsed_ms,
                agents_called=list(request_stats.agents_called),
                tools_used=dict(request_stats.tools_used),
                errors=request_stats.errors,
            )
            
            emit_result(response_text, elapsed_ms)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            elapsed_ms = (datetime.now() - start).total_seconds() * 1000
            # Log the failed turn
            session.logger.end_turn(
                assistant_response=f"[ERROR] {e}",
                processing_time_ms=elapsed_ms,
                agents_called=list(request_stats.agents_called),
                tools_used=dict(request_stats.tools_used),
                errors=request_stats.errors + 1,
            )
            emit_error(str(e))
        finally:
            unbind_thread()
    
    agent_thread = threading.Thread(target=run_agent, daemon=True)
    agent_thread.start()
    
    # Async SSE generator — polls the queue without blocking the event loop
    async def event_stream():
        from queue import Empty
        queue = get_queue(emitter_id)
        if not queue:
            return
        try:
            while True:
                # Non-blocking poll + async sleep = no event-loop starvation
                try:
                    event = queue.get_nowait()
                except Empty:
                    await asyncio.sleep(0.15)
                    continue
                
                event_type = event.get("type", "step")
                yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"
                
                if event_type in ("result", "error"):
                    break
        finally:
            destroy_emitter(emitter_id)
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    return {
        "sessions": [
            {
                "session_id": sid,
                "baseline": s.config.baseline,
                "persona": s.config.persona_id,
                "message_count": len(s.messages),
                "created_at": s.created_at.isoformat(),
            }
            for sid, s in sessions.items()
        ],
        "count": len(sessions),
    }


# =============================================================================
# Utility Endpoints
# =============================================================================

@app.get("/models/{provider}")
async def get_models_for_provider(provider: str):
    """Get available models for a specific provider."""
    if provider not in ["gemini", "openai", "anthropic"]:
        raise HTTPException(status_code=400, detail="Invalid provider")
    
    models = get_available_models(provider)
    return {
        "provider": provider,
        "models": [
            {"id": m.id, "name": m.name, "description": m.description}
            for m in models
        ],
    }


@app.get("/personas")
async def get_personas():
    """Get available personas."""
    return {"personas": list_personas()}


@app.get("/personas/{persona_id}")
async def get_persona_detail(persona_id: str):
    """Get detailed info about a specific persona."""
    personas = list_personas()
    for p in personas:
        if p["id"] == persona_id:
            return p
    raise HTTPException(status_code=404, detail="Persona not found")

# =============================================================================
# Run with: uvicorn api.main:app --reload
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

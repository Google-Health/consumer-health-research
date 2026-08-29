"""Streaming step emitter for real-time progress updates.

This module provides a thread-safe mechanism for agents to emit
intermediate steps that can be streamed to the frontend via SSE.

Usage in agents:
    from pha.streaming import emit_step
    emit_step("DS", "Generating analysis approach...")
    emit_step("DS", "Executing code...", detail="avg = df['sleep'].mean()")

Usage in API (SSE endpoint):
    from pha.streaming import create_emitter, get_events, destroy_emitter
    
    emitter_id = create_emitter()
    # ... run agent in thread ...
    for event in get_events(emitter_id, timeout=1.0):
        yield f"event: step\\ndata: {json.dumps(event)}\\n\\n"
    destroy_emitter(emitter_id)
"""

import threading
import json
import uuid
from datetime import datetime
from queue import Queue, Empty
from typing import Optional, Dict, Any, Generator


# =============================================================================
# Thread-safe emitter registry
# =============================================================================

# Map thread_id -> emitter_id for auto-routing from agent code
_thread_to_emitter: Dict[int, str] = {}

# Map emitter_id -> Queue for event storage
_emitter_queues: Dict[str, Queue] = {}

# Lock for registry operations
_registry_lock = threading.Lock()


def create_emitter() -> str:
    """Create a new emitter and return its ID.
    
    Call this before starting the agent thread.
    """
    emitter_id = str(uuid.uuid4())[:8]
    with _registry_lock:
        _emitter_queues[emitter_id] = Queue()
    return emitter_id


def bind_thread(emitter_id: str) -> None:
    """Bind the current thread to an emitter.
    
    Call this at the start of the agent thread so emit_step()
    knows where to route events.
    """
    tid = threading.get_ident()
    with _registry_lock:
        _thread_to_emitter[tid] = emitter_id


def unbind_thread() -> None:
    """Unbind the current thread from its emitter."""
    tid = threading.get_ident()
    with _registry_lock:
        _thread_to_emitter.pop(tid, None)


def destroy_emitter(emitter_id: str) -> None:
    """Clean up an emitter and its queue."""
    with _registry_lock:
        _emitter_queues.pop(emitter_id, None)
        # Also clean up any thread bindings pointing to this emitter
        to_remove = [tid for tid, eid in _thread_to_emitter.items() if eid == emitter_id]
        for tid in to_remove:
            del _thread_to_emitter[tid]


def get_queue(emitter_id: str) -> Optional[Queue]:
    """Get the raw queue for an emitter (for async polling)."""
    return _emitter_queues.get(emitter_id)


def emit_step(tag: str, message: str, detail: str = "") -> None:
    """Emit a streaming step from agent code.
    
    This is the main function agents call. It:
    1. Prints to stdout (for terminal logging, always)
    2. Pushes to the SSE queue if an emitter is bound to this thread
    
    Args:
        tag: Short label like "DS", "Coach", "Orch", "DE", "Parallel", "PHIA"
        message: Human-readable step description
        detail: Optional extra detail (code snippet, data preview, etc.)
    """
    ts = datetime.now().strftime("%H:%M:%S")
    
    # Always print to terminal
    if detail:
        print(f"[{tag}:{ts}] {message} | {detail[:150]}")
    else:
        print(f"[{tag}:{ts}] {message}")
    
    # Route to SSE queue if this thread has a bound emitter
    tid = threading.get_ident()
    emitter_id = _thread_to_emitter.get(tid)
    if emitter_id:
        queue = _emitter_queues.get(emitter_id)
        if queue:
            queue.put({
                "type": "step",
                "tag": tag,
                "message": message,
                "detail": detail[:500] if detail else "",
                "timestamp": ts,
            })


def emit_result(content: str, processing_time_ms: float) -> None:
    """Emit the final result event."""
    tid = threading.get_ident()
    emitter_id = _thread_to_emitter.get(tid)
    if emitter_id:
        queue = _emitter_queues.get(emitter_id)
        if queue:
            queue.put({
                "type": "result",
                "content": content,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })


def emit_error(error: str) -> None:
    """Emit an error event."""
    tid = threading.get_ident()
    emitter_id = _thread_to_emitter.get(tid)
    if emitter_id:
        queue = _emitter_queues.get(emitter_id)
        if queue:
            queue.put({
                "type": "error",
                "detail": error,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })


def get_events(emitter_id: str, timeout: float = 1.0) -> Generator[Dict[str, Any], None, None]:
    """Yield events from an emitter's queue.
    
    Blocks up to `timeout` seconds waiting for each event.
    Stops when a 'result' or 'error' event is received.
    
    Args:
        emitter_id: The emitter to read from.
        timeout: Seconds to wait for each event before yielding a keepalive.
    """
    queue = _emitter_queues.get(emitter_id)
    if not queue:
        return
    
    while True:
        try:
            event = queue.get(timeout=timeout)
            yield event
            if event.get("type") in ("result", "error"):
                return
        except Empty:
            # Yield a keepalive so the connection doesn't drop
            yield {"type": "keepalive", "timestamp": datetime.now().strftime("%H:%M:%S")}

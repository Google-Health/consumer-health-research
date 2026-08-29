import React, { useState, useEffect, useMemo, useRef } from 'react'
import './SplashScreen.css'

const API_BASE = '/api'

// localStorage keys
const STORAGE_KEYS = {
  gemini: 'pha_gemini_key',
  openai: 'pha_openai_key',
  anthropic: 'pha_anthropic_key',
  tavily: 'pha_tavily_key',
  provider: 'pha_provider',
  baseline: 'pha_baseline',
  persona: 'pha_persona',
  model: 'pha_model',
  userId: 'pha_user_id',
}

function SplashScreen({ config, onStart, loading, error }) {
  const userStudyMode = config?.user_study_mode ?? false

  // User study defaults fetched from backend (config/user_study_defaults.json)
  const [userStudyDefaults, setUserStudyDefaults] = useState(null)

  // Load saved values from localStorage
  const savedProvider = localStorage.getItem(STORAGE_KEYS.provider) || config?.defaults?.provider || 'gemini'
  const savedBaseline = localStorage.getItem(STORAGE_KEYS.baseline) || config?.defaults?.baseline || 'pha'
  const savedPersona = localStorage.getItem(STORAGE_KEYS.persona) || config?.defaults?.persona || 'sample'
  const savedModel = localStorage.getItem(STORAGE_KEYS.model) || config?.defaults?.model || ''
  const savedUserId = localStorage.getItem(STORAGE_KEYS.userId) || 'test'
  
  const [provider, setProvider] = useState(savedProvider)
  const [modelId, setModelId] = useState(savedModel)
  const [baseline, setBaseline] = useState(savedBaseline)
  const [persona, setPersona] = useState(savedPersona)
  const [userId, setUserId] = useState(savedUserId)
  const [apiKeys, setApiKeys] = useState({
    gemini: localStorage.getItem(STORAGE_KEYS.gemini) || '',
    openai: localStorage.getItem(STORAGE_KEYS.openai) || '',
    anthropic: localStorage.getItem(STORAGE_KEYS.anthropic) || '',
    tavily: localStorage.getItem(STORAGE_KEYS.tavily) || '',
  })
  const [validationError, setValidationError] = useState(null)
  const [dynamicModels, setDynamicModels] = useState(null)
  const [checkingAvailability, setCheckingAvailability] = useState(false)
  const isFirstAvailabilityRun = useRef(true)

  // Fetch user study defaults when in user study mode
  useEffect(() => {
    if (!userStudyMode) return
    const fetchDefaults = async () => {
      try {
        const res = await fetch(`${API_BASE}/user-study-defaults`)
        if (res.ok) {
          const data = await res.json()
          setUserStudyDefaults(data)
          if (data.provider) setProvider(data.provider)
          if (data.model) setModelId(data.model)
        }
      } catch (err) {
        console.warn('Could not fetch user study defaults:', err)
      }
    }
    fetchDefaults()
  }, [userStudyMode])

  // Re-check model availability whenever any API key changes.
  // - On first mount: fire immediately so saved keys are validated up front.
  // - On subsequent key changes: debounce 500ms so we don't fetch on every keystroke.
  // - If no keys are entered, do nothing (keep curated list with all-available defaults).
  useEffect(() => {
    const hasAnyKey = !!(apiKeys.gemini || apiKeys.openai || apiKeys.anthropic)
    if (!hasAnyKey) {
      setDynamicModels(null)
      setCheckingAvailability(false)
      return
    }

    const fetchModels = async () => {
      setCheckingAvailability(true)
      try {
        const response = await fetch(`${API_BASE}/models`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            gemini_api_key: apiKeys.gemini || null,
            openai_api_key: apiKeys.openai || null,
            anthropic_api_key: apiKeys.anthropic || null,
          }),
        })
        if (response.ok) {
          const data = await response.json()
          setDynamicModels(data)
        }
      } catch (err) {
        console.log('Could not fetch dynamic models:', err)
      } finally {
        setCheckingAvailability(false)
      }
    }

    if (isFirstAvailabilityRun.current) {
      isFirstAvailabilityRun.current = false
      fetchModels()
      return
    }

    const timer = setTimeout(fetchModels, 500)
    return () => clearTimeout(timer)
  }, [userStudyMode, apiKeys.gemini, apiKeys.openai, apiKeys.anthropic])

  // Use dynamic models if available, otherwise fall back to static config
  const effectiveConfig = useMemo(() => ({
    ...config,
    models: dynamicModels || config?.models,
  }), [config, dynamicModels])

  // Save API keys to localStorage whenever they change
  useEffect(() => {
    if (apiKeys.gemini) localStorage.setItem(STORAGE_KEYS.gemini, apiKeys.gemini)
    if (apiKeys.openai) localStorage.setItem(STORAGE_KEYS.openai, apiKeys.openai)
    if (apiKeys.anthropic) localStorage.setItem(STORAGE_KEYS.anthropic, apiKeys.anthropic)
    if (apiKeys.tavily) localStorage.setItem(STORAGE_KEYS.tavily, apiKeys.tavily)
  }, [apiKeys])

  // Save preferences to localStorage
  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.provider, provider)
  }, [provider])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.baseline, baseline)
  }, [baseline])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.persona, persona)
  }, [persona])

  // Save userId to localStorage
  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.userId, userId)
  }, [userId])

  // Save model to localStorage
  useEffect(() => {
    if (modelId) {
      localStorage.setItem(STORAGE_KEYS.model, modelId)
    }
  }, [modelId])

  // Initialize model when config loads or provider changes (skip in user study mode).
  // Preference order: stored model if available; first available model; first model.
  useEffect(() => {
    if (userStudyMode || !effectiveConfig?.models) return

    const providerModels = effectiveConfig.models[provider] || []
    if (providerModels.length === 0) return

    const isAvail = (m) => m.available !== false

    const storedModel = localStorage.getItem(STORAGE_KEYS.model)
    const stored = storedModel && providerModels.find(m => m.id === storedModel)
    if (stored && isAvail(stored)) {
      setModelId(stored.id)
      return
    }

    const firstAvailable = providerModels.find(isAvail)
    setModelId((firstAvailable || providerModels[0]).id)
  }, [userStudyMode, effectiveConfig, provider, baseline])

  // Update model when provider changes — pick first available, or first as fallback.
  const handleProviderChange = (newProvider) => {
    setProvider(newProvider)
    const providerModels = effectiveConfig?.models?.[newProvider] || []
    if (providerModels.length > 0) {
      const firstAvailable = providerModels.find(m => m.available !== false)
      setModelId((firstAvailable || providerModels[0]).id)
    }
    setValidationError(null)
  }

  // Handle baseline change - set appropriate defaults
  const handleBaselineChange = (newBaseline) => {
    setBaseline(newBaseline)
    setValidationError(null)
    
    // PHIA doesn't support Anthropic (no OneTwo backend)
    // If currently on Anthropic when switching to PHIA, switch to Gemini
    if (newBaseline === 'phia' && provider === 'anthropic') {
      setProvider('gemini')
      const geminiModels = effectiveConfig?.models?.gemini || []
      const pro25 = geminiModels.find(m => m.id.includes('gemini-2.5-pro'))
      if (pro25) {
        setModelId(pro25.id)
      } else if (geminiModels.length > 0) {
        setModelId(geminiModels[0].id)
      }
    }
  }

  // Determine which API key is required based on selections
   const getRequiredKeyType = () => {
    // Return the selected provider (PHIA now supports both Gemini and OpenAI)
    return provider
  }

  const requiredKeyType = getRequiredKeyType()
  const hasRequiredKey = userStudyMode ? true : (apiKeys[requiredKeyType]?.trim().length > 0)
  
  // Provider selection
  const providerDisabled = false
  
  // Get current effective provider (for model list)
  const effectiveProvider = provider

  const keyInfo = {
    gemini: { label: 'Gemini', placeholder: 'AIza...', url: 'https://aistudio.google.com/apikey' },
    openai: { label: 'OpenAI', placeholder: 'sk-...', url: 'https://platform.openai.com/api-keys' },
    anthropic: { label: 'Anthropic', placeholder: 'sk-ant-...', url: 'https://console.anthropic.com/' },
    tavily: { label: 'Tavily', placeholder: 'tvly-...', url: 'https://tavily.com' },
  }

  const handleSubmit = (e) => {
    e.preventDefault()

    if (!hasRequiredKey) {
      setValidationError(`${keyInfo[requiredKeyType].label} API key is required`)
      return
    }

    // Block submission if the selected model is known to be unavailable for
    // the supplied key. This prevents the user from kicking off a chat session
    // that will fail at the first LLM call.
    if (!userStudyMode) {
      const picked = currentModels.find(m => m.id === modelId)
      if (picked && picked.available === false) {
        setValidationError(
          `"${picked.name}" is not available for this ${effectiveProvider} key. Pick an enabled option.`
        )
        return
      }
    }

    setValidationError(null)

    // Build the session config
    const submittedProvider = userStudyMode ? (userStudyDefaults?.provider || provider) : provider
    const effectiveModelId = userStudyMode ? (userStudyDefaults?.model || modelId) : modelId
    const sessionConfig = {
      baseline,
      persona_id: persona,
      provider: submittedProvider,
      model_id: effectiveModelId,
      user_id: userId,
    }
    
    // API keys: in user study mode backend injects from config file; otherwise from form
    if (!userStudyMode) {
      if (apiKeys.gemini) sessionConfig.gemini_api_key = apiKeys.gemini
      if (apiKeys.openai) sessionConfig.openai_api_key = apiKeys.openai
      if (apiKeys.anthropic) sessionConfig.anthropic_api_key = apiKeys.anthropic
      if (apiKeys.tavily) sessionConfig.tavily_api_key = apiKeys.tavily
    }
    
    onStart(sessionConfig)
  }

  const currentModels = effectiveConfig?.models?.[effectiveProvider] || []
  const selectedBaseline = config?.baselines?.find(b => b.id === baseline)
  const selectedPersona = config?.personas?.find(p => p.id === persona)

  return (
    <div className="splash">
      <div className="splash-container">
        <header className="splash-header">
          <h1>🏥 PHA Portal</h1>
          <p className="subtitle">Personal Health Agent</p>
        </header>

        <div className="pha-disclaimer" role="note">
          <span className="pha-disclaimer-label">NOTE:</span>
          This research portal (i) is not intended to be a medical device; and (ii) is not intended for clinical use of any kind, including but not limited to diagnosis, prognosis, or treatment recommendations.
        </div>

        <form onSubmit={handleSubmit} className="config-form">
          
          {/* User ID Entry */}
          <section className="config-section user-id-section">
            <h2>User ID</h2>
            <input
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder="Enter your user ID"
              className="config-input user-id-input"
            />
          </section>

          {/* Step 1: Baseline Selection */}
          <section className="config-section">
            <h2>1. Select System Architecture</h2>
            <div className="baseline-grid">
              {config?.baselines?.map(b => (
                <div
                  key={b.id}
                  className={`baseline-card ${baseline === b.id ? 'active' : ''}`}
                  onClick={() => handleBaselineChange(b.id)}
                >
                  <div className="baseline-header">
                    <input
                      type="radio"
                      name="baseline"
                      value={b.id}
                      checked={baseline === b.id}
                      onChange={() => handleBaselineChange(b.id)}
                    />
                    <span className="baseline-name">{b.name}</span>
                  </div>
                  <p className="baseline-desc">{b.description}</p>
                  <span className="baseline-arch">{b.architecture}</span>
                </div>
              ))}
            </div>
          </section>

          {/* Step 2: Provider Selection (hidden in user study mode) */}
          {!userStudyMode && (
          <section className="config-section">
            <h2>
              2. Select Provider
            </h2>
            <div className="provider-grid">
              {config?.providers?.map(p => {
                const isAnthropic = p === 'anthropic'
                // Anthropic is supported by PHA and Parallel baselines (multimodel
                // backend), but not by PHIA (which uses OneTwo / no Anthropic backend).
                const isDisabled = isAnthropic && baseline === 'phia'

                let disabledReason = ''
                if (isDisabled) {
                  disabledReason = 'Anthropic is not supported by the PHIA baseline (OneTwo has no Anthropic backend). Use PHA or Parallel.'
                }
                
                return (
                  <div key={p} className="provider-wrapper">
                    {isDisabled && disabledReason && (
                      <p className="provider-unavailable-note">
                        {disabledReason}
                      </p>
                    )}
                    <button
                      type="button"
                      className={`provider-btn ${effectiveProvider === p ? 'active' : ''} ${isDisabled ? 'coming-soon' : ''}`}
                      onClick={() => !isDisabled && handleProviderChange(p)}
                      disabled={isDisabled}
                    >
                      <span className="provider-icon">
                        {p === 'gemini' && '🔷'}
                        {p === 'openai' && '🟢'}
                        {p === 'anthropic' && '🟠'}
                      </span>
                      <span className="provider-name">
                        {p.charAt(0).toUpperCase() + p.slice(1)}
                      </span>
                    </button>
                  </div>
                )
              })}
            </div>
          </section>
          )}

          {/* Step 3: Model Selection (hidden in user study mode) */}
          {!userStudyMode && (
          <section className="config-section">
            <h2>3. Select Model</h2>
            <select
              value={modelId}
              onChange={(e) => {
                const next = e.target.value
                const picked = currentModels.find(m => m.id === next)
                if (picked && picked.available === false) return
                setModelId(next)
              }}
              className="config-select"
            >
              {currentModels.map(m => {
                const isUnavailable = m.available === false
                return (
                  <option
                    key={m.id}
                    value={m.id}
                    disabled={isUnavailable}
                    style={isUnavailable ? { color: '#94a3b8' } : undefined}
                  >
                    {m.name}{isUnavailable ? ' — not available for this key' : ''}
                  </option>
                )
              })}
            </select>
            {(() => {
              const hasProviderKey = !!apiKeys[effectiveProvider]
              if (!hasProviderKey) {
                return (
                  <p className="model-availability-note">
                    Enter a {effectiveProvider.charAt(0).toUpperCase() + effectiveProvider.slice(1)} key below to verify which models your account can access.
                  </p>
                )
              }
              if (checkingAvailability) {
                return (
                  <p className="model-availability-note checking">
                    Verifying availability with {effectiveProvider}…
                  </p>
                )
              }
              const total = currentModels.length
              const available = currentModels.filter(m => m.available !== false).length
              const allOk = available === total
              return (
                <p className={`model-availability-note ${allOk ? 'ok' : 'partial'}`}>
                  {allOk
                    ? `✓ All ${total} models available for this ${effectiveProvider} key.`
                    : `${available} of ${total} models available for this ${effectiveProvider} key.`}
                </p>
              )
            })()}
          </section>
          )}

          {/* Step 4: Persona Selection */}
          <section className="config-section">
            <h2>{userStudyMode ? '2. Select User Profile' : '4. Select User Profile'}</h2>
            <select
              value={persona}
              onChange={(e) => setPersona(e.target.value)}
              className="config-select"
            >
              {config?.personas?.map(p => (
                <option key={p.id} value={p.id}>
                  {p.name} {p.description && `- ${p.description}`}
                </option>
              ))}
            </select>
            {selectedPersona?.demographics && (
              <div className="persona-details">
                {selectedPersona.demographics.age && <span>Age: {selectedPersona.demographics.age}</span>}
                {selectedPersona.demographics.gender && <span>Gender: {selectedPersona.demographics.gender}</span>}
                {selectedPersona.data_summary?.days && <span>{selectedPersona.data_summary.days} days of data</span>}
              </div>
            )}
          </section>

          {/* Step 5: API Keys (hidden in user study mode) */}
          {!userStudyMode && (
          <section className="config-section">
            <h2>5. API Keys</h2>
            
            {/* Required API Key */}
            <div className="api-key-group">
              <h3 className="api-key-title">
                <span className="required-dot"></span>
                Required: {keyInfo[requiredKeyType].label} API Key
              </h3>
              <p className="api-key-reason">
                {`Required for ${provider.charAt(0).toUpperCase() + provider.slice(1)} models`}
              </p>
              <input
                type="password"
                placeholder={keyInfo[requiredKeyType].placeholder}
                value={apiKeys[requiredKeyType]}
                onChange={(e) => setApiKeys({...apiKeys, [requiredKeyType]: e.target.value})}
                className={`api-key-input ${validationError && !hasRequiredKey ? 'error' : ''}`}
              />
              <a 
                href={keyInfo[requiredKeyType].url} 
                target="_blank" 
                rel="noopener noreferrer"
                className="api-key-link"
              >
                Get {keyInfo[requiredKeyType].label} API Key →
              </a>
            </div>
            
            {/* Optional Tavily Key */}
            <div className="api-key-group optional">
              <h3 className="api-key-title">
                <span className="optional-dot"></span>
                Optional: Tavily API Key
              </h3>
              <p className="api-key-reason">Enables web search for health information lookups</p>
              <input
                type="password"
                placeholder="tvly-..."
                value={apiKeys.tavily}
                onChange={(e) => setApiKeys({...apiKeys, tavily: e.target.value})}
                className="api-key-input"
              />
              <a 
                href="https://tavily.com" 
                target="_blank" 
                rel="noopener noreferrer"
                className="api-key-link"
              >
                Get Tavily API Key →
              </a>
            </div>
            
            <p className="api-env-hint">
              💡 Keys can also be set as environment variables on the server
              {(apiKeys.gemini || apiKeys.openai || apiKeys.anthropic || apiKeys.tavily) && (
                <span className="saved-indicator"> • ✓ Keys saved locally</span>
              )}
            </p>
          </section>
          )}

          {/* Error display */}
          {(validationError || error) && (
            <div className="config-error">
              ⚠️ {validationError || error}
            </div>
          )}

          {/* Start button */}
          <button type="submit" className="start-btn" disabled={loading}>
            {loading ? (
              <>
                <span className="btn-spinner"></span>
                Initializing...
              </>
            ) : (
              'Start Chat Session →'
            )}
          </button>

          {/* Summary */}
          <div className="config-summary">
            <span>{selectedBaseline?.name}</span>
            <span>•</span>
            <span>{modelId ? modelId.split('/').pop() : '—'}</span>
            <span>•</span>
            <span>{selectedPersona?.name}</span>
          </div>
        </form>

        <footer className="splash-footer">
          <p>PHA - A multi-agent system for personal health insights</p>
          <button
            type="button" 
            className="clear-data-btn"
            onClick={() => {
              Object.values(STORAGE_KEYS).forEach(key => localStorage.removeItem(key))
              setApiKeys({ gemini: '', openai: '', anthropic: '', tavily: '' })
              setProvider('gemini')
              setBaseline('pha')
              setPersona('sample')
              setUserId('test')
            }}
          >
            Clear saved data
          </button>
        </footer>
      </div>
    </div>
  )
}

export default SplashScreen

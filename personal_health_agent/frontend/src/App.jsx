import React, { useState, useEffect } from 'react'
import SplashScreen from './components/SplashScreen'
import ChatInterface from './components/ChatInterface'
import './App.css'

// API base URL - use proxy in dev
const API_BASE = '/api'
const ACTIVE_SESSION_KEY = 'pha_active_session'

function App() {
  const [config, setConfig] = useState(null)
  const [session, setSession] = useState(null)
  const [userId, setUserId] = useState('test')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Fetch configuration on mount
  useEffect(() => {
    fetchConfig()
  }, [])

  const clearActiveSession = () => {
    localStorage.removeItem(ACTIVE_SESSION_KEY)
  }

  const saveActiveSession = (sessionData, activeUserId) => {
    const payload = {
      session_id: sessionData.session_id,
      user_id: activeUserId || sessionData?.config?.user_id || 'test',
    }
    localStorage.setItem(ACTIVE_SESSION_KEY, JSON.stringify(payload))
  }

  const restoreSession = async (latestConfig) => {
    const raw = localStorage.getItem(ACTIVE_SESSION_KEY)
    if (!raw) return false

    let storedSession
    try {
      storedSession = JSON.parse(raw)
    } catch {
      clearActiveSession()
      return false
    }

    if (!storedSession?.session_id) {
      clearActiveSession()
      return false
    }

    try {
      const res = await fetch(`${API_BASE}/sessions/${storedSession.session_id}`)
      if (!res.ok) throw new Error('Stored session is unavailable')
      const restoredState = await res.json()

      // Validate that the restored session's model is still in the curated list
      // for its provider. If it isn't (because we updated the curated set since
      // the session was created), drop the session and force the user back to
      // the splash screen — otherwise they'd be silently using a stale model.
      const restoredProvider = restoredState.config?.provider
      const restoredModel = restoredState.config?.model_id
      const cfg = latestConfig || config
      const providerModels = (restoredProvider && cfg?.models?.[restoredProvider]) || []
      const restoredModelExists = providerModels.some(m => m.id === restoredModel)
      if (!restoredModelExists) {
        await fetch(`${API_BASE}/sessions/${storedSession.session_id}`, { method: 'DELETE' }).catch(() => {})
        clearActiveSession()
        return false
      }

      setSession({
        session_id: restoredState.session_id,
        config: restoredState.config,
        created_at: restoredState.created_at,
        status: restoredState.status,
      })
      setUserId(restoredState.config?.user_id || storedSession.user_id || 'test')
      return true
    } catch {
      clearActiveSession()
      return false
    }
  }

  const fetchConfig = async () => {
    try {
      setLoading(true)
      const res = await fetch(`${API_BASE}/config`)
      if (!res.ok) throw new Error('Failed to fetch config')
      const data = await res.json()
      setConfig(data)
      await restoreSession(data)
      setLoading(false)
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }

  const createSession = async (sessionConfig) => {
    try {
      setLoading(true)
      // Extract userId from sessionConfig and save it
      if (sessionConfig.user_id) {
        setUserId(sessionConfig.user_id)
      }
      const res = await fetch(`${API_BASE}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sessionConfig),
      })
      if (!res.ok) {
        const errData = await res.json()
        throw new Error(errData.detail || 'Failed to create session')
      }
      const data = await res.json()
      setSession(data)
      saveActiveSession(data, sessionConfig.user_id)
      setLoading(false)
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }

  const endSession = async () => {
    if (session) {
      try {
        await fetch(`${API_BASE}/sessions/${session.session_id}`, {
          method: 'DELETE',
        })
      } catch (err) {
        console.error('Error ending session:', err)
      }
    }
    setSession(null)
    clearActiveSession()
  }

  const handleRefresh = async () => {
    if (!session) return

    const sessionConfig = session.config
    setError(null)
    setLoading(true)

    try {
      try {
        await fetch(`${API_BASE}/sessions/${session.session_id}`, {
          method: 'DELETE',
        })
      } catch (err) {
        console.error('Error ending session during refresh:', err)
      }

      clearActiveSession()

      const res = await fetch(`${API_BASE}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sessionConfig),
      })
      if (!res.ok) {
        let detail = 'Failed to refresh session'
        try {
          const errData = await res.json()
          detail = errData.detail || detail
        } catch {
          // Keep default detail if response is not JSON.
        }
        throw new Error(detail)
      }

      const data = await res.json()
      setSession(data)
      setUserId(sessionConfig.user_id || data?.config?.user_id || 'test')
      saveActiveSession(data, sessionConfig.user_id)
    } catch (err) {
      setSession(null)
      setError(err.message)
      clearActiveSession()
    } finally {
      setLoading(false)
    }
  }

  // Loading state
  if (loading && !config) {
    return (
      <div className="app-loading">
        <div className="spinner"></div>
        <p>Loading PHA Portal...</p>
      </div>
    )
  }

  // Error state
  if (error && !config) {
    return (
      <div className="app-error">
        <h2>⚠️ Connection Error</h2>
        <p>{error}</p>
        <p className="hint">Make sure the API server is running:</p>
        <code>cd personal-health-agent-main && uvicorn api.main:app --reload</code>
        <button onClick={() => { setError(null); fetchConfig(); }}>
          Retry
        </button>
      </div>
    )
  }

  return (
    <div className="app">
      {!session ? (
        <SplashScreen
          config={config}
          onStart={createSession}
          loading={loading}
          error={error}
        />
      ) : (
        <ChatInterface
          session={session}
          apiBase={API_BASE}
          onEndSession={endSession}
          onRefresh={handleRefresh}
          refreshing={loading}
          userId={userId}
          userStudyMode={config?.user_study_mode || false}
        />
      )}
    </div>
  )
}

export default App

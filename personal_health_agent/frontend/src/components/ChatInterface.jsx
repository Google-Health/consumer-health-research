import React, { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './ChatInterface.css'


const TAG_LABELS = {
  Orch:     { label: 'Orchestrator',   icon: '🎯' },
  DS:       { label: 'Data Science',   icon: '📊' },
  Coach:    { label: 'Health Coach',   icon: '💬' },
  DE:       { label: 'Domain Expert',  icon: '🩺' },
  LLM:      { label: 'LLM Call',       icon: '🧠' },
  Parallel: { label: 'Parallel',       icon: '⚡' },
  PHIA:     { label: 'PHIA Agent',     icon: '🤖' },
}

/* ── Thinking indicator (Gemini-style) ──
   While active: shows spinner + current step message
   When done: green checkmark + summary, expandable to see all steps */
function ThinkingIndicator({ steps, isActive, elapsed }) {
  const [expanded, setExpanded] = useState(false)

  // For live display, skip LLM "Calling..." steps — show only what's happening
  const displaySteps = steps.filter(s => !s.message.startsWith('Calling '))
  const lastDisplayStep = displaySteps[displaySteps.length - 1]

  if (steps.length === 0 && isActive) {
    return (
      <div className="thinking-indicator active">
        <span className="thinking-spinner" />
        <span className="thinking-label">Thinking...</span>
      </div>
    )
  }

  // Nothing displayable yet (only LLM calls so far)
  if (!lastDisplayStep && isActive) {
    return (
      <div className="thinking-indicator active">
        <span className="thinking-spinner" />
        <span className="thinking-label">Thinking...</span>
      </div>
    )
  }

  if (!lastDisplayStep) return null

  const lastTag = TAG_LABELS[lastDisplayStep?.tag] || { label: lastDisplayStep?.tag, icon: '⚙️' }

  return (
    <div className={`thinking-indicator ${isActive ? 'active' : 'done'}`}>
      {/* Main row — clickable to expand when done */}
      <button className="thinking-row" onClick={() => !isActive && setExpanded(!expanded)}>
        {isActive ? (
          <>
            <span className="thinking-spinner" />
            <span className="thinking-label">
              {lastTag.icon} {lastDisplayStep.message}
            </span>
            <span className="thinking-elapsed">{elapsed}s</span>
          </>
        ) : (
          <>
            <span className="thinking-check">✓</span>
            <span className="thinking-label done-label">
              {lastTag.icon} {lastDisplayStep.message}
            </span>
            <span className="thinking-elapsed">{elapsed}s</span>
            {steps.length > 1 && (
              <span className={`thinking-chevron ${expanded ? 'open' : ''}`}>▾</span>
            )}
          </>
        )}
      </button>

      {/* Expanded steps list — shows ALL steps including LLM calls */}
      {expanded && !isActive && steps.length > 1 && (
        <div className="thinking-expanded">
          {steps.map((step, i) => {
            const tag = TAG_LABELS[step.tag] || { label: step.tag, icon: '⚙️' }
            return (
              <div key={i} className="thinking-exp-step">
                <span className="exp-icon">{tag.icon}</span>
                <span className="exp-msg">{step.message}</span>
                <span className="exp-time">{step.timestamp}</span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function ChatInterface({ session, apiBase, onEndSession, onRefresh, refreshing, userId, userStudyMode }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [thinkingSteps, setThinkingSteps] = useState([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const messagesEndRef = useRef(null)

  // Function to format timestamp for display in conversation log
  const formatTimestampForLog = (timestamp) => {
    if (!timestamp) return ''
    const date = new Date(timestamp)
    return date.toLocaleString()
  }

  // Function to save conversation log as txt file
  const saveConversationLog = async () => {
    // Format the conversation log
    let logContent = ''
    messages.forEach(msg => {
      const role = msg.role === 'user' ? '**User**' : '**Agent**'
      const timestamp = formatTimestampForLog(msg.timestamp)
      logContent += `${role} [${timestamp}]: ${msg.content}\n\n`
    })

    // Create filename: [user id]-[agent name]-[hh-mm]-[MM-DD-YYYY].txt
    const now = new Date()
    const hours = String(now.getHours()).padStart(2, '0')
    const minutes = String(now.getMinutes()).padStart(2, '0')
    const seconds = String(now.getSeconds()).padStart(2, '0')
    const month = String(now.getMonth() + 1).padStart(2, '0')
    const day = String(now.getDate()).padStart(2, '0')
    const year = now.getFullYear()
    
    const agentName = session.config.baseline
    const filename = `${userId}-${agentName}-${hours}-${minutes}-${seconds}-${month}-${day}-${year}.txt`

    // (raw_conversation.log + summary.json) handles all persistence.
  }

  // Handle end session - save log then call parent callback
  const handleEndSession = async () => {
    await saveConversationLog()
    onEndSession()
  }
  const timerRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, thinkingSteps])

  useEffect(() => {
    let isMounted = true

    const loadMessages = async () => {
      try {
        const res = await fetch(`${apiBase}/sessions/${session.session_id}/messages`)
        if (!res.ok) throw new Error('Failed to load session messages')
        const data = await res.json()

        if (!isMounted) return

        if (Array.isArray(data) && data.length > 0) {
          setMessages(data)
        } else {
          setMessages([{
            role: 'assistant',
            content: getWelcomeMessage(),
            timestamp: new Date().toISOString(),
          }])
        }
      } catch {
        if (!isMounted) return
        setMessages([{
          role: 'assistant',
          content: getWelcomeMessage(),
          timestamp: new Date().toISOString(),
        }])
      }
    }

    loadMessages()

    return () => {
      isMounted = false
    }
  }, [apiBase, session.session_id, userStudyMode])

  // Elapsed timer
  useEffect(() => {
    if (loading) {
      setElapsed(0)
      timerRef.current = setInterval(() => {
        setElapsed(prev => prev + 1)
      }, 1000)
    } else {
      clearInterval(timerRef.current)
    }
    return () => clearInterval(timerRef.current)
  }, [loading])

  const getWelcomeMessage = () => {
    // Use anonymous names (Agent A/B/C) in user study mode
    const baselineNames = userStudyMode
      ? {
          pha: 'Agent A',
          parallel: 'Agent B',
          phia: 'Agent C',
        }
      : {
          pha: 'PHA (Orchestrated Multi-Agent)',
          parallel: 'Parallel (Multi-Agent, No Orchestration)',
          phia: 'PHIA (Single Agent)',
        }
    return `👋 Hello! I'm your Personal Health Agent running on **${baselineNames[session.config.baseline]}**. How can I help you today?`
  }

  const sendMessage = useCallback(async (e) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMessage = {
      role: 'user',
      content: input.trim(),
      timestamp: new Date().toISOString(),
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)
    setIsStreaming(true)
    setThinkingSteps([])
    setError(null)

    try {
      const res = await fetch(`${apiBase}/sessions/${session.session_id}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage.content }),
      })

      if (!res.ok) {
        let errDetail
        try {
          const errData = await res.json()
          errDetail = errData.detail || JSON.stringify(errData)
        } catch {
          errDetail = `HTTP ${res.status}: ${res.statusText}`
        }
        throw new Error(errDetail)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop()

        for (const part of parts) {
          if (!part.trim()) continue

          const lines = part.split('\n')
          let eventType = 'step'
          let data = null

          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventType = line.slice(7).trim()
            } else if (line.startsWith('data: ')) {
              try { data = JSON.parse(line.slice(6)) } catch {}
            }
          }

          if (!data) continue

          if (eventType === 'step') {
            setThinkingSteps(prev => [...prev, data])
          } else if (eventType === 'result') {
            setIsStreaming(false)
            setThinkingSteps(currentSteps => {
              setMessages(prev => [...prev, {
                role: 'assistant',
                content: data.content,
                timestamp: new Date().toISOString(),
                processingTime: data.processing_time_ms,
                thinkingSteps: currentSteps,
              }])
              return []
            })
          } else if (eventType === 'error') {
            setIsStreaming(false)
            const errorContent = `⚠️ **Error from backend**\n\n\`\`\`\n${data.detail}\n\`\`\`\n\nCheck the terminal/server logs for the full stack trace.`
            setThinkingSteps(currentSteps => {
              setMessages(prev => [...prev, {
                role: 'assistant',
                content: errorContent,
                timestamp: new Date().toISOString(),
                isError: true,
                thinkingSteps: currentSteps,
              }])
              return []
            })
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') return
      setError(err.message)
      setIsStreaming(false)
      const errorContent = `⚠️ **Error**\n\n\`\`\`\n${err.message}\n\`\`\``
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: errorContent,
        timestamp: new Date().toISOString(),
        isError: true,
      }])
      setThinkingSteps([])
    } finally {
      setLoading(false)
    }
  }, [input, loading, apiBase, session.session_id])

  const formatTime = (timestamp) => {
    if (!timestamp) return ''
    return new Date(timestamp).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const renderMessage = (msg, index) => {
    return (
      <div
        key={index}
        className={`message ${msg.role} ${msg.isError ? 'error' : ''}`}
      >
        <div className="message-avatar">
          {msg.role === 'user' ? '👤' : '🤖'}
        </div>
        <div className="message-content">
          {/* Completed thinking summary */}
          {msg.thinkingSteps && msg.thinkingSteps.length > 0 && (
            <ThinkingIndicator
              steps={msg.thinkingSteps}
              isActive={false}
              elapsed={msg.processingTime ? (msg.processingTime / 1000).toFixed(0) : '?'}
            />
          )}
          <div className="message-text markdown-body">
            {msg.role === 'user' ? (
              <p>{msg.content}</p>
            ) : (
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {msg.content}
              </ReactMarkdown>
            )}
          </div>
          <div className="message-meta">
            <span className="message-time">{formatTime(msg.timestamp)}</span>
            {msg.processingTime && (
              <span className="message-processing">
                {(msg.processingTime / 1000).toFixed(1)}s
              </span>
            )}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="chat-container">
      <header className="chat-header">
        <div className="chat-header-info">
          <h1>🏥 PHA Chat</h1>
          <div className="session-info">
            <span className="session-badge baseline">
              {userStudyMode
                ? { pha: 'AGENT A', parallel: 'AGENT B', phia: 'AGENT C' }[session.config.baseline]
                : session.config.baseline.toUpperCase()}
            </span>
            {!userStudyMode && (
              <>
                <span className="session-badge model">
                  {session.config.model_id.split('/').pop()}
                </span>
                <span className="session-badge persona">
                  {session.config.persona_id}
                </span>
              </>
            )}
          </div>
        </div>
        <div className="chat-header-actions">
          <button className="refresh-btn" onClick={onRefresh} disabled={refreshing}>
            {refreshing ? 'Refreshing...' : 'Refresh'}
          </button>
          <button className="end-session-btn" onClick={handleEndSession} disabled={refreshing}>
            End Session
          </button>
        </div>
      </header>

      <div className="pha-disclaimer" role="note">
        <span className="pha-disclaimer-label">NOTE:</span>
        This research portal (i) is not intended to be a medical device; and (ii) is not intended for clinical use of any kind, including but not limited to diagnosis, prognosis, or treatment recommendations.
      </div>

      <div className="chat-messages">
        {messages.map((msg, i) => renderMessage(msg, i))}

        {/* Live thinking indicator while streaming */}
        {isStreaming && (
          <div className="message assistant">
            <div className="message-avatar">🤖</div>
            <div className="message-content">
              <ThinkingIndicator
                steps={thinkingSteps}
                isActive={true}
                elapsed={elapsed}
              />
            </div>
          </div>
        )}

        {/* Fallback dots if loading but stream hasn't started */}
        {loading && !isStreaming && thinkingSteps.length === 0 && (
          <div className="message assistant loading">
            <div className="message-avatar">🤖</div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span><span></span><span></span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <form className="chat-input-form" onSubmit={sendMessage}>
        <div className="chat-input-container">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your health data..."
            disabled={loading}
            autoFocus
          />
          <button type="submit" disabled={loading || !input.trim()}>
            {loading ? '...' : 'Send'}
          </button>
        </div>
        <p className="input-hint">
          Press Enter to send • Session ID: {session.session_id}
        </p>
      </form>
    </div>
  )
}

export default ChatInterface

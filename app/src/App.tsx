import { useState, useCallback, useEffect } from 'react'
import type { FormEvent } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import Markdown from 'react-markdown'

type QueryMode = 'simple' | 'cited' | 'validated' | 'agentic' | 'graph'

interface EvidenceSource {
  document_id: string
  section: string | null
  similarity: number
  title: string | null
  author: string | null
  url: string | null
}

interface QueryResponse {
  query: string
  response: string
  sources: EvidenceSource[]
  mode: string
  model: string
  input_tokens: number
  output_tokens: number
  // Validated mode fields
  total_claims?: number
  supported_claims?: number
  support_ratio?: number
  is_trustworthy?: boolean
  validation_report?: string
  // Agentic mode fields
  llm_calls?: number
  retrieval_count?: number
  reasoning_chain?: string[]
  termination_reason?: string
  // Graph mode fields
  related_eips?: number[]
  dependency_chain?: number[]
}

interface HealthResponse {
  status: string
  timestamp: string
  chunks_count: number
  documents_count: number
}

const MODE_DESCRIPTIONS: Record<QueryMode, string> = {
  simple: 'Fast, basic retrieval',
  cited: 'Includes source citations',
  validated: 'NLI-verified claims',
  agentic: 'ReAct reasoning loop',
  graph: 'With EIP dependencies',
}

const EXAMPLE_QUERIES = [
  'What is EIP-1559?',
  'How does EIP-4844 reduce gas costs?',
  'What are the requirements for EIP-4895?',
  'Explain the difference between EIP-2718 and EIP-2930',
]

function App() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<QueryMode>('cited')
  const [topK, setTopK] = useState(10)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<QueryResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [expandedSources, setExpandedSources] = useState<Set<number>>(new Set())

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch('/api/health')
      if (res.ok) {
        const data = await res.json()
        setHealth(data)
      }
    } catch {
      setHealth(null)
    }
  }, [])

  useEffect(() => {
    checkHealth()
  }, [checkHealth])

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    if (!query.trim() || isLoading) return

    setIsLoading(true)
    setError(null)
    setResult(null)
    setExpandedSources(new Set())

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query.trim(),
          mode,
          top_k: topK,
          validate: mode === 'validated',
        }),
      })

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({ detail: 'Request failed' }))
        throw new Error(errorData.detail || `HTTP ${res.status}`)
      }

      const data: QueryResponse = await res.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unexpected error occurred')
    } finally {
      setIsLoading(false)
    }
  }

  const handleExampleClick = (exampleQuery: string) => {
    setQuery(exampleQuery)
  }

  const toggleSourceExpanded = (index: number) => {
    setExpandedSources((prev) => {
      const next = new Set(prev)
      if (next.has(index)) {
        next.delete(index)
      } else {
        next.add(index)
      }
      return next
    })
  }

  const formatSimilarity = (similarity: number) => {
    return `${(similarity * 100).toFixed(1)}%`
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div className="logo">
            <div className="logo-mark">EPI</div>
            <div className="logo-text">
              Ethereum Protocol Intelligence <span>/ Preview</span>
            </div>
          </div>
          <div
            className={`status-badge ${health ? 'connected' : 'error'}`}
            role="status"
            aria-live="polite"
          >
            {health ? `${health.chunks_count.toLocaleString()} chunks` : 'disconnected'}
          </div>
        </div>
      </header>

      <main className="main-content">
        <section className="query-section">
          <form className="query-form" onSubmit={handleSubmit}>
            <div className="query-input-wrapper">
              <input
                type="text"
                className="query-input"
                placeholder="Ask a question about Ethereum protocol..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                disabled={isLoading}
                aria-label="Search query"
              />
              <button
                type="submit"
                className={`query-submit ${isLoading ? 'loading' : ''}`}
                disabled={isLoading || !query.trim()}
              >
                {isLoading ? 'Searching...' : 'Search'}
              </button>
            </div>

            <div className="query-options">
              <div className="option-group" role="group" aria-labelledby="mode-label">
                <span id="mode-label" className="option-label">Mode</span>
                <div className="mode-selector" role="radiogroup" aria-labelledby="mode-label">
                  {(['simple', 'cited', 'validated', 'agentic', 'graph'] as QueryMode[]).map((m) => (
                    <button
                      key={m}
                      type="button"
                      role="radio"
                      aria-checked={mode === m}
                      className={`mode-btn ${mode === m ? 'active' : ''}`}
                      onClick={() => setMode(m)}
                    >
                      {m}
                    </button>
                  ))}
                </div>
              </div>

              <div className="option-group">
                <label htmlFor="top-k-input" className="option-label">Sources</label>
                <input
                  type="number"
                  id="top-k-input"
                  className="top-k-input"
                  value={topK}
                  onChange={(e) => setTopK(Math.max(1, Math.min(50, parseInt(e.target.value) || 10)))}
                  min={1}
                  max={50}
                />
              </div>

              <span className="mode-description">{MODE_DESCRIPTIONS[mode]}</span>
            </div>
          </form>
        </section>

        <AnimatePresence mode="wait">
          {!result && !isLoading && !error && (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="empty-state"
            >
              <div className="empty-state-icon">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
                  <path d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
                </svg>
              </div>
              <h2 className="empty-state-title">Query the Ethereum Protocol</h2>
              <p className="empty-state-text">
                Ask questions about EIPs, protocol specifications, and Ethereum governance.
                Responses are grounded in official documentation.
              </p>
              <div className="example-queries">
                {EXAMPLE_QUERIES.map((eq) => (
                  <button
                    key={eq}
                    className="example-query"
                    onClick={() => handleExampleClick(eq)}
                  >
                    {eq}
                  </button>
                ))}
              </div>
            </motion.div>
          )}

          {isLoading && (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="results-container"
            >
              <div className="answer-panel">
                <div className="answer-header">
                  <span className="answer-title">Response</span>
                </div>
                <div className="answer-content">
                  <div className="loading-skeleton skeleton-line" />
                  <div className="loading-skeleton skeleton-line" />
                  <div className="loading-skeleton skeleton-line" />
                  <div className="loading-skeleton skeleton-line" style={{ width: '80%' }} />
                  <div className="loading-skeleton skeleton-line" style={{ width: '60%' }} />
                </div>
              </div>
              <div className="sources-panel">
                <div className="sources-header">
                  <span className="sources-title">Sources</span>
                </div>
                <div className="sources-list">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="source-item">
                      <div className="loading-skeleton skeleton-line" style={{ width: '40%' }} />
                      <div className="loading-skeleton skeleton-line" style={{ width: '60%', marginTop: '8px' }} />
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          {error && (
            <motion.div
              key="error"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="error-message"
            >
              <strong>Error:</strong> {error}
            </motion.div>
          )}

          {result && (
            <motion.div
              key="result"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="results-container"
            >
              <motion.div
                className="answer-panel"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
              >
                <div className="answer-header">
                  <span className="answer-title">Response</span>
                  <div className="answer-meta">
                    <span>{result.model}</span>
                    <span>{result.input_tokens + result.output_tokens} tokens</span>
                  </div>
                </div>
                <div className="answer-content prose">
                  <Markdown>{result.response}</Markdown>
                </div>

                {result.mode === 'validated' && result.total_claims !== undefined && (
                  <div className="validation-section">
                    <div className="validation-header">
                      <span className="validation-title">Claim Validation</span>
                      <span className={`trust-badge ${result.is_trustworthy ? 'trustworthy' : 'low-trust'}`}>
                        {result.is_trustworthy ? 'Trustworthy' : 'Low Trust'}
                      </span>
                    </div>
                    <div className="validation-stats">
                      <div className="stat">
                        <span className="stat-label">Claims</span>
                        <span className="stat-value">{result.total_claims}</span>
                      </div>
                      <div className="stat">
                        <span className="stat-label">Supported</span>
                        <span className="stat-value">{result.supported_claims}</span>
                      </div>
                      <div className="stat">
                        <span className="stat-label">Ratio</span>
                        <span className="stat-value">
                          {result.support_ratio !== undefined
                            ? `${(result.support_ratio * 100).toFixed(0)}%`
                            : 'N/A'}
                        </span>
                      </div>
                    </div>
                    {result.validation_report && (
                      <pre className="validation-report">{result.validation_report}</pre>
                    )}
                  </div>
                )}

                {result.mode === 'agentic' && result.reasoning_chain && (
                  <div className="validation-section">
                    <div className="validation-header">
                      <span className="validation-title">Reasoning Chain</span>
                      <span className="trust-badge trustworthy">
                        {result.termination_reason}
                      </span>
                    </div>
                    <div className="validation-stats">
                      <div className="stat">
                        <span className="stat-label">LLM Calls</span>
                        <span className="stat-value">{result.llm_calls}</span>
                      </div>
                      <div className="stat">
                        <span className="stat-label">Retrievals</span>
                        <span className="stat-value">{result.retrieval_count}</span>
                      </div>
                    </div>
                    <div className="reasoning-chain">
                      {result.reasoning_chain.map((step, i) => (
                        <div key={i} className="reasoning-step">
                          <span className="step-number">{i + 1}</span>
                          <span className="step-content">{step}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {result.mode === 'graph' && (result.related_eips || result.dependency_chain) && (
                  <div className="validation-section">
                    <div className="validation-header">
                      <span className="validation-title">EIP Graph Context</span>
                      <span className="trust-badge trustworthy">
                        {(result.related_eips?.length || 0) + (result.dependency_chain?.length || 0)} related
                      </span>
                    </div>
                    {result.related_eips && result.related_eips.length > 0 && (
                      <div className="graph-section">
                        <span className="graph-label">Related EIPs:</span>
                        <div className="eip-links">
                          {result.related_eips.map((eip) => (
                            <a
                              key={eip}
                              href={`https://eips.ethereum.org/EIPS/eip-${eip}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="eip-link"
                            >
                              EIP-{eip}
                            </a>
                          ))}
                        </div>
                      </div>
                    )}
                    {result.dependency_chain && result.dependency_chain.length > 0 && (
                      <div className="graph-section">
                        <span className="graph-label">Dependency Chain:</span>
                        <div className="dependency-chain">
                          {result.dependency_chain.map((eip, i) => (
                            <span key={eip} className="chain-item">
                              {i > 0 && <span className="chain-arrow">→</span>}
                              <a
                                href={`https://eips.ethereum.org/EIPS/eip-${eip}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="eip-link"
                              >
                                EIP-{eip}
                              </a>
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </motion.div>

              <motion.div
                className="sources-panel"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
              >
                <div className="sources-header">
                  <span className="sources-title">Sources</span>
                  <span className="sources-count">{result.sources.length} retrieved</span>
                </div>
                <div className="sources-list">
                  {result.sources.map((source, index) => (
                    <div
                      key={index}
                      className={`source-item ${expandedSources.has(index) ? 'expanded' : ''}`}
                      onClick={() => toggleSourceExpanded(index)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault()
                          toggleSourceExpanded(index)
                        }
                      }}
                      role="button"
                      tabIndex={0}
                      aria-expanded={expandedSources.has(index)}
                    >
                      <div className="source-header">
                        {source.url ? (
                          <a
                            href={source.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="source-id source-link"
                            onClick={(e) => e.stopPropagation()}
                          >
                            {source.document_id}
                          </a>
                        ) : (
                          <span className="source-id">{source.document_id}</span>
                        )}
                        <span className="source-similarity">{formatSimilarity(source.similarity)}</span>
                      </div>
                      {source.title && (
                        <div className="source-title">{source.title}</div>
                      )}
                      {source.author && (
                        <span className="source-author">{source.author}</span>
                      )}
                      {source.section && (
                        <div className="source-section">{source.section}</div>
                      )}
                    </div>
                  ))}
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  )
}

export default App

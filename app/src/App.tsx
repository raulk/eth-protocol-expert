import { useState, useCallback, useEffect, useRef, useMemo } from 'react'
import type { FormEvent, CSSProperties } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import Markdown from 'react-markdown'
import uFuzzy from '@leeoniya/ufuzzy'

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
  total_claims?: number
  supported_claims?: number
  support_ratio?: number
  is_trustworthy?: boolean
  validation_report?: string
  llm_calls?: number
  retrieval_count?: number
  reasoning_chain?: string[]
  termination_reason?: string
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

interface ModelOption {
  id: string
  name: string
  provider: string
  pricing: { input: number; output: number }
}

const uf = new uFuzzy({ intraMode: 1, intraIns: 1 })

const EXAMPLE_QUERIES = [
  'What is EIP-1559?',
  'How does EIP-4844 reduce gas costs?',
  'What are the requirements for EIP-4895?',
  'Explain the difference between EIP-2718 and EIP-2930',
]

function App() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<QueryMode>('agentic')
  const [topK, setTopK] = useState(10)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<QueryResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [expandedSources, setExpandedSources] = useState<Set<number>>(new Set())
  const [glowActive, setGlowActive] = useState(true)
  const [selectedModel, setSelectedModel] = useState<string | null>(null)
  const [maxTokens, setMaxTokens] = useState(2048)
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false)
  const [modelSearch, setModelSearch] = useState('')
  const [allModels, setAllModels] = useState<ModelOption[]>([])
  const [modelsLoading, setModelsLoading] = useState(false)

  const dropdownRef = useRef<HTMLDivElement>(null)
  const modelHaystack = useMemo(() => allModels.map((m) => `${m.name} ${m.id}`), [allModels])

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setModelDropdownOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleModelDropdownOpen = async () => {
    setModelDropdownOpen((prev) => !prev)
    if (allModels.length > 0) return
    setModelsLoading(true)
    try {
      const res = await fetch('/api/models')
      if (res.ok) {
        const data: ModelOption[] = await res.json()
        setAllModels(data)
      }
    } catch {
      // Models list is best-effort
    } finally {
      setModelsLoading(false)
    }
  }

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
          model: selectedModel,
          max_tokens: maxTokens,
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
            <div className={`search-glow ${glowActive ? 'intro' : ''}`}>
              <div className="query-input-wrapper">
                <input
                  type="text"
                  className="query-input"
                  placeholder="Ask a question about Ethereum protocol..."
                  value={query}
                  onChange={(e) => { setQuery(e.target.value); setGlowActive(false) }}
                  onFocus={() => setGlowActive(false)}
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

              <div className="option-group model-dropdown" ref={dropdownRef}>
                <span className="option-label">Model</span>
                <button
                  type="button"
                  className="model-trigger"
                  onClick={handleModelDropdownOpen}
                >
                  {selectedModel
                    ? allModels.find((m) => m.id === selectedModel)?.name ?? selectedModel
                    : 'Default'}
                  <span className="chevron">{modelDropdownOpen ? '\u25B4' : '\u25BE'}</span>
                </button>
                {modelDropdownOpen && (
                  <div className="model-panel">
                    <input
                      type="text"
                      className="model-search"
                      placeholder="Search models..."
                      value={modelSearch}
                      onChange={(e) => setModelSearch(e.target.value)}
                      autoFocus
                    />
                    {modelsLoading ? (
                      <div className="model-section-label">Loading...</div>
                    ) : (
                      (() => {
                        let filtered: ModelOption[]
                        if (!modelSearch.trim()) {
                          filtered = allModels
                        } else {
                          const [idxs, , order] = uf.search(modelHaystack, modelSearch)
                          if (!idxs || idxs.length === 0) {
                            filtered = []
                          } else {
                            const sorted = order ? order.map((i) => idxs[i]) : idxs
                            filtered = sorted.map((i) => allModels[i])
                          }
                        }
                        const grouped = new Map<string, ModelOption[]>()
                        for (const m of filtered) {
                          const list = grouped.get(m.provider) ?? []
                          list.push(m)
                          grouped.set(m.provider, list)
                        }
                        const providerOrder = ['Anthropic', 'Google', 'OpenRouter']
                        const sortedProviders = [...grouped.keys()].sort(
                          (a, b) => (providerOrder.indexOf(a) === -1 ? 99 : providerOrder.indexOf(a)) -
                                     (providerOrder.indexOf(b) === -1 ? 99 : providerOrder.indexOf(b))
                        )
                        return (
                          <>
                            <button
                              type="button"
                              className={`model-option ${selectedModel === null ? 'active' : ''}`}
                              onClick={() => { setSelectedModel(null); setModelDropdownOpen(false) }}
                            >
                              <span className="model-name">Default (Claude Opus 4.6)</span>
                            </button>
                            {sortedProviders.map((provider) => (
                              <div key={provider}>
                                <div className="model-section-label">{provider}</div>
                                {(grouped.get(provider) ?? []).slice(0, provider === 'OpenRouter' ? 30 : undefined).map((m) => (
                                  <button
                                    key={m.id}
                                    type="button"
                                    className={`model-option ${selectedModel === m.id ? 'active' : ''}`}
                                    onClick={() => { setSelectedModel(m.id); setModelDropdownOpen(false) }}
                                  >
                                    <span className="model-name">{m.name}</span>
                                    <span className="model-price">
                                      ${m.pricing.input} / ${m.pricing.output}
                                    </span>
                                  </button>
                                ))}
                              </div>
                            ))}
                          </>
                        )
                      })()
                    )}
                  </div>
                )}
              </div>

              <div className="option-group max-tokens-group">
                <label htmlFor="max-tokens-slider" className="option-label">Tokens</label>
                <input
                  type="range"
                  id="max-tokens-slider"
                  className="max-tokens-slider"
                  min={256}
                  max={16384}
                  step={256}
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(Number(e.target.value))}
                />
                <input
                  type="number"
                  className="top-k-input"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(Math.max(256, Math.min(16384, Number(e.target.value) || 2048)))}
                  min={256}
                  max={16384}
                  step={256}
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
              <svg className="eth-diamond" viewBox="0 0 256 417" fill="none" stroke="currentColor" strokeWidth="1" aria-hidden="true">
                <path d="M127.961 0L127.962 154.158V287.958L255.923 212.32z" />
                <path d="M127.962 0L0 212.32L127.962 287.959V154.158z" />
                <path d="M127.961 312.187L127.962 416.905L255.999 236.587z" />
                <path d="M127.962 416.905V312.185L0 236.585z" />
              </svg>
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
                    {result.support_ratio !== undefined && (
                      <div className="trust-meter">
                        <div
                          className="trust-fill"
                          style={{ width: `${result.support_ratio * 100}%` }}
                        />
                      </div>
                    )}
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
                        {(result.related_eips?.length ?? 0) + (result.dependency_chain?.length ?? 0)} related
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
                              {i > 0 && <span className="chain-arrow">&rarr;</span>}
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
                      style={{ '--relevance': source.similarity } as CSSProperties}
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

import { useState, useEffect } from 'react'
import axios from 'axios'
import '../styles/KeywordAnalysis.css'

interface Keyword {
  id: string
  keyword: string
  main_intent: string
  secondary_intents: string[]
  language: string | null
  confidence: number
  model: string
  created_at: string
}

interface PaginationInfo {
  current_page: number
  page_size: number
  total_items: number
  total_pages: number
}

interface KeywordStats {
  total_keywords: number
  intent_distribution: Record<string, number>
}

const PAGE_SIZE_OPTIONS = [10, 20, 50, 100]

const INTENT_COLORS: Record<string, string> = {
  informational: '#64c8ff',
  navigational: '#ffb164',
  commercial: '#64ff96',
  transactional: '#c864ff',
}

export default function KeywordAnalysis() {
  const [keywords, setKeywords] = useState<Keyword[]>([])
  const [pagination, setPagination] = useState<PaginationInfo>({
    current_page: 1,
    page_size: 20,
    total_items: 0,
    total_pages: 0,
  })
  const [stats, setStats] = useState<KeywordStats>({
    total_keywords: 0,
    intent_distribution: {},
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedPageSize, setSelectedPageSize] = useState(20)

  useEffect(() => {
    loadKeywords(1, pagination.page_size)
    loadStats()
  }, [])

  const loadKeywords = async (page: number, pageSize: number) => {
    setLoading(true)
    setError(null)
    try {
      const response = await axios.get('http://localhost:8000/api/keywords', {
        params: {
          page,
          page_size: pageSize,
        },
      })
      setKeywords(response.data.items)
      setPagination(response.data.pagination)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Failed to load keywords'
      )
    } finally {
      setLoading(false)
    }
  }

  const loadStats = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/keywords/stats')
      setStats(response.data)
    } catch (err) {
      console.error('Failed to load stats:', err)
    }
  }

  const handlePageChange = (newPage: number) => {
    if (newPage >= 1 && newPage <= pagination.total_pages) {
      loadKeywords(newPage, pagination.page_size)
    }
  }

  const handlePageSizeChange = (newPageSize: number) => {
    setSelectedPageSize(newPageSize)
  }

  const handleSavePageSize = () => {
    if (selectedPageSize !== pagination.page_size) {
      loadKeywords(1, selectedPageSize)
    }
  }

  const getIntentColor = (intent: string): string => {
    return INTENT_COLORS[intent.toLowerCase()] || '#999'
  }

  return (
    <div className="keyword-analysis">
      <div className="section">
        <h2>Keyword Intent Analysis</h2>
        <p className="description">
          Browse and analyze search keywords with their classified intents
        </p>

        {error && <div className="error-message">{error}</div>}

        <div className="stats-section">
          <div className="stat-card">
            <span className="stat-label">Total Keywords</span>
            <span className="stat-value">{stats.total_keywords}</span>
          </div>
          {Object.entries(stats.intent_distribution).map(([intent, count]) => (
            <div key={intent} className="stat-card">
              <span className="stat-label">{intent.toUpperCase()}</span>
              <span className="stat-value" style={{ color: getIntentColor(intent) }}>
                {count}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="section table-section">
        <div className="table-header">
          <h3>Keywords & Intents</h3>
          <div className="page-size-control">
            <label htmlFor="page-size">Items per page:</label>
            <select
              id="page-size"
              value={selectedPageSize}
              onChange={(e) => handlePageSizeChange(Number(e.target.value))}
            >
              {PAGE_SIZE_OPTIONS.map((size) => (
                <option key={size} value={size}>
                  {size}
                </option>
              ))}
            </select>
            <button
              onClick={handleSavePageSize}
              className="save-btn"
              disabled={selectedPageSize === pagination.page_size}
            >
              Save
            </button>
          </div>
        </div>

        {loading ? (
          <div className="loading">Loading keywords...</div>
        ) : keywords.length === 0 ? (
          <div className="empty-state">
            No keywords found. Import data from test_db.json to get started.
          </div>
        ) : (
          <>
            <table className="keywords-table">
              <thead>
                <tr>
                  <th>Keyword</th>
                  <th>Main Intent</th>
                  <th>Secondary Intents</th>
                  <th>Language</th>
                  <th>Confidence</th>
                  <th>Model</th>
                </tr>
              </thead>
              <tbody>
                {keywords.map((kw) => (
                  <tr key={kw.id}>
                    <td className="keyword-cell">{kw.keyword}</td>
                    <td>
                      <span
                        className="intent-badge"
                        style={{
                          backgroundColor: `${getIntentColor(kw.main_intent)}33`,
                          color: getIntentColor(kw.main_intent),
                        }}
                      >
                        {kw.main_intent}
                      </span>
                    </td>
                    <td>
                      <div className="secondary-intents">
                        {kw.secondary_intents.length > 0 ? (
                          kw.secondary_intents.map((intent, idx) => (
                            <span
                              key={idx}
                              className="secondary-badge"
                              style={{
                                backgroundColor: `${getIntentColor(intent)}22`,
                                color: getIntentColor(intent),
                              }}
                            >
                              {intent}
                            </span>
                          ))
                        ) : (
                          <span className="no-secondary">-</span>
                        )}
                      </div>
                    </td>
                    <td className="model-cell">{kw.language ?? '-'}</td>
                    <td>
                      <span className="confidence">
                        {(kw.confidence * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td className="model-cell">{kw.model}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div className="pagination">
              <button
                onClick={() => handlePageChange(pagination.current_page - 1)}
                disabled={pagination.current_page === 1}
                className="pagination-btn"
              >
                ← Previous
              </button>

              <div className="page-info">
                Page {pagination.current_page} of {pagination.total_pages}
                <span className="total-items">
                  ({pagination.total_items} total items)
                </span>
              </div>

              <button
                onClick={() => handlePageChange(pagination.current_page + 1)}
                disabled={pagination.current_page === pagination.total_pages}
                className="pagination-btn"
              >
                Next →
              </button>
            </div>
          </>
        )}
      </div>

      <div className="section info-section">
        <h3>How to Import Data</h3>
        <div className="import-guide">
          <p>To populate this table with keywords from test_db.json:</p>
          <ol>
            <li>Run the import script from the backend directory:</li>
            <li>
              <code>python scripts/import_test_data.py</code>
            </li>
            <li>The script will read test_db.json and store all keywords with their intents</li>
            <li>Refresh this page to see the imported data</li>
          </ol>
        </div>
      </div>
    </div>
  )
}

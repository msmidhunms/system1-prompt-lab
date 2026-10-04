import { useState } from 'react'
import './App.css'
import SERPAnalysis from './components/SERPAnalysis'
import KarpathyLoop from './components/KarpathyLoop'

type Page = 'serp' | 'karpathy'

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('serp')

  return (
    <div className="App">
      <header>
        <h1>System 1 Experiments</h1>
        <nav className="nav-tabs">
          <button
            className={`nav-button ${currentPage === 'serp' ? 'active' : ''}`}
            onClick={() => setCurrentPage('serp')}
          >
            SERP Analysis
          </button>
          <button
            className={`nav-button ${currentPage === 'karpathy' ? 'active' : ''}`}
            onClick={() => setCurrentPage('karpathy')}
          >
            Karpathy Loop
          </button>
        </nav>
      </header>
      <main>
        {currentPage === 'serp' && <SERPAnalysis />}
        {currentPage === 'karpathy' && <KarpathyLoop />}
      </main>
    </div>
  )
}

export default App

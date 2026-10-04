import { useState } from 'react'
import './App.css'
import SERPAnalysis from './components/SERPAnalysis'
import KarpathyLoop from './components/KarpathyLoop'
import KeywordAnalysis from './components/KeywordAnalysis'
import Evaluation from './components/Evaluation'

type Page = 'serp' | 'karpathy' | 'keywords' | 'evaluation'

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
            className={`nav-button ${currentPage === 'keywords' ? 'active' : ''}`}
            onClick={() => setCurrentPage('keywords')}
          >
            Keywords & Intents
          </button>
          <button
            className={`nav-button ${currentPage === 'evaluation' ? 'active' : ''}`}
            onClick={() => setCurrentPage('evaluation')}
          >
            Model Evaluation
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
        {currentPage === 'keywords' && <KeywordAnalysis />}
        {currentPage === 'evaluation' && <Evaluation />}
        {currentPage === 'karpathy' && <KarpathyLoop />}
      </main>
    </div>
  )
}

export default App

import { useState } from 'react'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <div className="App">
      <header>
        <h1>System 1 Experiments</h1>
      </header>
      <main>
        <p>Welcome to the System 1 Experiments frontend</p>
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
      </main>
    </div>
  )
}

export default App

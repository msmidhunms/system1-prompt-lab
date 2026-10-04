import { PromptConfig } from '../../api'

// Runs recorded by older versions have no score / macro-F1 / counts.
export interface Metrics {
  score?: number
  accuracy: number
  macro_f1: number
  raw_accuracy?: number
  predicted_counts?: Record<string, number>
}

export interface Round {
  iteration: number
  timestamp: string
  status: 'keep' | 'discard' | 'crash'
  hypothesis: string | null
  config: PromptConfig | null
  score?: number | null
  accuracy: number | null
  macro_f1: number | null
  delta_vs_best: number | null
  p_better?: number | null
  predicted_counts?: Record<string, number>
  warnings: string[]
  error: string | null
}

export interface Run {
  run_id: string
  task: string
  labels: string[]
  items?: string
  status: 'running' | 'completed' | 'stopped' | 'failed' | 'interrupted'
  phase: string
  started_at: string
  llm: { provider: string; model: string }
  loops: number
  start_version: string
  metric?: string
  calibrate?: boolean
  use_serp?: boolean
  laya_model?: string
  language?: string | null
  dev_size: number
  holdout_size: number
  baseline_config: PromptConfig
  baseline: Metrics | null
  best_config: PromptConfig
  best_score?: number | null
  best_accuracy: number | null
  best_macro_f1?: number | null
  best_iteration: number
  improved: boolean
  iterations: Round[]
  holdout: { baseline: Metrics; best: Metrics; p_better?: number | null } | null
  error: string | null
  log: string[]
  saved_versions?: { version: string; source: string; iteration: number | null }[]
}

export interface RunSummary {
  run_id: string
  status: Run['status']
  started_at: string
  llm: { provider: string; model: string }
  loops: number
  completed_iterations: number
  baseline_accuracy: number | null
  best_accuracy: number | null
  best_iteration: number | null
  improved: boolean
}

// Older runs were scored by accuracy alone.
export const scoreOf = (m: { score?: number | null; accuracy: number | null } | null | undefined) => m?.score ?? m?.accuracy ?? null

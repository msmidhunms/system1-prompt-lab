import axios from 'axios'

export const API = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000/api'

export const api = axios.create({ baseURL: API })

export interface ImportStatus {
  status: 'queued' | 'downloading' | 'done' | 'failed'
  message: string
}

// An example the harness can run: one text input classified into a fixed set of labels.
export interface TaskInfo {
  id: string
  name: string
  summary: string
  input_field: string
  input_label: string
  item: string
  items: string
  question_type: 'choice' | 'noul'
  labels: string[]
  label_descriptions: Record<string, string>
  language: string | null
  supports_serp: boolean
  source: string | null
  golden_rows: number
  importable: boolean
  import_status: ImportStatus | null
  setup_hint: string
  max_criterion_words: number
  max_instruction_words: number
}

// A model an example can be run on.
export interface Engine {
  id: string
  name: string
  repo: string
  params: string | null
  family: string
  passes: string
  notes: string
}

export interface PromptConfig {
  // The model the prompt is for. Left out, it is Laya.
  engine?: string
  state_template: string | Record<string, string>
  serp_results?: number
  instructions: string
  criteria: Record<string, string>
  model?: string
  label_bias?: Record<string, number>
}

export interface EvaluationBrief {
  eval_id: string
  accuracy: number
  macro_f1: number
  sample_size: number
}

export interface ModelVersion {
  version: string
  source: 'baseline' | 'auto' | 'optimizer' | 'manual'
  base_version: string | null
  description: string | null
  run_id: string | null
  iteration: number | null
  accuracy: number | null
  macro_f1: number | null
  holdout_accuracy: number | null
  latest_evaluation: EvaluationBrief | null
  created_at: string | null
  config: PromptConfig
}

export interface Activity {
  loop: { run_id: string; task: string; phase: string; rounds_done: number; rounds: number } | null
  evaluations: { eval_id: string; task: string; engine: string; version: string; status: string; progress_done: number; progress_total: number; comparison_id: string | null }[]
  imports: { task: string; status: string; message: string }[]
}

export const TABS = [
  { id: 'playground', label: 'Playground' },
  { id: 'dataset', label: 'Dataset' },
  { id: 'models', label: 'Models' },
  { id: 'evaluation', label: 'Evaluation' },
  { id: 'compare', label: 'Compare Models' },
  { id: 'optimizer', label: 'Prompt Optimizer' },
] as const
export type TabId = (typeof TABS)[number]['id']

export function apiError(err: unknown, fallback: string): string {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail
    if (typeof detail === 'string') return detail
    if (!err.response) return 'The backend is not reachable. Start it on port 8000 and try again.'
    return err.message
  }
  return err instanceof Error ? err.message : fallback
}

// Labels take the categorical colours in a fixed order (defined for both themes in tokens.css).
export function labelColor(labels: string[], label: string): string {
  const index = labels.indexOf(label)
  return index < 0 ? 'var(--text-muted)' : `var(--series-${(index % 8) + 1})`
}

export const pct = (value: number | null | undefined, digits = 1) =>
  value == null ? '–' : `${(value * 100).toFixed(digits)}%`

// Server timestamps are UTC without a zone suffix.
export const when = (timestamp: string | null | undefined) =>
  timestamp ? new Date(timestamp.endsWith('Z') ? timestamp : `${timestamp}Z`).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '–'

export const stateText = (state: PromptConfig['state_template']) => (typeof state === 'string' ? state : JSON.stringify(state))

export const SOURCE_LABELS: Record<string, string> = {
  baseline: 'Baseline',
  auto: 'Auto-saved',
  optimizer: 'Optimizer',
  manual: 'Manual',
}

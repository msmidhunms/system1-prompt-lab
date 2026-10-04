import axios from 'axios'

export const API = 'http://localhost:8000/api'

export const api = axios.create({ baseURL: API })

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
}

export function apiError(err: unknown, fallback: string): string {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail
    if (typeof detail === 'string') return detail
    return err.message
  }
  return err instanceof Error ? err.message : fallback
}

// The first four are the colours search intent has always used.
const LABEL_COLORS = ['#64c8ff', '#ffb164', '#64ff96', '#c864ff', '#ff8fa3', '#ffe066', '#5eead4', '#b0b8ff']

export function labelColor(labels: string[], label: string): string {
  const index = labels.indexOf(label)
  return index < 0 ? '#999' : LABEL_COLORS[index % LABEL_COLORS.length]
}

export const pct = (value: number | null | undefined) => (value == null ? '-' : `${(value * 100).toFixed(1)}%`)

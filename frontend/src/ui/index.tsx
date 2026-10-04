// The shared building blocks every page is made of.
import { ReactNode, useEffect, useRef, useState } from 'react'

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'default' | 'primary' | 'ghost' | 'danger'
  small?: boolean
}

export function Button({ variant = 'default', small, className = '', ...rest }: ButtonProps) {
  const classes = ['btn', variant !== 'default' && variant, small && 'sm', className].filter(Boolean).join(' ')
  return <button type="button" className={classes} {...rest} />
}

export function Card({ title, sub, actions, children, flush }: {
  title?: ReactNode
  sub?: ReactNode
  actions?: ReactNode
  children: ReactNode
  flush?: boolean
}) {
  return (
    <section className="card">
      {(title || actions) && (
        <div className="card-header">
          <div>
            {title && <h3>{title}</h3>}
            {sub && <p className="sub">{sub}</p>}
          </div>
          {actions && <div className="row">{actions}</div>}
        </div>
      )}
      <div className={`card-body ${flush ? 'flush' : ''}`}>{children}</div>
    </section>
  )
}

// A label of an example: the colour is a dot beside the text, the text stays in ink.
export function LabelBadge({ label, color }: { label: string; color: string }) {
  return (
    <span className="badge">
      <span className="dot" style={{ background: color }} />
      {label}
    </span>
  )
}

export function Badge({ tone, children }: { tone?: 'good' | 'bad' | 'warn' | 'accent'; children: ReactNode }) {
  return <span className={`badge ${tone ?? ''}`}>{children}</span>
}

const STATUS_TONES: Record<string, 'good' | 'bad' | 'warn' | 'accent' | undefined> = {
  completed: 'good', keep: 'good', done: 'good',
  failed: 'bad', crash: 'bad',
  running: 'accent', queued: 'accent', downloading: 'accent',
  stopped: 'warn', interrupted: 'warn',
}
const STATUS_TEXT: Record<string, string> = { keep: 'Kept', discard: 'Discarded', crash: 'Failed' }

export function StatusBadge({ status }: { status: string }) {
  return <Badge tone={STATUS_TONES[status]}>{STATUS_TEXT[status] ?? status.charAt(0).toUpperCase() + status.slice(1)}</Badge>
}

export function Field({ label, hint, over, children }: { label: string; hint?: ReactNode; over?: boolean; children: ReactNode }) {
  return (
    <div className="field">
      <span className="label">{label}</span>
      {children}
      {hint && <span className={`hint ${over ? 'over' : ''}`}>{hint}</span>}
    </div>
  )
}

// A whole number between min and max. What is typed stays on screen as typed until the field is left:
// clamping it on each keystroke would turn a "2" on the way to "250" into the minimum.
export function NumberInput({ value, min, max, onChange, disabled }: {
  value: number
  min: number
  max: number
  onChange: (value: number) => void
  disabled?: boolean
}) {
  const [draft, setDraft] = useState<string | null>(null)
  return (
    <input
      className="input narrow num"
      type="number"
      min={min}
      max={max}
      disabled={disabled}
      value={draft ?? value}
      onChange={(e) => {
        setDraft(e.target.value)
        const typed = parseInt(e.target.value)
        if (!Number.isNaN(typed)) onChange(Math.max(min, Math.min(max, typed)))
      }}
      onBlur={() => setDraft(null)}
    />
  )
}

export function Banner({ tone, children, action }: { tone?: 'error' | 'success' | 'warning'; children: ReactNode; action?: ReactNode }) {
  return (
    <div className={`banner ${tone ?? ''}`} role={tone === 'error' ? 'alert' : 'status'}>
      <div className="grow">{children}</div>
      {action}
    </div>
  )
}

export function ProgressBar({ done, total }: { done: number; total: number }) {
  const share = total > 0 ? Math.min(100, (done / total) * 100) : 0
  return (
    <div className="progress" role="progressbar" aria-valuenow={Math.round(share)} aria-valuemin={0} aria-valuemax={100}>
      <div style={{ width: `${share}%` }} />
    </div>
  )
}

export function Spinner() {
  return <span className="spinner" aria-label="In progress" />
}

export function StatTile({ label, value, note, noteUp, dot }: { label: string; value: ReactNode; note?: ReactNode; noteUp?: boolean; dot?: string }) {
  return (
    <div className="stat">
      <div className="label">
        {dot && <span className="dot" style={{ width: 8, height: 8, borderRadius: '50%', background: dot }} />}
        {label}
      </div>
      <div className="value">{value}</div>
      {note && <div className={`note ${noteUp ? 'up' : ''}`}>{note}</div>}
    </div>
  )
}

export function EmptyState({ title, children, actions }: { title: string; children?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      {children}
      {actions && <div className="row">{actions}</div>}
    </div>
  )
}

export interface MenuItem {
  label: string
  onSelect: () => void
  danger?: boolean
}

// The "⋯" menu on a table row.
export function Menu({ items, label = 'Actions' }: { items: MenuItem[]; label?: string }) {
  // Where the open menu sits on screen. It is placed against the viewport, so a scrolling table cannot clip it.
  const [place, setPlace] = useState<{ top: number; right: number } | null>(null)
  const ref = useRef<HTMLDivElement>(null)
  const open = place !== null
  const setOpen = (next: boolean) => {
    const rect = ref.current?.getBoundingClientRect()
    setPlace(next && rect ? { top: rect.bottom + 4, right: window.innerWidth - rect.right } : null)
  }

  useEffect(() => {
    if (!open) return
    const close = (e: Event) => {
      if (e instanceof KeyboardEvent ? e.key === 'Escape' : e.type === 'scroll' || !ref.current?.contains(e.target as Node)) setPlace(null)
    }
    document.addEventListener('mousedown', close)
    document.addEventListener('keydown', close)
    window.addEventListener('scroll', close, true)
    return () => {
      document.removeEventListener('mousedown', close)
      document.removeEventListener('keydown', close)
      window.removeEventListener('scroll', close, true)
    }
  }, [open])

  return (
    <div className="menu" ref={ref} onClick={(e) => e.stopPropagation()}>
      <button type="button" className="btn ghost sm icon" aria-label={label} aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(!open)}>
        ⋯
      </button>
      {open && (
        <div className="menu-list" role="menu" style={{ top: place.top, right: place.right }}>
          {items.map((item) => (
            <button
              key={item.label}
              type="button"
              role="menuitem"
              className={item.danger ? 'danger' : ''}
              onClick={() => {
                setOpen(false)
                item.onSelect()
              }}
            >
              {item.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export function Dialog({ title, onClose, children, footer, wide }: {
  title: string
  onClose: () => void
  children: ReactNode
  footer?: ReactNode
  wide?: boolean
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div className="dialog-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className={`dialog ${wide ? 'wide' : ''}`} role="dialog" aria-modal="true" aria-label={title}>
        <div className="dialog-header">
          <h2>{title}</h2>
          <Button variant="ghost" small className="icon" aria-label="Close" onClick={onClose}>
            ✕
          </Button>
        </div>
        <div className="dialog-body">{children}</div>
        {footer && <div className="dialog-footer">{footer}</div>}
      </div>
    </div>
  )
}

// Asks for one line of text (a name), in a dialog.
export function NameDialog({ title, label, initial = '', hint, confirm, onSubmit, onClose }: {
  title: string
  label: string
  initial?: string
  hint?: string
  confirm: string
  onSubmit: (value: string) => Promise<string | null>
  onClose: () => void
}) {
  const [value, setValue] = useState(initial)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const submit = async () => {
    if (!value.trim()) return
    setBusy(true)
    const problem = await onSubmit(value.trim())
    setBusy(false)
    if (problem) setError(problem)
    else onClose()
  }

  return (
    <Dialog
      title={title}
      onClose={onClose}
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" onClick={submit} disabled={busy || !value.trim()}>
            {confirm}
          </Button>
        </>
      }
    >
      <div className="stack">
        <Field label={label} hint={hint ?? "Letters, digits, '_', '-' and '.'"}>
          <input className="input" autoFocus value={value} onChange={(e) => setValue(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && submit()} />
        </Field>
        {error && <Banner tone="error">{error}</Banner>}
      </div>
    </Dialog>
  )
}

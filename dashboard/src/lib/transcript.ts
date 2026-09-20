import { useSyncExternalStore } from "react"
import type { Served } from "@/lib/chat"

// The chat transcript as an external store: it survives a reload during the demo, "New
// chat" clears it, and streaming updates never touch storage more than a few times a second.
export interface Message {
  id: string
  role: "user" | "assistant"
  text: string
  served?: Served
  ms?: number
  error?: string
}

const KEY = "pi-router-chat"
const SAVE_DELAY_MS = 250

function isMessage(x: unknown): x is Message {
  if (typeof x !== "object" || x === null) return false
  const m = x as Record<string, unknown>
  return typeof m.id === "string" && (m.role === "user" || m.role === "assistant") && typeof m.text === "string"
}

// Anything malformed is dropped; an answer that was still streaming when the page
// reloaded is marked so it does not sit there saying "routing" forever.
export function parseTranscript(raw: string | null): Message[] {
  let parsed: unknown
  try {
    parsed = JSON.parse(raw ?? "[]")
  } catch {
    return []
  }
  if (!Array.isArray(parsed)) return []
  return parsed.filter(isMessage).map((m) =>
    m.role === "assistant" && m.ms == null && !m.error ? { ...m, error: "Interrupted by a reload." } : m,
  )
}

function load(): Message[] {
  try {
    return parseTranscript(sessionStorage.getItem(KEY))
  } catch {
    return []
  }
}

let messages: Message[] = load()
const listeners = new Set<() => void>()
let saveTimer: number | undefined

function flush(): void {
  if (saveTimer !== undefined) window.clearTimeout(saveTimer)
  saveTimer = undefined
  try {
    sessionStorage.setItem(KEY, JSON.stringify(messages))
  } catch {
    // storage blocked: the chat still works for this page view
  }
}

if (typeof window !== "undefined") window.addEventListener("pagehide", flush)

function subscribe(fn: () => void): () => void {
  listeners.add(fn)
  return () => listeners.delete(fn)
}

export const getTranscript = (): Message[] => messages

export function setTranscript(next: Message[]): void {
  messages = next
  if (saveTimer === undefined) saveTimer = window.setTimeout(flush, SAVE_DELAY_MS)
  for (const fn of listeners) fn()
}

export const appendMessages = (...added: Message[]): void => setTranscript([...messages, ...added])

export const patchMessage = (id: string, f: (m: Message) => Message): void =>
  setTranscript(messages.map((m) => (m.id === id ? f(m) : m)))

const EMPTY: Message[] = []
export const useTranscript = (): Message[] => useSyncExternalStore(subscribe, getTranscript, () => EMPTY)

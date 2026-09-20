import type { ChatCompletionMessageParam } from "openai/resources/chat/completions"
import type { Message } from "@/lib/transcript"

// The router sizes a request by the whole conversation, so one pasted wall of text would
// push every later question to the cloud. Send the newest turns that fit a budget instead,
// the way chat clients do; the new message always goes, however long it is.
// Earlier answers go back exactly as received, <think> blocks included: dllama-api keeps a
// KV cache keyed on the messages it saw, so an unchanged prefix means the Pis only prefill
// the new turn.
export const HISTORY_BUDGET_CHARS = 6000

export function recentHistory(messages: Message[], text: string, budget = HISTORY_BUDGET_CHARS): ChatCompletionMessageParam[] {
  const kept: ChatCompletionMessageParam[] = []
  let used = 0
  for (const m of [...messages].reverse()) {
    if (m.error || !m.text) continue
    if (used + m.text.length > budget) break
    used += m.text.length
    kept.unshift({ role: m.role, content: m.text })
  }
  return [...kept, { role: "user", content: text }]
}

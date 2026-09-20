import OpenAI from "openai"
import type { ChatCompletionChunk, ChatCompletionMessageParam } from "openai/resources/chat/completions"
import { ROUTER_URL } from "@/lib/config"

export interface Served {
  servedBy: string
  reason: string
  requestId: string | null
  continuedBy?: string // set mid-stream when the Pis died and a cloud tier finished the answer
  continuedAfter?: string // what the router says about where the handover happened
}

// The router adds one field to the chunk where a fallback took over mid-answer.
interface PiHiveChunk extends ChatCompletionChunk {
  pihive?: { continued_by?: string; after?: string }
}

// The router is OpenAI-compatible, so the browser talks to it with the real SDK.
// Asking for a tier's model by name pins that tier; "auto" lets it decide.
const client = new OpenAI({
  baseURL: `${ROUTER_URL || window.location.origin}/v1`,
  apiKey: "dashboard",
  dangerouslyAllowBrowser: true,
})

export const isAbort = (e: unknown): boolean => e instanceof OpenAI.APIUserAbortError

export async function streamChat(
  model: string,
  messages: ChatCompletionMessageParam[],
  onDelta: (text: string) => void,
  onServed: (served: Served) => void,
  signal?: AbortSignal,
): Promise<void> {
  const { data: stream, response } = await client.chat.completions
    .create({ model, stream: true, messages }, { signal })
    .withResponse()
  let served: Served = {
    servedBy: response.headers.get("x-served-by") ?? "?",
    reason: response.headers.get("x-route-reason") ?? "",
    requestId: response.headers.get("x-request-id"),
  }
  onServed(served)
  for await (const chunk of stream as AsyncIterable<PiHiveChunk>) {
    if (chunk.pihive?.continued_by) {
      served = { ...served, continuedBy: chunk.pihive.continued_by, continuedAfter: chunk.pihive.after }
      onServed(served)
    }
    const piece = chunk.choices[0]?.delta?.content
    if (piece) onDelta(piece)
  }
}

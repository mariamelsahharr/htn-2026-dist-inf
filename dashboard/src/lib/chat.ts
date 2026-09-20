import OpenAI from "openai"
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions"

export interface Served {
  servedBy: string
  reason: string
  requestId: string | null
}

// The router is OpenAI-compatible, so the browser talks to it with the real SDK on the
// same origin. Asking for a tier's model by name pins that tier; "auto" lets it decide.
const client = new OpenAI({ baseURL: `${window.location.origin}/v1`, apiKey: "dashboard", dangerouslyAllowBrowser: true })

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
  onServed({
    servedBy: response.headers.get("x-served-by") ?? "?",
    reason: response.headers.get("x-route-reason") ?? "",
    requestId: response.headers.get("x-request-id"),
  })
  for await (const chunk of stream) {
    const piece = chunk.choices[0]?.delta?.content
    if (piece) onDelta(piece)
  }
}

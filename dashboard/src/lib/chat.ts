import OpenAI from "openai"
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions"
import { baseUrl, headersFor, type Settings } from "@/lib/settings"

export interface Served {
  servedBy: string
  reason: string
  requestId: string | null
}

// The router is OpenAI-compatible, so the browser talks to it with the real SDK.
// A forced provider rides along as X-Force-Upstream; the router ignores the key.
export async function streamChat(
  settings: Settings,
  messages: ChatCompletionMessageParam[],
  onDelta: (text: string) => void,
  onServed: (served: Served) => void,
  signal?: AbortSignal,
): Promise<void> {
  const client = new OpenAI({
    baseURL: `${baseUrl(settings)}/v1`,
    apiKey: settings.apiKey || "dashboard",
    dangerouslyAllowBrowser: true,
    defaultHeaders: headersFor(settings),
  })
  const { data: stream, response } = await client.chat.completions
    .create({ model: settings.model === "auto" ? "auto" : settings.model, stream: true, messages }, { signal })
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

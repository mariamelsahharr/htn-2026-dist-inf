import { describe, expect, it } from "vitest"
import { recentHistory } from "./history"
import type { Message } from "./transcript"

const msg = (role: Message["role"], text: string, error?: string): Message => ({ id: text.slice(0, 8), role, text, error })

describe("recentHistory", () => {
  it("always ends with the new message", () => {
    expect(recentHistory([], "hi")).toEqual([{ role: "user", content: "hi" }])
  })
  it("keeps the newest turns that fit the budget and drops a wall of text behind them", () => {
    const history = recentHistory([msg("user", "x".repeat(9000)), msg("assistant", "long answer"), msg("user", "short?"), msg("assistant", "short.")], "next", 100)
    expect(history.map((m) => m.content)).toEqual(["long answer", "short?", "short.", "next"])
  })
  it("skips failed and empty turns", () => {
    const history = recentHistory([msg("user", "a"), msg("assistant", "", "Stopped."), msg("assistant", "")], "b")
    expect(history.map((m) => m.content)).toEqual(["a", "b"])
  })
})

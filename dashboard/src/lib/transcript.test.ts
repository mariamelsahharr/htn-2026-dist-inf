import { describe, expect, it } from "vitest"
import { parseTranscript } from "./transcript"

describe("parseTranscript", () => {
  it("returns nothing for garbage", () => {
    expect(parseTranscript(null)).toEqual([])
    expect(parseTranscript("{not json")).toEqual([])
    expect(parseTranscript('{"a":1}')).toEqual([])
    expect(parseTranscript('[{"id":1},{"role":"user"},"x"]')).toEqual([])
  })
  it("keeps well-formed messages", () => {
    const rows = parseTranscript(
      JSON.stringify([
        { id: "u", role: "user", text: "hi" },
        { id: "a", role: "assistant", text: "hello", ms: 12 },
      ]),
    )
    expect(rows.map((m) => m.id)).toEqual(["u", "a"])
    expect(rows[1].error).toBeUndefined()
  })
  it("marks an answer that was still streaming at reload", () => {
    const rows = parseTranscript(JSON.stringify([{ id: "a", role: "assistant", text: "" }]))
    expect(rows[0].error).toBe("Interrupted by a reload.")
  })
})

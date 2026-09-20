import { describe, expect, it } from "vitest"
import { shownText, visibleText, whyFellBack, whyRouted } from "./words"

describe("whyRouted", () => {
  it("names the rule", () => {
    expect(whyRouted("default_local")).toBe("short question")
    expect(whyRouted("over_size_threshold")).toBe("long prompt")
  })
  it("adds the no-cloud and fallback suffixes", () => {
    expect(whyRouted("cluster_unreachable_no_cloud")).toBe("Pis unreachable, no cloud configured")
    expect(whyRouted("cluster_down+fallback")).toBe("Pis down, after a fallback")
  })
  it("falls back to the key in words", () => {
    expect(whyRouted("some_new_rule")).toBe("some new rule")
  })
})

describe("whyFellBack", () => {
  it("strips the exception class and trailing punctuation", () => {
    expect(whyFellBack('pre_commit:UpstreamError: HTTP 403: {"error":"bad key"}')).toBe("before the first token: HTTP 403")
    expect(whyFellBack("mid_stream:ReadTimeout: read timed out (30s)")).toBe("mid-answer: read timed out")
  })
  it("keeps an unknown stage", () => {
    expect(whyFellBack("weird")).toBe("weird")
  })
})

describe("visibleText", () => {
  it("hides closed and open think blocks", () => {
    expect(visibleText("<think>hmm</think>\nHi")).toBe("Hi")
    expect(visibleText("Hi <think>still going")).toBe("Hi ")
    expect(visibleText("plain")).toBe("plain")
  })
})

describe("shownText", () => {
  it("collapses a wall of text", () => {
    const long = "x".repeat(1000)
    expect(shownText(long)).toMatch(/… \(1,000 characters\)$/)
    expect(shownText("short")).toBe("short")
  })
})

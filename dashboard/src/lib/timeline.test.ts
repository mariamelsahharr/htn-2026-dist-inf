import { describe, expect, it } from "vitest"
import type { RecentAnswer } from "@/lib/api"
import { SLOTS, toMarks, toSlots } from "./timeline"

const answer = (ts: number, tps = 40): RecentAnswer => ({
  served_by: "cluster",
  routed_to: "cluster",
  reason: "default_local",
  fallback: false,
  stream: true,
  latency_ms: 500,
  prompt_tokens: 10,
  decode_tps: tps,
  ts,
})

describe("toSlots", () => {
  it("right-aligns answers in fixed slots", () => {
    const slots = toSlots([answer(1), answer(2)])
    expect(slots).toHaveLength(SLOTS)
    expect(slots.slice(0, SLOTS - 2).every((s) => s.answer === null)).toBe(true)
    expect(slots[SLOTS - 1].answer?.ts).toBe(2)
    expect(slots[SLOTS - 1].tps).toBe(40)
  })
  it("keeps only the newest SLOTS answers", () => {
    const slots = toSlots(Array.from({ length: SLOTS + 5 }, (_, i) => answer(i)))
    expect(slots[0].answer?.ts).toBe(5)
  })
})

describe("toMarks", () => {
  it("puts an event on the first answer after it, labelled by state", () => {
    const slots = toSlots([answer(10), answer(20), answer(30)])
    const marks = toMarks(slots, [{ t: 15, msg: "state=degraded: lost a worker" }])
    expect(marks).toEqual([{ slot: SLOTS - 2, msg: "degraded" }])
  })
  it("ignores events older than every answer", () => {
    const slots = toSlots([answer(10), answer(20)])
    expect(toMarks(slots, [{ t: 5, msg: "state=healthy" }])).toEqual([])
  })
})

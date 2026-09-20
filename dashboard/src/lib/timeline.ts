import type { ClusterStatus, RecentAnswer } from "@/lib/api"

export const SLOTS = 24

export interface Slot {
  i: number
  tps: number | null
  ttft: number | null
  answer: RecentAnswer | null
}

export interface Mark {
  slot: number
  msg: string
}

// Fixed slots, newest at the right: one answer is one bar and the chart fills up over
// time like a level meter instead of stretching one answer across the width.
export function toSlots(recent: RecentAnswer[]): Slot[] {
  const tail = recent.slice(-SLOTS)
  const pad = SLOTS - tail.length
  return Array.from({ length: SLOTS }, (_, i) => {
    const answer = i >= pad ? tail[i - pad] : null
    return {
      i,
      tps: answer ? (answer.decode_tps ?? answer.tps ?? 0) : null,
      ttft: answer?.ttft_ms ?? null,
      answer,
    }
  })
}

// Supervisor events land on the first answer that came after them: one mark per slot,
// the latest event wins, and the label is the state word ("degraded"), not the sentence.
export function toMarks(slots: Slot[], events: ClusterStatus["events"]): Mark[] {
  const bySlot = new Map<number, string>()
  for (const e of events ?? []) {
    const slot = slots.find((s) => s.answer && s.answer.ts >= e.t)
    if (slot && slots.some((s) => s.answer && s.answer.ts < e.t)) {
      const state = /state=(\w+)/.exec(e.msg)?.[1]
      bySlot.set(slot.i, state ?? e.msg.split(":")[0].slice(0, 24))
    }
  }
  return [...bySlot.entries()].map(([slot, msg]) => ({ slot, msg }))
}

// Three prompts that each trip a different routing rule, so a judge can press them in turn.
const PARAGRAPH =
  "A Raspberry Pi is a small single-board computer built around an ARM system on a chip. It has a few gigabytes " +
  "of memory, gigabit ethernet, and enough compute to run a small language model if the weights are split across " +
  "several boards over the network, which is what this cluster does. "

export const DEMO_PROMPTS: { label: string; hint: string; text: string }[] = [
  { label: "Short question", hint: "stays on the Pis", text: "What is a Raspberry Pi? One sentence." },
  {
    label: "Long prompt",
    hint: "does not fit the Pis' 4096-token context, goes to the cloud",
    text: `Summarise this in two sentences:\n\n${PARAGRAPH.repeat(80)}`,
  },
  {
    label: "Pasted code",
    hint: "hundreds of lines of code, too complex for the Pis",
    // Many short lines: over the code-lines rule while staying under the size rule, so the
    // badge says "pasted code" rather than "long prompt".
    text:
      "Explain what this does in one paragraph:\n\n```python\n" +
      Array.from({ length: 420 }, (_, i) => `x${i} = ${i}`).join("\n") +
      "\n```",
  },
]

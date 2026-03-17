import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function CoTMethodComparison() {
  const [method, setMethod] = useState('few_shot')
  const methods = {
    few_shot: { name: 'Few-Shot CoT', desc: 'Provide examples with step-by-step solutions in the prompt. The model mimics the reasoning pattern.', example: 'Q: Roger has 5 balls. He buys 2 cans of 3. How many? A: He started with 5. 2 cans of 3 = 6. 5 + 6 = 11. The answer is 11.', improvement: '+15-25% on math/reasoning benchmarks' },
    zero_shot: { name: 'Zero-Shot CoT', desc: 'Simply append "Let\'s think step by step" to the prompt. Surprisingly effective without examples.', example: 'Q: [question] A: Let\'s think step by step...', improvement: '+10-15% on average, no examples needed' },
    self_refine: { name: 'Self-Refinement', desc: 'Generate an initial answer, then critique and revise it. Multiple rounds of refinement possible.', example: 'Initial answer -> "Wait, let me check..." -> Revised answer -> "Yes, this is correct"', improvement: '+5-10% additional over basic CoT' },
  }
  const m = methods[method]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Chain-of-Thought Methods</h3>
      <div className="flex gap-2 mb-3 flex-wrap">
        {Object.entries(methods).map(([key, val]) => (
          <button key={key} onClick={() => setMethod(key)}
            className={`px-3 py-1 rounded-lg text-sm transition ${method === key ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20 text-sm space-y-2">
        <p className="text-gray-600 dark:text-gray-400">{m.desc}</p>
        <p className="text-xs font-mono bg-white dark:bg-gray-800 p-2 rounded">{m.example}</p>
        <p className="text-xs text-violet-600 dark:text-violet-400 font-medium">{m.improvement}</p>
      </div>
    </div>
  )
}

export default function ChainOfThoughtReasoning() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Chain-of-thought (CoT) prompting transforms reasoning problems by having the model
        generate intermediate steps before the final answer. This converts single-step prediction
        into multi-step computation, effectively giving the model "thinking time."
      </p>

      <DefinitionBlock title="Chain-of-Thought as Computation">
        <p>Standard prediction directly maps input to answer: <InlineMath math="p(a|q)" />. CoT introduces intermediate reasoning tokens <InlineMath math="r_1, \ldots, r_n" />:</p>
        <BlockMath math="p(a|q) = \sum_{r_1, \ldots, r_n} p(r_1|q) \cdot p(r_2|q, r_1) \cdots p(a|q, r_1, \ldots, r_n)" />
        <p className="mt-2">Each reasoning token adds computation. A model generating 100 reasoning tokens performs ~100x more FLOPs than direct answering. This trades inference compute for accuracy.</p>
      </DefinitionBlock>

      <CoTMethodComparison />

      <ExampleBlock title="When Does CoT Help?">
        <p>CoT is most effective for multi-step reasoning tasks:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li><strong>Math:</strong> GSM8K +35% (PaLM 540B), MATH +15%</li>
          <li><strong>Logic:</strong> Symbolic reasoning, constraint satisfaction</li>
          <li><strong>Common sense:</strong> Multi-hop questions requiring world knowledge</li>
          <li><strong>Minimal help:</strong> Factual recall, sentiment analysis, simple classification</li>
          <li><strong>Key condition:</strong> Model must be large enough (~100B+) to generate coherent reasoning</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Implementing Chain-of-Thought with Verification"
        code={`def chain_of_thought_with_verification(llm, question, max_retries=3):
    """Generate CoT reasoning with self-verification.

    1. Generate step-by-step solution
    2. Ask model to verify each step
    3. If verification fails, regenerate from the error
    """
    for attempt in range(max_retries):
        # Step 1: Generate CoT
        cot_prompt = f"""Solve step by step:
Q: {question}
A: Let me work through this carefully."""
        reasoning = llm.generate(cot_prompt)

        # Step 2: Verify
        verify_prompt = f"""Verify this solution step by step.
Question: {question}
Solution: {reasoning}

Check each step. Is the final answer correct? Reply YES or NO with explanation."""
        verification = llm.generate(verify_prompt)

        if "YES" in verification.upper():
            return reasoning, attempt + 1

        # Step 3: Self-correct
        print(f"Attempt {attempt + 1} failed verification, retrying...")

    return reasoning, max_retries  # Return best attempt

# Compute tokens used
def estimate_cot_tokens(direct_tokens=5, cot_tokens=150, verification_tokens=100):
    """Compare compute for direct vs CoT with verification."""
    direct_flops = direct_tokens
    cot_flops = cot_tokens + verification_tokens
    print(f"Direct answer: ~{direct_tokens} tokens")
    print(f"CoT + verify: ~{cot_tokens + verification_tokens} tokens")
    print(f"Compute multiplier: {cot_flops / direct_flops:.0f}x")
    print(f"Accuracy improvement: typically 20-40% on reasoning tasks")

estimate_cot_tokens()`}
      />

      <NoteBlock type="note" title="Reasoning Models (o1, R1)">
        <p>
          Models like OpenAI's o1 and DeepSeek R1 are trained to produce long internal reasoning
          chains before answering. Unlike prompting-based CoT, these models learn <em>when</em> and
          <em>how</em> to reason through reinforcement learning. They often outperform larger models
          that use standard prompting, demonstrating that test-time compute can substitute for
          model scale.
        </p>
      </NoteBlock>
    </div>
  )
}

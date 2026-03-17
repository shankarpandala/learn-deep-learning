import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ReasoningMethodComparison() {
  const [method, setMethod] = useState('cot')
  const methods = {
    cot: { name: 'Chain-of-Thought', desc: 'Linear step-by-step reasoning in a single pass.', branches: 1, verifications: 0, strength: 'Simple, no extra infrastructure' },
    sc: { name: 'Self-Consistency', desc: 'Sample multiple CoT paths, take majority vote.', branches: 'K (e.g., 40)', verifications: 0, strength: 'Robust to individual errors' },
    tot: { name: 'Tree-of-Thought', desc: 'Explore multiple reasoning branches with backtracking.', branches: 'b^d (branching)', verifications: 'At each node', strength: 'Handles complex search problems' },
  }
  const m = methods[method]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Reasoning Strategies Comparison</h3>
      <div className="flex gap-2 mb-3 flex-wrap">
        {Object.entries(methods).map(([key, val]) => (
          <button key={key} onClick={() => setMethod(key)}
            className={`px-3 py-1 rounded-lg text-sm transition ${method === key ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20 text-sm space-y-1">
        <p className="font-medium text-violet-700 dark:text-violet-300">{m.name}</p>
        <p className="text-gray-600 dark:text-gray-400">{m.desc}</p>
        <p className="text-xs text-gray-500">Reasoning paths: {m.branches} | Verifications: {m.verifications} | Strength: {m.strength}</p>
      </div>
    </div>
  )
}

export default function ReasoningPlanning() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Chain-of-thought prompting unlocked LLM reasoning, but more sophisticated strategies —
        self-consistency, tree-of-thought, and process reward models — push reasoning accuracy
        further by exploring multiple solution paths and verifying intermediate steps.
      </p>

      <DefinitionBlock title="Self-Consistency Decoding">
        <p>Sample <InlineMath math="K" /> independent chain-of-thought solutions and take the majority vote on the final answer:</p>
        <BlockMath math="\hat{a} = \arg\max_{a} \sum_{k=1}^{K} \mathbb{1}[\text{answer}(r_k) = a], \quad r_k \sim p(r | q, T)" />
        <p className="mt-2">where <InlineMath math="r_k" /> are sampled reasoning chains at temperature <InlineMath math="T > 0" />. This exploits the fact that correct reasoning paths are more common than any specific incorrect path.</p>
      </DefinitionBlock>

      <ReasoningMethodComparison />

      <ExampleBlock title="Process Reward Models (PRM)">
        <p>Instead of only scoring the final answer, PRMs score each reasoning step:</p>
        <BlockMath math="\text{PRM}(r) = \prod_{i=1}^{n} p(\text{step}_i \text{ is correct} | \text{step}_{1:i})" />
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>Outcome Reward Model (ORM): only scores the final answer</li>
          <li>Process Reward Model (PRM): scores every intermediate step</li>
          <li>PRM + best-of-N: sample N solutions, pick the one with highest PRM score</li>
          <li>On MATH benchmark: PRM + best-of-1860 achieves 78.2% (vs 72.4% majority voting)</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Self-Consistency with Majority Voting"
        code={`import random
from collections import Counter

def self_consistency(generate_cot, question, K=40, temperature=0.7):
    """Self-consistency decoding: sample K chains, majority vote.

    Args:
        generate_cot: function(question, temp) -> (reasoning, answer)
        question: the input question
        K: number of samples
        temperature: sampling temperature
    Returns:
        best_answer: majority vote answer
        confidence: fraction of samples agreeing
    """
    answers = []
    for _ in range(K):
        reasoning, answer = generate_cot(question, temperature)
        answers.append(answer)

    # Majority voting
    counts = Counter(answers)
    best_answer = counts.most_common(1)[0][0]
    confidence = counts[best_answer] / K
    return best_answer, confidence

# Simulate: correct answer appears more often in diverse samples
def mock_cot(q, temp):
    # 70% chance of correct reasoning at high temperature
    answer = "42" if random.random() < 0.7 else random.choice(["41", "43", "44"])
    return "...", answer

ans, conf = self_consistency(mock_cot, "What is 6 * 7?", K=40)
print(f"Answer: {ans}, Confidence: {conf:.2f}")
# With 70% per-sample accuracy, majority vote >> individual accuracy`}
      />

      <NoteBlock type="note" title="Test-Time Compute Scaling">
        <p>
          Self-consistency, tree-of-thought, and PRM-guided search all trade inference compute
          for accuracy. This creates a new scaling axis: instead of making models larger, spend
          more compute at inference time. For many reasoning tasks, doubling inference compute
          is more effective than doubling model parameters.
        </p>
      </NoteBlock>
    </div>
  )
}

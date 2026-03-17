import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function BestOfNSimulator() {
  const [baseAccuracy, setBaseAccuracy] = useState(0.5)
  const [N, setN] = useState(16)
  const bonAccuracy = 1 - Math.pow(1 - baseAccuracy, N)
  const oracleBoN = bonAccuracy
  const majorityAccuracy = Math.min(0.99, baseAccuracy + 0.15 * Math.log2(N))

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Best-of-N vs Majority Voting</h3>
      <div className="flex items-center gap-4 mb-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Base accuracy: {(baseAccuracy * 100).toFixed(0)}%
          <input type="range" min={0.1} max={0.9} step={0.05} value={baseAccuracy} onChange={e => setBaseAccuracy(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          N samples: {N}
          <input type="range" min={1} max={128} step={1} value={N} onChange={e => setN(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <div className="grid grid-cols-3 gap-3 text-sm text-center">
        <div className="p-2 rounded bg-gray-100 dark:bg-gray-800">
          <p className="text-gray-500 font-medium">Single Sample</p>
          <p className="text-lg font-bold">{(baseAccuracy * 100).toFixed(0)}%</p>
        </div>
        <div className="p-2 rounded bg-violet-50 dark:bg-violet-900/20">
          <p className="text-violet-700 dark:text-violet-300 font-medium">Majority Vote (N={N})</p>
          <p className="text-lg font-bold">{(majorityAccuracy * 100).toFixed(1)}%</p>
        </div>
        <div className="p-2 rounded bg-violet-100 dark:bg-violet-900/40">
          <p className="text-violet-700 dark:text-violet-300 font-medium">Oracle Best-of-{N}</p>
          <p className="text-lg font-bold">{(oracleBoN * 100).toFixed(1)}%</p>
        </div>
      </div>
      <p className="mt-2 text-xs text-gray-500 text-center">Oracle BoN assumes a perfect verifier. Real BoN performance depends on verifier quality.</p>
    </div>
  )
}

export default function SearchVerification() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Search and verification strategies sample multiple candidate solutions and use a reward
        model or verifier to select the best one. This decouples generation quality from
        selection quality, enabling significant accuracy improvements at inference time.
      </p>

      <DefinitionBlock title="Best-of-N with Reward Model">
        <p>Generate <InlineMath math="N" /> candidate solutions and select the one with the highest reward model score:</p>
        <BlockMath math="\hat{a} = \arg\max_{a_i} R(q, a_i), \quad a_i \sim p_\theta(\cdot | q), \quad i = 1, \ldots, N" />
        <p className="mt-2">With an oracle verifier and base accuracy <InlineMath math="p" />, the probability that at least one of <InlineMath math="N" /> samples is correct is:</p>
        <BlockMath math="P(\text{at least one correct}) = 1 - (1-p)^N" />
      </DefinitionBlock>

      <BestOfNSimulator />

      <ExampleBlock title="Process Reward Models (PRM) vs Outcome Reward Models (ORM)">
        <p>Two types of reward models for verifying solutions:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li><strong>ORM:</strong> Scores only the final answer. Cheap to train but can be fooled by lucky wrong reasoning.</li>
          <li><strong>PRM:</strong> Scores each intermediate step. More robust but requires step-level labels.</li>
          <li>On MATH: PRM + BoN-1860 achieves 78.2% vs ORM + BoN-1860 at 72.4%</li>
          <li>PRM enables step-level search (beam search over reasoning steps)</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Best-of-N with Process Reward Model"
        code={`import random
import math

def best_of_n_with_prm(generator, prm, question, N=64):
    """Best-of-N sampling with process reward model scoring.

    Args:
        generator: function(question) -> (steps, final_answer)
        prm: function(question, steps) -> step_scores
        question: input question
        N: number of samples
    """
    candidates = []
    for _ in range(N):
        steps, answer = generator(question)
        step_scores = prm(question, steps)
        # PRM score = product of step correctness probabilities
        # (or min, to catch any bad step)
        total_score = min(step_scores)  # Conservative: worst step
        candidates.append((answer, total_score, steps))

    # Select highest-scoring candidate
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]

# Simulate search scaling
def simulate_bon_scaling(base_acc=0.5, verifier_acc=0.85, max_n=256):
    """Show how accuracy scales with N samples and verifier quality."""
    print(f"Base accuracy: {base_acc*100:.0f}%, Verifier accuracy: {verifier_acc*100:.0f}%")
    print(f"{'N':>6} | {'Oracle BoN':>10} | {'Real BoN':>10} | {'Majority':>10}")
    print("-" * 45)
    for n in [1, 4, 16, 64, 256]:
        if n > max_n:
            break
        oracle = 1 - (1 - base_acc)**n
        # Real BoN: limited by verifier imperfection
        real = oracle * verifier_acc + (1 - oracle) * (1 - verifier_acc) * base_acc
        # Majority: improves with sqrt(N) roughly
        majority = min(0.99, base_acc + 0.15 * math.log2(max(n, 1)))
        print(f"{n:>6} | {oracle*100:>9.1f}% | {real*100:>9.1f}% | {majority*100:>9.1f}%")

simulate_bon_scaling()`}
      />

      <NoteBlock type="note" title="MCTS for LLM Reasoning">
        <p>
          Monte Carlo Tree Search (MCTS), the technique behind AlphaGo, is being applied to
          LLM reasoning. Each node in the tree is a partial reasoning chain, and the tree is
          expanded by generating next steps and scoring them with a value model. This enables
          exploration of diverse reasoning strategies while pruning unpromising paths early.
        </p>
      </NoteBlock>
    </div>
  )
}

import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ValueIterationViz() {
  const [iteration, setIteration] = useState(0)
  const gamma = 0.9
  const values = [
    [0, 0, 0, 0],
    [-1, -1, -1, 0],
    [-1.9, -1.9, -1, 0],
    [-2.71, -1.9, -1, 0],
    [-2.71, -2.71, -1, 0],
  ]
  const row = values[Math.min(iteration, values.length - 1)]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Value Iteration (1D Grid)</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Iteration: {iteration}
          <input type="range" min={0} max={4} step={1} value={iteration} onChange={e => setIteration(parseInt(e.target.value))} className="w-32 accent-violet-500" />
        </label>
      </div>
      <div className="flex justify-center gap-2">
        {row.map((v, i) => (
          <div key={i} className={`w-20 h-16 rounded-lg flex flex-col items-center justify-center text-sm font-mono ${i === 3 ? 'bg-violet-100 dark:bg-violet-900/40 border-2 border-violet-500' : 'bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-600'}`}>
            <span className="text-xs text-gray-500">S{i}</span>
            <span className="font-bold text-violet-700 dark:text-violet-400">{v.toFixed(2)}</span>
          </div>
        ))}
      </div>
      <p className="text-center text-xs text-gray-500 mt-2">S3 is the goal (V=0). Each step costs -1.</p>
    </div>
  )
}

export default function DynamicProgramming() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        When the MDP model is fully known, dynamic programming methods can compute
        optimal policies exactly. Value iteration and policy iteration are the two classical algorithms.
      </p>

      <DefinitionBlock title="Value Iteration">
        <p>Repeatedly apply the Bellman optimality operator until convergence:</p>
        <BlockMath math="V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V_k(s')\right]" />
        <p className="mt-2">Converges to <InlineMath math="V^*" /> as <InlineMath math="k \to \infty" /> due to the contraction mapping theorem.</p>
      </DefinitionBlock>

      <TheoremBlock title="Contraction Mapping" id="contraction">
        <p>The Bellman optimality operator <InlineMath math="\mathcal{T}" /> is a <InlineMath math="\gamma" />-contraction in the sup-norm:</p>
        <BlockMath math="\|\mathcal{T}V_1 - \mathcal{T}V_2\|_\infty \le \gamma \|V_1 - V_2\|_\infty" />
        <p className="mt-2">By the Banach fixed-point theorem, iteration converges at rate <InlineMath math="O(\gamma^k)" />.</p>
      </TheoremBlock>

      <ValueIterationViz />

      <DefinitionBlock title="Policy Iteration">
        <p>Alternates between two steps:</p>
        <BlockMath math="\text{1. Evaluate: } V^{\pi_k}(s) = \sum_a \pi_k(a|s)\sum_{s'}P(s'|s,a)[R + \gamma V^{\pi_k}(s')]" />
        <BlockMath math="\text{2. Improve: } \pi_{k+1}(s) = \arg\max_a \sum_{s'}P(s'|s,a)[R + \gamma V^{\pi_k}(s')]" />
      </DefinitionBlock>

      <ExampleBlock title="Convergence Comparison">
        <p>For a 100-state MDP with <InlineMath math="\gamma=0.99" />:</p>
        <p><strong>Value iteration</strong>: ~500 iterations to converge (each iteration is cheap).</p>
        <p><strong>Policy iteration</strong>: ~10 iterations (each requires solving a linear system).</p>
        <p>Policy iteration often converges in fewer outer loops but each step is more expensive.</p>
      </ExampleBlock>

      <PythonCode
        title="Value Iteration Implementation"
        code={`import numpy as np

def value_iteration(P, R, gamma=0.99, theta=1e-8):
    """P: (S,A,S') transition probs, R: (S,A) rewards"""
    n_states, n_actions, _ = P.shape
    V = np.zeros(n_states)
    for i in range(10000):
        V_new = np.max(
            np.sum(P * (R[:, :, None] + gamma * V[None, None, :]), axis=2),
            axis=1
        )
        if np.max(np.abs(V_new - V)) < theta:
            print(f"Converged in {i+1} iterations")
            break
        V = V_new
    policy = np.argmax(
        np.sum(P * (R[:, :, None] + gamma * V[None, None, :]), axis=2),
        axis=1
    )
    return V, policy

# Small 3-state, 2-action MDP
P = np.array([
    [[0.7, 0.3, 0.0], [0.0, 0.5, 0.5]],
    [[0.0, 0.9, 0.1], [0.0, 0.0, 1.0]],
    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
])
R = np.array([[1.0, 0.5], [0.5, -1.0], [0.0, 0.0]])
V, pi = value_iteration(P, R)
print("Optimal values:", V.round(3))
print("Optimal policy:", pi)`}
      />

      <NoteBlock type="note" title="From DP to Deep RL">
        <p>
          Dynamic programming requires a complete model <InlineMath math="P(s'|s,a)" />. In practice,
          this is rarely available. Modern RL methods replace exact computation with sampling
          (Monte Carlo, TD learning) and function approximation (neural networks), connecting
          these classical ideas to deep learning.
        </p>
      </NoteBlock>
    </div>
  )
}

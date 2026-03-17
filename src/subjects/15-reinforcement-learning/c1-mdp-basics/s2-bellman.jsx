import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function BellmanBackup() {
  const [gamma, setGamma] = useState(0.9)
  const [reward, setReward] = useState(1.0)
  const [nextV, setNextV] = useState(5.0)
  const backup = reward + gamma * nextV

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Bellman Backup Calculator</h3>
      <div className="flex flex-wrap items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          gamma = {gamma.toFixed(2)}
          <input type="range" min={0} max={0.99} step={0.01} value={gamma} onChange={e => setGamma(parseFloat(e.target.value))} className="w-24 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          r = {reward.toFixed(1)}
          <input type="range" min={-5} max={5} step={0.1} value={reward} onChange={e => setReward(parseFloat(e.target.value))} className="w-24 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          V(s') = {nextV.toFixed(1)}
          <input type="range" min={0} max={10} step={0.1} value={nextV} onChange={e => setNextV(parseFloat(e.target.value))} className="w-24 accent-violet-500" />
        </label>
      </div>
      <p className="text-center text-violet-700 dark:text-violet-400 font-mono text-lg">
        V(s) = {reward.toFixed(1)} + {gamma.toFixed(2)} x {nextV.toFixed(1)} = <strong>{backup.toFixed(2)}</strong>
      </p>
    </div>
  )
}

export default function BellmanEquations() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The Bellman equations express a recursive relationship between the value of a state and
        the values of its successor states. They are the backbone of almost every RL algorithm.
      </p>

      <DefinitionBlock title="State-Value Function">
        <p>The value of state <InlineMath math="s" /> under policy <InlineMath math="\pi" />:</p>
        <BlockMath math="V^\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s\right]" />
      </DefinitionBlock>

      <DefinitionBlock title="Action-Value Function">
        <p>The value of taking action <InlineMath math="a" /> in state <InlineMath math="s" /> under policy <InlineMath math="\pi" />:</p>
        <BlockMath math="Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s, A_t = a\right]" />
      </DefinitionBlock>

      <TheoremBlock title="Bellman Expectation Equation" id="bellman-expectation">
        <BlockMath math="V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V^\pi(s')\right]" />
        <p className="mt-2">This decomposes the value into an immediate reward plus the discounted future value.</p>
      </TheoremBlock>

      <TheoremBlock title="Bellman Optimality Equation" id="bellman-optimality">
        <BlockMath math="V^*(s) = \max_a \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V^*(s')\right]" />
        <p className="mt-2">And for the optimal action-value function:</p>
        <BlockMath math="Q^*(s,a) = \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')\right]" />
      </TheoremBlock>

      <BellmanBackup />

      <ExampleBlock title="Two-State MDP">
        <p>State A: action 'stay' gives r=2 and stays in A. Action 'go' gives r=0 and moves to B (terminal).</p>
        <BlockMath math="V^*(A) = \max\{2 + \gamma V^*(A),\; 0\} = \frac{2}{1-\gamma}" />
        <p>With <InlineMath math="\gamma=0.9" />: <InlineMath math="V^*(A)=20" />.</p>
      </ExampleBlock>

      <PythonCode
        title="Solving Bellman Equations with Linear Algebra"
        code={`import numpy as np

# 3-state MDP: transition matrix under a fixed policy
P = np.array([
    [0.7, 0.3, 0.0],  # state 0 transitions
    [0.0, 0.6, 0.4],  # state 1 transitions
    [0.0, 0.0, 1.0],  # state 2 (terminal)
])
R = np.array([1.0, 0.5, 0.0])  # expected rewards
gamma = 0.9

# Bellman equation: V = R + gamma * P @ V
# => (I - gamma * P) @ V = R
V = np.linalg.solve(np.eye(3) - gamma * P, R)
print("State values:", V.round(3))
# V[0] should be highest since it collects reward longest`}
      />

      <NoteBlock type="note" title="Bellman Equations in Practice">
        <p>
          Direct matrix solution has <InlineMath math="O(|\mathcal{S}|^3)" /> complexity, making it infeasible
          for large state spaces. In practice, iterative methods (value iteration, TD learning) or
          function approximation (deep RL) are used instead.
        </p>
      </NoteBlock>
    </div>
  )
}

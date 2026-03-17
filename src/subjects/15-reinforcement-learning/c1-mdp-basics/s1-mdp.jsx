import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function MDPDiagram() {
  const [step, setStep] = useState(0)
  const states = ['S0', 'S1', 'S2', 'S3']
  const rewards = [0, +1, -1, +5]
  const actions = ['right', 'right', 'right']
  const positions = [60, 150, 240, 330]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">MDP Trajectory Visualizer</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Step: {step}
          <input type="range" min={0} max={3} step={1} value={step} onChange={e => setStep(parseInt(e.target.value))} className="w-32 accent-violet-500" />
        </label>
        <span className="text-sm text-violet-600 dark:text-violet-400 font-medium">
          Reward: {rewards[step]} | Cumulative: {rewards.slice(0, step + 1).reduce((a, b) => a + b, 0)}
        </span>
      </div>
      <svg width={400} height={100} className="mx-auto block">
        {states.map((s, i) => (
          <g key={s}>
            <circle cx={positions[i]} cy={50} r={22} fill={i === step ? '#7c3aed' : '#e5e7eb'} stroke="#7c3aed" strokeWidth={2} />
            <text x={positions[i]} y={55} textAnchor="middle" fill={i === step ? 'white' : '#374151'} fontSize={13} fontWeight="bold">{s}</text>
            {i < 3 && <line x1={positions[i] + 24} y1={50} x2={positions[i + 1] - 24} y2={50} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#arrow)" />}
            <text x={positions[i]} y={88} textAnchor="middle" fill="#7c3aed" fontSize={11}>r={rewards[i]}</text>
          </g>
        ))}
        <defs><marker id="arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#9ca3af" /></marker></defs>
      </svg>
    </div>
  )
}

export default function MDP() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        A Markov Decision Process (MDP) provides the mathematical framework for sequential
        decision-making under uncertainty. Nearly all of reinforcement learning builds on this foundation.
      </p>

      <DefinitionBlock title="Markov Decision Process">
        <p>An MDP is a tuple <InlineMath math="(\mathcal{S}, \mathcal{A}, P, R, \gamma)" /> where:</p>
        <BlockMath math="\mathcal{S} \text{ (states)}, \quad \mathcal{A} \text{ (actions)}, \quad P(s'|s,a) \text{ (transitions)}, \quad R(s,a) \text{ (reward)}, \quad \gamma \in [0,1) \text{ (discount)}" />
        <p className="mt-2">The <strong>Markov property</strong>: <InlineMath math="P(s_{t+1}|s_t, a_t) = P(s_{t+1}|s_0,...,s_t, a_0,...,a_t)" /></p>
      </DefinitionBlock>

      <DefinitionBlock title="Policy">
        <p>A policy <InlineMath math="\pi" /> maps states to action distributions:</p>
        <BlockMath math="\pi(a|s) = P(A_t = a \mid S_t = s)" />
        <p className="mt-2">The goal is to find <InlineMath math="\pi^*" /> that maximizes the expected discounted return:</p>
        <BlockMath math="G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}" />
      </DefinitionBlock>

      <MDPDiagram />

      <ExampleBlock title="GridWorld MDP">
        <p>Consider a 4x4 grid. The agent starts at (0,0) and the goal is (3,3).</p>
        <BlockMath math="\mathcal{S} = \{(i,j) : 0 \le i,j \le 3\}, \quad \mathcal{A} = \{\uparrow, \downarrow, \leftarrow, \rightarrow\}" />
        <p>With <InlineMath math="R = -1" /> per step and <InlineMath math="R = 0" /> at the goal, the agent learns to take the shortest path.</p>
      </ExampleBlock>

      <PythonCode
        title="Simple MDP Environment in Python"
        code={`import numpy as np

class GridWorldMDP:
    def __init__(self, size=4, gamma=0.99):
        self.size = size
        self.gamma = gamma
        self.state = (0, 0)
        self.goal = (size - 1, size - 1)
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R, L, D, U

    def step(self, action_idx):
        dr, dc = self.actions[action_idx]
        r = max(0, min(self.size - 1, self.state[0] + dr))
        c = max(0, min(self.size - 1, self.state[1] + dc))
        self.state = (r, c)
        done = self.state == self.goal
        reward = 0.0 if done else -1.0
        return self.state, reward, done

    def reset(self):
        self.state = (0, 0)
        return self.state

env = GridWorldMDP()
state = env.reset()
total_reward = 0
for _ in range(20):
    action = np.random.randint(4)  # random policy
    state, reward, done = env.step(action)
    total_reward += reward
    if done:
        break
print(f"Final state: {state}, Total reward: {total_reward}")`}
      />

      <NoteBlock type="note" title="Why Discount?">
        <p>
          The discount factor <InlineMath math="\gamma" /> serves two purposes: it ensures the
          infinite sum converges and encodes a preference for sooner rewards. With <InlineMath math="\gamma = 0" />,
          the agent is myopic; with <InlineMath math="\gamma \to 1" />, it plans far ahead.
        </p>
      </NoteBlock>
    </div>
  )
}

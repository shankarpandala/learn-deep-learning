import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function QTableViz() {
  const [alpha, setAlpha] = useState(0.1)
  const [epsilon, setEpsilon] = useState(0.1)
  const initQ = [[0, 0], [0, 0], [0, 0], [0, 0]]
  const [qTable, setQTable] = useState(initQ)
  const [stepCount, setStepCount] = useState(0)

  const doStep = () => {
    const s = Math.floor(Math.random() * 3)
    const a = Math.random() < epsilon ? Math.floor(Math.random() * 2) : (qTable[s][0] >= qTable[s][1] ? 0 : 1)
    const r = s === 2 && a === 0 ? 1.0 : -0.1
    const sp = Math.min(s + 1, 3)
    const maxQn = Math.max(qTable[sp][0], qTable[sp][1])
    const newQ = qTable.map(row => [...row])
    newQ[s][a] = newQ[s][a] + alpha * (r + 0.9 * maxQn - newQ[s][a])
    setQTable(newQ)
    setStepCount(stepCount + 1)
  }

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Q-Table Update Simulator</h3>
      <div className="flex flex-wrap items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          alpha = {alpha.toFixed(2)} <input type="range" min={0.01} max={1} step={0.01} value={alpha} onChange={e => setAlpha(parseFloat(e.target.value))} className="w-20 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          epsilon = {epsilon.toFixed(2)} <input type="range" min={0} max={1} step={0.01} value={epsilon} onChange={e => setEpsilon(parseFloat(e.target.value))} className="w-20 accent-violet-500" />
        </label>
        <button onClick={doStep} className="px-3 py-1 rounded bg-violet-500 text-white text-sm hover:bg-violet-600">Step</button>
        <button onClick={() => { setQTable(initQ.map(r => [...r])); setStepCount(0) }} className="px-3 py-1 rounded bg-gray-300 text-gray-700 text-sm hover:bg-gray-400">Reset</button>
        <span className="text-sm text-gray-500">Steps: {stepCount}</span>
      </div>
      <table className="mx-auto text-sm border-collapse">
        <thead><tr><th className="px-3 py-1 text-gray-600">State</th><th className="px-3 py-1 text-violet-600">Q(s,a0)</th><th className="px-3 py-1 text-violet-600">Q(s,a1)</th></tr></thead>
        <tbody>{qTable.map((row, i) => (
          <tr key={i}><td className="px-3 py-1 text-center font-mono">S{i}</td>{row.map((v, j) => (
            <td key={j} className="px-3 py-1 text-center font-mono text-violet-700 dark:text-violet-400">{v.toFixed(3)}</td>
          ))}</tr>
        ))}</tbody>
      </table>
    </div>
  )
}

export default function QLearning() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Q-learning is an off-policy temporal difference method that learns the optimal action-value
        function directly, without requiring a model of the environment.
      </p>

      <DefinitionBlock title="Q-Learning Update Rule">
        <BlockMath math="Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]" />
        <p className="mt-2">
          The term <InlineMath math="r + \gamma \max_{a'} Q(s',a') - Q(s,a)" /> is the <strong>TD error</strong>.
          The key insight: we use <InlineMath math="\max" /> over next actions regardless of what action
          was actually taken (off-policy).
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="Epsilon-Greedy Exploration">
        <BlockMath math="a = \begin{cases} \arg\max_a Q(s,a) & \text{with probability } 1-\varepsilon \\ \text{random action} & \text{with probability } \varepsilon \end{cases}" />
      </DefinitionBlock>

      <QTableViz />

      <ExampleBlock title="Cliff Walking">
        <p>In the cliff walking problem, Q-learning learns the optimal path along the cliff edge
        (shortest but risky), while SARSA learns a safer path further from the cliff because
        it accounts for its own exploratory behavior.</p>
      </ExampleBlock>

      <PythonCode
        title="Tabular Q-Learning"
        code={`import numpy as np

def q_learning(env, n_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = np.zeros((env.n_states, env.n_actions))
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                a = np.random.randint(env.n_actions)
            else:
                a = np.argmax(Q[s])
            s_next, reward, done = env.step(a)
            # Q-learning update (off-policy: uses max over next actions)
            td_error = reward + gamma * np.max(Q[s_next]) - Q[s]
            Q[s, a] += alpha * td_error
            s = s_next
    return Q

# The key difference from SARSA:
# SARSA: Q[s,a] += alpha * (r + gamma * Q[s',a'] - Q[s,a])   (on-policy)
# Q-learning: Q[s,a] += alpha * (r + gamma * max Q[s',:] - Q[s,a])  (off-policy)`}
      />

      <WarningBlock title="Maximization Bias">
        <p>
          Q-learning's <InlineMath math="\max" /> operator introduces an upward bias in value estimates
          because <InlineMath math="\mathbb{E}[\max Q] \ge \max \mathbb{E}[Q]" />. This is the motivation
          for Double Q-learning and later Double DQN.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Tabular to Deep">
        <p>
          Tabular Q-learning stores one value per state-action pair, limiting it to small discrete
          spaces. Deep Q-Networks replace the table with a neural network <InlineMath math="Q_\theta(s,a)" />,
          enabling Q-learning on high-dimensional inputs like images.
        </p>
      </NoteBlock>
    </div>
  )
}

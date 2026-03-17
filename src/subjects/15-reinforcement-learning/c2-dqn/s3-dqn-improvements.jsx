import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function DuelingViz() {
  const [stateValue, setStateValue] = useState(5.0)
  const advantages = [-1.2, 0.0, 0.8, -0.3]
  const actions = ['Left', 'Stay', 'Right', 'Jump']
  const meanAdv = advantages.reduce((a, b) => a + b, 0) / advantages.length
  const qValues = advantages.map(a => stateValue + (a - meanAdv))

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Dueling DQN Decomposition</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          V(s) = {stateValue.toFixed(1)}
          <input type="range" min={0} max={10} step={0.1} value={stateValue} onChange={e => setStateValue(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <div className="flex flex-wrap gap-2 justify-center">
        {actions.map((act, i) => (
          <div key={i} className="px-3 py-2 rounded-lg bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 text-center">
            <div className="text-xs text-gray-500">{act}</div>
            <div className="text-xs text-violet-500">A={advantages[i].toFixed(1)}</div>
            <div className="font-bold text-violet-700 dark:text-violet-400 text-sm">Q={qValues[i].toFixed(2)}</div>
          </div>
        ))}
      </div>
      <p className="text-center text-xs text-gray-500 mt-2">Q(s,a) = V(s) + A(s,a) - mean(A)</p>
    </div>
  )
}

export default function DQNImprovements() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Several improvements to the basic DQN address its limitations: overestimation bias,
        inefficient replay sampling, and entangled value-advantage estimation. Rainbow combines
        six of these improvements into a single agent.
      </p>

      <DefinitionBlock title="Double DQN">
        <p>Decouples action selection from evaluation to reduce overestimation:</p>
        <BlockMath math="y = r + \gamma Q_{\theta^-}\!\left(s',\; \arg\max_{a'} Q_\theta(s', a')\right)" />
        <p className="mt-2">The online network <InlineMath math="\theta" /> selects the action; the target
        network <InlineMath math="\theta^-" /> evaluates it. This simple change significantly reduces
        the maximization bias of standard DQN.</p>
      </DefinitionBlock>

      <DefinitionBlock title="Prioritized Experience Replay">
        <p>Sample transitions proportional to their TD error magnitude:</p>
        <BlockMath math="P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \quad p_i = |\delta_i| + \varepsilon" />
        <p className="mt-2">Importance sampling weights correct the bias: <InlineMath math="w_i = (N \cdot P(i))^{-\beta}" /></p>
      </DefinitionBlock>

      <DefinitionBlock title="Dueling Architecture">
        <p>Decomposes Q into state value and advantage streams:</p>
        <BlockMath math="Q(s,a;\theta) = V(s;\theta_v) + A(s,a;\theta_a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a';\theta_a)" />
      </DefinitionBlock>

      <DuelingViz />

      <ExampleBlock title="Rainbow DQN Components">
        <p>Rainbow (Hessel et al., 2018) combines six improvements:</p>
        <ol className="list-decimal ml-5 mt-2 space-y-1">
          <li>Double DQN (reduce overestimation)</li>
          <li>Prioritized replay (focus on surprising transitions)</li>
          <li>Dueling networks (separate value and advantage)</li>
          <li>Multi-step returns (faster credit assignment)</li>
          <li>Distributional RL (model return distribution)</li>
          <li>Noisy networks (parameter-space exploration)</li>
        </ol>
      </ExampleBlock>

      <PythonCode
        title="Double DQN Update"
        code={`import torch
import torch.nn as nn

def double_dqn_loss(q_net, target_net, batch, gamma=0.99):
    states, actions, rewards, next_states, dones = batch

    # Current Q-values
    q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Double DQN: select action with online net, evaluate with target
        best_actions = q_net(next_states).argmax(dim=1, keepdim=True)
        next_q = target_net(next_states).gather(1, best_actions).squeeze(1)
        targets = rewards + gamma * next_q * (1 - dones.float())

    loss = nn.functional.smooth_l1_loss(q_values, targets)  # Huber loss
    return loss

# Comparison: standard DQN uses target_net for BOTH selection and evaluation
# next_q = target_net(next_states).max(dim=1)[0]  # overestimates!`}
      />

      <WarningBlock title="Hyperparameter Sensitivity">
        <p>
          DQN variants are notoriously sensitive to hyperparameters. Buffer size, target update
          frequency, learning rate, and epsilon schedule all interact. Prioritized replay adds
          <InlineMath math="\alpha" /> and <InlineMath math="\beta" /> annealing. Always start with
          published defaults before tuning.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Beyond Discrete Actions">
        <p>
          DQN methods require a discrete action space to compute <InlineMath math="\max_a Q(s,a)" />.
          For continuous actions, policy gradient and actor-critic methods (covered next) are needed.
        </p>
      </NoteBlock>
    </div>
  )
}

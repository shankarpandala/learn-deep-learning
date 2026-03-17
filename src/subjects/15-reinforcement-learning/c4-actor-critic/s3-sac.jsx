import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function EntropyTempViz() {
  const [alpha, setAlpha] = useState(0.2)
  const logits = [2.0, 1.0, 0.5, -0.5]
  const maxL = Math.max(...logits)
  const expL = logits.map(l => Math.exp((l - maxL) / Math.max(alpha, 0.01)))
  const sumExp = expL.reduce((a, b) => a + b, 0)
  const probs = expL.map(e => e / sumExp)
  const entropy = -probs.reduce((s, p) => s + (p > 0 ? p * Math.log(p) : 0), 0)
  const actions = ['a0', 'a1', 'a2', 'a3']

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Entropy Temperature Effect</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          alpha = {alpha.toFixed(2)}
          <input type="range" min={0.01} max={2.0} step={0.01} value={alpha} onChange={e => setAlpha(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <span className="text-sm text-violet-600 dark:text-violet-400">H = {entropy.toFixed(3)}</span>
      </div>
      <div className="flex gap-2 justify-center items-end" style={{ height: 80 }}>
        {actions.map((a, i) => (
          <div key={i} className="flex flex-col items-center">
            <div className="w-12 bg-violet-500 rounded-t" style={{ height: probs[i] * 70 }} />
            <span className="text-xs text-gray-500 mt-1">{a}</span>
            <span className="text-xs text-violet-600">{(probs[i] * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
      <p className="text-center text-xs text-gray-500 mt-2">Higher alpha = more uniform (exploratory) policy</p>
    </div>
  )
}

export default function SAC() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Soft Actor-Critic (SAC) is the leading off-policy actor-critic algorithm for continuous
        control. It maximizes a combination of expected return and policy entropy, encouraging
        robust exploration.
      </p>

      <DefinitionBlock title="Maximum Entropy RL Objective">
        <BlockMath math="J(\pi) = \sum_{t=0}^T \mathbb{E}\left[r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]" />
        <p className="mt-2">The temperature <InlineMath math="\alpha" /> balances reward maximization
        and entropy. The soft value functions become:</p>
        <BlockMath math="Q^{\text{soft}}(s,a) = r + \gamma \mathbb{E}_{s'}\left[V^{\text{soft}}(s')\right], \quad V^{\text{soft}}(s) = \mathbb{E}_{a \sim \pi}\left[Q(s,a) - \alpha \log \pi(a|s)\right]" />
      </DefinitionBlock>

      <TheoremBlock title="SAC Components" id="sac-components">
        <p>SAC maintains five networks:</p>
        <BlockMath math="\pi_\theta \text{ (actor)}, \quad Q_{\phi_1}, Q_{\phi_2} \text{ (twin critics)}, \quad \bar{Q}_{\phi_1}, \bar{Q}_{\phi_2} \text{ (target critics)}" />
        <p className="mt-2">Twin critics (clipped double Q) take the minimum to combat overestimation:</p>
        <BlockMath math="y = r + \gamma \left(\min_{i=1,2} Q_{\bar{\phi}_i}(s', \tilde{a}') - \alpha \log \pi(\tilde{a}'|s')\right)" />
      </TheoremBlock>

      <EntropyTempViz />

      <ExampleBlock title="Automatic Temperature Tuning">
        <p>SAC can automatically adjust <InlineMath math="\alpha" /> to maintain a target entropy:</p>
        <BlockMath math="\alpha^* = \arg\min_\alpha \mathbb{E}_{a \sim \pi}\left[-\alpha \log \pi(a|s) - \alpha \bar{\mathcal{H}}\right]" />
        <p>where <InlineMath math="\bar{\mathcal{H}} = -\dim(\mathcal{A})" /> is a common heuristic target for continuous actions.</p>
      </ExampleBlock>

      <PythonCode
        title="SAC Critic and Actor Updates"
        code={`import torch
import torch.nn as nn

def sac_critic_loss(q1, q2, target_q1, target_q2, policy,
                    states, actions, rewards, next_states, dones,
                    alpha=0.2, gamma=0.99):
    with torch.no_grad():
        next_actions, next_log_probs = policy.sample(next_states)
        q1_next = target_q1(next_states, next_actions)
        q2_next = target_q2(next_states, next_actions)
        min_q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
        target = rewards + gamma * (1 - dones) * min_q_next

    q1_pred = q1(states, actions)
    q2_pred = q2(states, actions)
    loss = nn.functional.mse_loss(q1_pred, target) + \\
           nn.functional.mse_loss(q2_pred, target)
    return loss

def sac_actor_loss(policy, q1, q2, states, alpha=0.2):
    actions, log_probs = policy.sample(states)
    q_val = torch.min(q1(states, actions), q2(states, actions))
    loss = (alpha * log_probs - q_val).mean()  # maximize Q - alpha*logpi
    return loss`}
      />

      <NoteBlock type="note" title="SAC vs PPO">
        <p>
          <strong>SAC</strong>: Off-policy, sample-efficient, best for continuous control (robotics,
          locomotion). <strong>PPO</strong>: On-policy, simpler, better for discrete actions and
          language model fine-tuning. SAC reuses past data via replay buffers; PPO discards
          data after each update.
        </p>
      </NoteBlock>
    </div>
  )
}

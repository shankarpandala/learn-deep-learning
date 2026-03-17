import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ClipViz() {
  const [epsilon, setEpsilon] = useState(0.2)
  const [advantage, setAdvantage] = useState(1.0)
  const W = 360, H = 180, ox = 60, oy = H - 30
  const sx = 120, sy = 50

  const range = Array.from({ length: 101 }, (_, i) => 0.2 + i * 0.026)
  const toX = r => ox + (r - 1) * sx
  const toY = v => oy - v * sy

  const unclipped = range.map(r => r * advantage)
  const clipped = range.map(r => Math.min(Math.max(r, 1 - epsilon), 1 + epsilon) * advantage)
  const ppoObj = range.map((r, i) => advantage >= 0 ? Math.min(unclipped[i], clipped[i]) : Math.max(unclipped[i], clipped[i]))

  const path = (vals) => range.map((r, i) => `${i === 0 ? 'M' : 'L'}${toX(r)},${toY(vals[i])}`).join(' ')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">PPO Clipped Objective</h3>
      <div className="flex flex-wrap items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          epsilon = {epsilon.toFixed(2)}
          <input type="range" min={0.05} max={0.5} step={0.01} value={epsilon} onChange={e => setEpsilon(parseFloat(e.target.value))} className="w-24 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          A = {advantage.toFixed(1)}
          <input type="range" min={-2} max={2} step={0.1} value={advantage} onChange={e => setAdvantage(parseFloat(e.target.value))} className="w-24 accent-violet-500" />
        </label>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={ox} y1={10} x2={ox} y2={oy + 10} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={ox - 10} y1={oy} x2={W - 10} y2={oy} stroke="#d1d5db" strokeWidth={0.5} />
        <path d={path(unclipped)} fill="none" stroke="#d1d5db" strokeWidth={1.5} strokeDasharray="4,3" />
        <path d={path(ppoObj)} fill="none" stroke="#7c3aed" strokeWidth={2.5} />
        <line x1={toX(1 - epsilon)} y1={10} x2={toX(1 - epsilon)} y2={oy} stroke="#f97316" strokeWidth={1} strokeDasharray="3,3" />
        <line x1={toX(1 + epsilon)} y1={10} x2={toX(1 + epsilon)} y2={oy} stroke="#f97316" strokeWidth={1} strokeDasharray="3,3" />
        <text x={toX(1)} y={oy + 18} textAnchor="middle" fill="#374151" fontSize={10}>r=1</text>
        <text x={W - 30} y={oy + 18} fill="#374151" fontSize={10}>r(theta)</text>
      </svg>
      <div className="flex justify-center gap-4 text-xs mt-1">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-violet-500" /> PPO objective</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-gray-400" style={{ borderTop: '1px dashed' }} /> Unclipped</span>
      </div>
    </div>
  )
}

export default function PPO() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Proximal Policy Optimization (PPO) is the most widely used RL algorithm in practice,
        powering everything from game AI to RLHF for language models. It achieves TRPO-like
        stability with a much simpler clipped surrogate objective.
      </p>

      <DefinitionBlock title="PPO-Clip Objective">
        <BlockMath math="L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\!\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]" />
        <p className="mt-2">where <InlineMath math="r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}" /> is the
        probability ratio. Typical <InlineMath math="\varepsilon = 0.2" />.</p>
      </DefinitionBlock>

      <ClipViz />

      <TheoremBlock title="How Clipping Works" id="clipping-mechanism">
        <p>When <InlineMath math="\hat{A} > 0" /> (good action): the objective is capped at <InlineMath math="(1+\varepsilon)\hat{A}" />,
        preventing the ratio from growing too large.</p>
        <p>When <InlineMath math="\hat{A} < 0" /> (bad action): the objective is capped at <InlineMath math="(1-\varepsilon)\hat{A}" />,
        preventing the ratio from shrinking too much.</p>
        <p className="mt-2">This creates a pessimistic bound: the policy cannot change too aggressively in either direction.</p>
      </TheoremBlock>

      <ExampleBlock title="PPO in Practice">
        <p>Standard PPO hyperparameters: <InlineMath math="\varepsilon=0.2" />, learning rate <InlineMath math="3 \times 10^{-4}" />,
        GAE <InlineMath math="\lambda=0.95" />, <InlineMath math="\gamma=0.99" />, minibatch size 64, 4 epochs per rollout.
        PPO is the algorithm behind InstructGPT and ChatGPT's RLHF stage.</p>
      </ExampleBlock>

      <PythonCode
        title="PPO Update Step"
        code={`import torch
import torch.nn as nn
from torch.distributions import Categorical

def ppo_update(model, optimizer, states, actions, old_log_probs,
               returns, advantages, clip_eps=0.2, epochs=4):
    for _ in range(epochs):
        logits, values = model(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Probability ratio
        ratio = torch.exp(log_probs - old_log_probs.detach())

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = nn.functional.mse_loss(values.squeeze(), returns)
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    return loss.item()`}
      />

      <NoteBlock type="note" title="PPO Variants">
        <p>
          PPO-Penalty uses an adaptive KL penalty instead of clipping. In practice, PPO-Clip
          dominates due to simplicity. For continuous control, the actor outputs Gaussian
          parameters <InlineMath math="(\mu, \sigma)" /> instead of categorical logits.
        </p>
      </NoteBlock>
    </div>
  )
}

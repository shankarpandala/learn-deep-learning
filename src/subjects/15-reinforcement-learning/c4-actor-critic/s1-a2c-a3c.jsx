import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ParallelEnvsViz() {
  const [nWorkers, setNWorkers] = useState(4)
  const colors = ['#7c3aed', '#f97316', '#10b981', '#f43f5e', '#3b82f6', '#eab308', '#8b5cf6', '#ec4899']

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">A3C Parallel Workers</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Workers: {nWorkers}
          <input type="range" min={1} max={8} step={1} value={nWorkers} onChange={e => setNWorkers(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <svg width={360} height={120} className="mx-auto block">
        <rect x={140} y={10} width={80} height={30} rx={6} fill="#7c3aed" opacity={0.15} stroke="#7c3aed" strokeWidth={1.5} />
        <text x={180} y={30} textAnchor="middle" fill="#7c3aed" fontSize={11} fontWeight="bold">Shared Model</text>
        {Array.from({ length: nWorkers }, (_, i) => {
          const x = 20 + (i * 320) / Math.max(nWorkers - 1, 1)
          return (
            <g key={i}>
              <line x1={180} y1={42} x2={x + 20} y2={65} stroke={colors[i % colors.length]} strokeWidth={1} opacity={0.5} />
              <rect x={x} y={68} width={40} height={40} rx={5} fill={colors[i % colors.length]} opacity={0.2} stroke={colors[i % colors.length]} strokeWidth={1.5} />
              <text x={x + 20} y={85} textAnchor="middle" fill={colors[i % colors.length]} fontSize={9}>Env {i}</text>
              <text x={x + 20} y={100} textAnchor="middle" fill={colors[i % colors.length]} fontSize={8}>Worker</text>
            </g>
          )
        })}
      </svg>
      <p className="text-center text-xs text-gray-500 mt-1">Each worker collects experience and sends gradients to the shared model</p>
    </div>
  )
}

export default function A2CA3C() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Actor-critic methods combine policy gradients (actor) with learned value functions (critic).
        A2C synchronizes parallel workers; A3C uses asynchronous gradient updates.
      </p>

      <DefinitionBlock title="Actor-Critic Architecture">
        <p>Two networks (often sharing a backbone):</p>
        <BlockMath math="\text{Actor: } \pi_\theta(a|s) \qquad \text{Critic: } V_\phi(s)" />
        <p className="mt-2">The actor is updated with the policy gradient using advantage from the critic:</p>
        <BlockMath math="\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot \hat{A}(s,a)\right]" />
        <p>The critic minimizes <InlineMath math="\|V_\phi(s) - G_t\|^2" /> or uses TD targets.</p>
      </DefinitionBlock>

      <TheoremBlock title="A3C: Asynchronous Advantage Actor-Critic" id="a3c">
        <p>Key idea: run multiple workers in parallel, each with its own environment copy. Each worker:</p>
        <BlockMath math="\text{1. Copy global params} \to \text{2. Collect n-step data} \to \text{3. Compute gradients} \to \text{4. Update global params}" />
        <p className="mt-2">Asynchronous updates provide implicit exploration through parameter diversity across workers.</p>
      </TheoremBlock>

      <ParallelEnvsViz />

      <ExampleBlock title="A2C vs A3C">
        <p><strong>A3C</strong>: Workers update asynchronously (stale gradients possible). Simpler to scale across CPUs.</p>
        <p><strong>A2C</strong>: Workers synchronize before each update (no stale gradients). Often preferred because
        it is easier to implement with GPUs and gives equivalent or better results.</p>
      </ExampleBlock>

      <PythonCode
        title="A2C Implementation"
        code={`import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)

def a2c_update(model, optimizer, states, actions, returns, gamma=0.99):
    logits, values = model(states)
    values = values.squeeze(-1)
    dist = Categorical(logits=logits)
    log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    advantages = returns - values.detach()
    actor_loss = -(log_probs * advantages).mean()
    critic_loss = nn.functional.mse_loss(values, returns)
    loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy  # entropy bonus

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    return loss.item()`}
      />

      <NoteBlock type="note" title="Entropy Bonus">
        <p>
          Adding an entropy term <InlineMath math="-\beta H(\pi)" /> to the loss encourages exploration
          by preventing the policy from becoming too deterministic too early. Typical values:
          <InlineMath math="\beta = 0.01" /> for discrete actions.
        </p>
      </NoteBlock>
    </div>
  )
}

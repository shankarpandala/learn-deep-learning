import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function GradientVarianceViz() {
  const [nSamples, setNSamples] = useState(10)
  const [seed, setSeed] = useState(0)
  const estimates = Array.from({ length: nSamples }, (_, i) => {
    const x = Math.sin((i + seed) * 7.3) * 5 + Math.cos((i + seed) * 3.1) * 3
    return x
  })
  const mean = estimates.reduce((a, b) => a + b, 0) / nSamples
  const variance = estimates.reduce((a, b) => a + (b - mean) ** 2, 0) / nSamples

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Gradient Estimate Variance</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Trajectories: {nSamples}
          <input type="range" min={1} max={50} step={1} value={nSamples} onChange={e => setNSamples(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <button onClick={() => setSeed(seed + 1)} className="px-3 py-1 rounded bg-violet-500 text-white text-sm hover:bg-violet-600">Resample</button>
        <span className="text-sm text-violet-600 dark:text-violet-400">Var: {variance.toFixed(2)}</span>
      </div>
      <svg width={360} height={80} className="mx-auto block">
        <line x1={20} y1={40} x2={340} y2={40} stroke="#d1d5db" strokeWidth={1} />
        {estimates.map((v, i) => (
          <circle key={i} cx={180 + v * 12} cy={40} r={3} fill="#7c3aed" opacity={0.5} />
        ))}
        <line x1={180 + mean * 12} y1={20} x2={180 + mean * 12} y2={60} stroke="#f97316" strokeWidth={2} />
        <text x={180 + mean * 12} y={15} textAnchor="middle" fill="#f97316" fontSize={10}>mean</text>
      </svg>
    </div>
  )
}

export default function REINFORCE() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        REINFORCE is the simplest policy gradient algorithm: it directly differentiates the expected
        return with respect to policy parameters using the log-derivative trick.
      </p>

      <TheoremBlock title="Policy Gradient Theorem" id="policy-gradient-theorem">
        <BlockMath math="\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]" />
        <p className="mt-2">where <InlineMath math="G_t = \sum_{k=t}^T \gamma^{k-t} r_k" /> is the return from time step <InlineMath math="t" />.</p>
      </TheoremBlock>

      <DefinitionBlock title="REINFORCE Algorithm">
        <p>For each episode:</p>
        <BlockMath math="\theta \leftarrow \theta + \alpha \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t" />
        <p className="mt-2">This is a Monte Carlo method: it requires complete episodes and uses
        the actual return <InlineMath math="G_t" /> rather than a bootstrapped estimate.</p>
      </DefinitionBlock>

      <GradientVarianceViz />

      <ExampleBlock title="Intuition Behind the Gradient">
        <p>The gradient <InlineMath math="\nabla \log \pi(a|s) \cdot G" /> does two things:</p>
        <p>If <InlineMath math="G > 0" />: increase the probability of action <InlineMath math="a" /> (it led to good returns).</p>
        <p>If <InlineMath math="G < 0" />: decrease the probability (it led to bad returns).</p>
        <p>The magnitude of the update scales with how good or bad the outcome was.</p>
      </ExampleBlock>

      <PythonCode
        title="REINFORCE in PyTorch"
        code={`import torch
import torch.nn as nn
from torch.distributions import Categorical

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim), nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

def reinforce_episode(policy, optimizer, env, gamma=0.99):
    log_probs, rewards = [], []
    state = env.reset()
    done = False
    while not done:
        probs = policy(torch.FloatTensor(state))
        dist = Categorical(probs)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        state, reward, done, _ = env.step(action.item())
        rewards.append(reward)

    # Compute discounted returns
    G, returns = 0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Policy gradient update
    loss = sum(-lp * G for lp, G in zip(log_probs, returns))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()`}
      />

      <NoteBlock type="note" title="High Variance Problem">
        <p>
          REINFORCE suffers from high variance because it uses full Monte Carlo returns. Subtracting
          a baseline <InlineMath math="b(s)" /> from <InlineMath math="G_t" /> reduces variance without
          introducing bias. The natural choice <InlineMath math="b(s) = V(s)" /> leads to the
          advantage function, covered in the next section.
        </p>
      </NoteBlock>
    </div>
  )
}

import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ReplayBufferViz() {
  const [bufferSize, setBufferSize] = useState(5)
  const [batchSize, setBatchSize] = useState(3)
  const experiences = Array.from({ length: bufferSize }, (_, i) => ({
    id: i, s: `s${i}`, a: `a${i % 2}`, r: (Math.random() * 2 - 1).toFixed(1), sn: `s${i + 1}`
  }))
  const sampled = new Set()
  while (sampled.size < Math.min(batchSize, bufferSize)) {
    sampled.add(Math.floor(Math.random() * bufferSize))
  }

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Experience Replay Buffer</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Buffer: {bufferSize} <input type="range" min={3} max={8} step={1} value={bufferSize} onChange={e => setBufferSize(parseInt(e.target.value))} className="w-20 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Batch: {batchSize} <input type="range" min={1} max={bufferSize} step={1} value={Math.min(batchSize, bufferSize)} onChange={e => setBatchSize(parseInt(e.target.value))} className="w-20 accent-violet-500" />
        </label>
      </div>
      <div className="flex flex-wrap gap-2 justify-center">
        {experiences.map((exp, i) => (
          <div key={i} className={`px-3 py-2 rounded-lg text-xs font-mono ${sampled.has(i) ? 'bg-violet-100 dark:bg-violet-900/40 border-2 border-violet-500' : 'bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-600'}`}>
            ({exp.s}, {exp.a}, {exp.r}, {exp.sn})
          </div>
        ))}
      </div>
      <p className="text-center text-xs text-gray-500 mt-2">Violet = sampled for training batch</p>
    </div>
  )
}

export default function DQNArchitecture() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Deep Q-Networks (DQN) combined neural network function approximation with Q-learning,
        achieving human-level play on Atari games. Two key innovations made this stable:
        experience replay and target networks.
      </p>

      <DefinitionBlock title="DQN Loss Function">
        <p>The network <InlineMath math="Q_\theta(s,a)" /> is trained to minimize:</p>
        <BlockMath math="\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\left[\left(r + \gamma \max_{a'} Q_{\theta^-}(s',a') - Q_\theta(s,a)\right)^2\right]" />
        <p className="mt-2">
          where <InlineMath math="\theta^-" /> are the <strong>target network</strong> parameters,
          updated periodically: <InlineMath math="\theta^- \leftarrow \theta" /> every <InlineMath math="C" /> steps.
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="Experience Replay">
        <p>Store transitions <InlineMath math="(s, a, r, s')" /> in a buffer <InlineMath math="\mathcal{D}" /> and
        sample random mini-batches for training. This breaks temporal correlations and reuses data efficiently.</p>
      </DefinitionBlock>

      <ReplayBufferViz />

      <TheoremBlock title="Why Target Networks Stabilize Training" id="target-stability">
        <p>Without a target network, the TD target <InlineMath math="r + \gamma \max Q_\theta(s', a')" /> changes
        with every gradient step, creating a moving target. The target network provides a stable
        objective for <InlineMath math="C" /> steps, reducing oscillations and divergence.</p>
      </TheoremBlock>

      <ExampleBlock title="DQN on Atari">
        <p>The original DQN (Mnih et al., 2015) processed 84x84 grayscale frames through 3 conv layers
        and 2 FC layers. With a replay buffer of 1M transitions and target update every 10K steps,
        it surpassed human performance on 29 of 49 Atari games.</p>
      </ExampleBlock>

      <PythonCode
        title="DQN in PyTorch"
        code={`import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(torch.stack, zip(*[(torch.tensor(s), torch.tensor([a]),
            torch.tensor([r]), torch.tensor(sn), torch.tensor([d]))
            for s, a, r, sn, d in batch]))

def train_step(q_net, target_net, buffer, optimizer, gamma=0.99, batch=32):
    s, a, r, s_next, done = buffer.sample(batch)
    q_values = q_net(s).gather(1, a.long())
    with torch.no_grad():
        max_next_q = target_net(s_next).max(1, keepdim=True)[0]
        target = r + gamma * max_next_q * (1 - done)
    loss = nn.functional.mse_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()`}
      />

      <NoteBlock type="note" title="Soft Target Updates">
        <p>
          Instead of hard copies every <InlineMath math="C" /> steps, many implementations use
          Polyak averaging: <InlineMath math="\theta^- \leftarrow \tau \theta + (1-\tau)\theta^-" /> with
          <InlineMath math="\tau \approx 0.005" />, providing smoother target network updates.
        </p>
      </NoteBlock>
    </div>
  )
}

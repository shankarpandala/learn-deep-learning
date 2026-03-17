import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function NASMethodComparison() {
  const [selected, setSelected] = useState('rl')
  const methods = {
    rl: { name: 'RL-based NAS', cost: '~800 GPU-days', approach: 'Train controller RNN to sample architectures, use validation accuracy as reward',
      pros: 'Found NASNet, flexible search space', cons: 'Extremely expensive' },
    evo: { name: 'Evolutionary NAS', cost: '~300 GPU-days', approach: 'Maintain population of architectures, mutate and select the fittest',
      pros: 'Parallelizable, found AmoebaNet', cons: 'Still very costly, many evaluations needed' },
    darts: { name: 'DARTS (Differentiable)', cost: '~1 GPU-day', approach: 'Relax discrete choices to continuous, optimize with gradient descent',
      pros: '1000x cheaper than RL', cons: 'Can collapse to degenerate solutions' },
    ofa: { name: 'Once-for-All', cost: 'Train once', approach: 'Train a supernetwork supporting many sub-networks, sample without retraining',
      pros: 'Deploy to any hardware target', cons: 'Complex training procedure' },
  }
  const m = methods[selected]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">NAS Method Comparison</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.entries(methods).map(([key, val]) => (
          <button key={key} onClick={() => setSelected(key)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${selected === key ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="space-y-2 p-4 rounded-lg bg-violet-50 dark:bg-violet-900/20">
        <div className="flex justify-between">
          <span className="text-sm font-bold text-violet-800 dark:text-violet-200">{m.name}</span>
          <span className="text-xs text-gray-500">Cost: {m.cost}</span>
        </div>
        <p className="text-sm text-gray-700 dark:text-gray-300">{m.approach}</p>
        <p className="text-xs text-green-700 dark:text-green-400">+ {m.pros}</p>
        <p className="text-xs text-red-600 dark:text-red-400">- {m.cons}</p>
      </div>
    </div>
  )
}

export default function NAS() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Neural Architecture Search (NAS) automates the design of neural network architectures.
        Instead of manual design, NAS methods search over a defined space of possible architectures
        to find optimal structures for a given task and computational budget.
      </p>

      <DefinitionBlock title="NAS Problem Formulation">
        <BlockMath math="\alpha^* = \arg\max_{\alpha \in \mathcal{A}} \; \text{Acc}_{\text{val}}(w^*(\alpha), \alpha)" />
        <BlockMath math="\text{s.t.} \quad w^*(\alpha) = \arg\min_w \; \mathcal{L}_{\text{train}}(w, \alpha)" />
        <p className="mt-2">
          A bilevel optimization: the outer loop searches over architectures <InlineMath math="\alpha" />,
          and the inner loop trains weights <InlineMath math="w" /> for each candidate.
        </p>
      </DefinitionBlock>

      <TheoremBlock title="DARTS: Continuous Relaxation" id="darts">
        <p>DARTS relaxes the discrete choice over <InlineMath math="|\mathcal{O}|" /> operations to a softmax:</p>
        <BlockMath math="\bar{o}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o)}{\sum_{o'} \exp(\alpha_{o'})} \cdot o(x)" />
        <p className="mt-2">
          Architecture parameters <InlineMath math="\alpha" /> and weights <InlineMath math="w" /> are
          optimized alternately with gradient descent. Final architecture uses the argmax operation at each edge.
        </p>
      </TheoremBlock>

      <NASMethodComparison />

      <ExampleBlock title="NAS-Discovered Architectures">
        <p>
          NASNet (2018): Found cells via RL, transferred to ImageNet. EfficientNet-B0 (2019):
          found via MnasNet RL search with FLOPs constraint. These architectures achieved
          state-of-the-art accuracy/efficiency tradeoffs at time of publication.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Simplified DARTS Search Space"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedOp(nn.Module):
    """DARTS mixed operation with continuous relaxation."""
    def __init__(self, channels):
        super().__init__()
        self.ops = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                          nn.BatchNorm2d(channels), nn.ReLU()),
            nn.Sequential(nn.Conv2d(channels, channels, 5, padding=2, bias=False),
                          nn.BatchNorm2d(channels), nn.ReLU()),
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Identity(),  # skip connection
        ])

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.ops))

# Architecture parameters (learnable)
n_ops = 4
arch_params = nn.Parameter(torch.randn(n_ops))
mixed = MixedOp(channels=32)

x = torch.randn(1, 32, 16, 16)
weights = F.softmax(arch_params, dim=0)
out = mixed(x, weights)
print(f"Output: {out.shape}")  # [1, 32, 16, 16]
print(f"Op weights: {weights.data.numpy().round(3)}")
print(f"Selected op: {['conv3x3', 'conv5x5', 'maxpool', 'skip'][weights.argmax()]}")`}
      />

      <NoteBlock type="note" title="Hardware-Aware NAS">
        <p>
          Modern NAS methods incorporate hardware constraints (latency, memory, energy) directly
          into the search objective. MnasNet optimizes <InlineMath math="\text{Acc} \times (\text{LAT}/T)^w" />,
          where <InlineMath math="T" /> is the target latency, producing architectures
          optimized for specific deployment targets like mobile phones.
        </p>
      </NoteBlock>
    </div>
  )
}

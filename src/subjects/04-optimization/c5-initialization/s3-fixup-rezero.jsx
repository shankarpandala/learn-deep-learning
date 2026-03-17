import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function ResidualScaleViz() {
  const [depth, setDepth] = useState(20)
  const [method, setMethod] = useState('rezero')
  const W = 380, H = 180

  const variances = []
  let v = 1.0
  for (let l = 0; l <= depth; l++) {
    variances.push(v)
    if (method === 'naive') {
      v += 1.0
    } else if (method === 'fixup') {
      v += Math.pow(depth, -0.5)
    } else {
      v += 0.0
    }
  }

  const maxV = Math.max(...variances) * 1.1
  const sx = W / (depth + 2), sy = (H - 30) / maxV
  const path = variances.map((val, i) => `${i === 0 ? 'M' : 'L'}${(i + 0.5) * sx},${H - 20 - val * sy}`).join(' ')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Signal Variance Through Residual Blocks</h3>
      <div className="flex items-center gap-4 mb-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Depth: {depth}
          <input type="range" min={5} max={100} step={5} value={depth} onChange={e => setDepth(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <div className="flex gap-2">
          {['naive', 'fixup', 'rezero'].map(m => (
            <button key={m} onClick={() => setMethod(m)}
              className={`px-3 py-1 rounded text-xs font-medium ${method === m ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'}`}>
              {m}
            </button>
          ))}
        </div>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={0} y1={H - 20} x2={W} y2={H - 20} stroke="#d1d5db" strokeWidth={0.5} />
        <path d={path} fill="none" stroke="#8b5cf6" strokeWidth={2} />
        <line x1={0} y1={H - 20 - 1.0 * sy} x2={W} y2={H - 20 - 1.0 * sy} stroke="#f97316" strokeWidth={0.8} strokeDasharray="3,3" />
      </svg>
      <div className="mt-1 text-center text-xs text-gray-500">
        Orange = input variance | {method === 'naive' ? 'Variance grows linearly with depth!' : method === 'fixup' ? 'Growth controlled by L^(-0.5) scaling' : 'α=0 at init: no growth'}
      </div>
    </div>
  )
}

export default function FixupReZero() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Fixup and ReZero enable training very deep residual networks without normalization layers.
        They address the signal explosion problem in deep ResNets through careful initialization
        of residual branches.
      </p>

      <DefinitionBlock title="The Residual Variance Problem">
        <p>In a standard ResNet with <InlineMath math="L" /> blocks:</p>
        <BlockMath math="x_{l+1} = x_l + F_l(x_l)" />
        <p>If each branch adds unit-variance signal, the output variance grows as:</p>
        <BlockMath math="\text{Var}(x_L) = \text{Var}(x_0) + L \cdot \text{Var}(F)" />
        <p className="mt-2">For <InlineMath math="L = 100" />, the signal is 100x larger than the input.</p>
      </DefinitionBlock>

      <ResidualScaleViz />

      <DefinitionBlock title="Fixup Initialization">
        <BlockMath math="W_l^{(1)} \sim \mathcal{N}(0, \text{He variance}) \cdot L^{-1/(2m)}" />
        <BlockMath math="W_l^{(m)} = 0 \quad \text{(last layer in each residual branch)}" />
        <p className="mt-2">
          Scale down early layers in each block by <InlineMath math="L^{-1/(2m)}" /> where
          <InlineMath math="m" /> is the number of layers per block. Zero-initialize the last
          layer so each block starts as an identity function.
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="ReZero">
        <BlockMath math="x_{l+1} = x_l + \alpha_l \cdot F_l(x_l), \quad \alpha_l = 0 \text{ at init}" />
        <p className="mt-2">
          Simply multiply each residual branch by a learnable scalar <InlineMath math="\alpha_l" /> initialized
          to zero. The network starts as the identity and gradually learns to incorporate
          residual contributions.
        </p>
      </DefinitionBlock>

      <TheoremBlock title="Training Signal Preservation" id="fixup-theory">
        <p>
          With Fixup scaling, the output variance satisfies:
        </p>
        <BlockMath math="\text{Var}(x_L) \leq \text{Var}(x_0) + O(\sqrt{L})" />
        <p>
          With ReZero at initialization: <InlineMath math="\text{Var}(x_L) = \text{Var}(x_0)" /> exactly,
          since all <InlineMath math="\alpha_l = 0" />.
        </p>
      </TheoremBlock>

      <ExampleBlock title="1000-Layer ResNets">
        <p>
          ReZero successfully trains ResNets with 1000+ layers without any normalization,
          converging faster than BatchNorm-equipped counterparts in the early stages. The
          learned <InlineMath math="\alpha_l" /> values reveal which residual blocks the network
          considers most important.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Fixup & ReZero Residual Blocks"
        code={`import torch
import torch.nn as nn

class ReZeroBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))  # init to 0!
        self.fn = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.alpha * self.fn(x)

class FixupBlock(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.ones(1))
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)
        # Fixup scaling
        nn.init.kaiming_normal_(self.linear1.weight)
        self.linear1.weight.data *= num_layers ** (-0.25)
        nn.init.zeros_(self.linear2.weight)  # zero last layer

    def forward(self, x):
        out = torch.relu(self.linear1(x) + self.bias1)
        out = self.linear2(out) * self.scale + self.bias2
        return x + out

# Build deep ReZero network
depth = 100
model = nn.Sequential(
    nn.Linear(128, 256),
    *[ReZeroBlock(256) for _ in range(depth)],
    nn.Linear(256, 10)
)
x = torch.randn(8, 128)
out = model(x)
print(f"Output shape: {out.shape}")
print(f"Output std: {out.std().item():.4f}")
alphas = [m.alpha.item() for m in model if isinstance(m, ReZeroBlock)]
print(f"All alphas zero at init: {all(a == 0 for a in alphas)}")`}
      />

      <WarningBlock title="ReZero and Generalization">
        <p>
          While ReZero speeds up early training convergence, some studies show it may slightly
          underperform BatchNorm-equipped networks in final accuracy. Consider combining ReZero
          with normalization for the best of both worlds.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Modern Impact">
        <p>
          The zero-init residual idea from ReZero appears in many modern architectures. GPT-2
          uses a <InlineMath math="1/\sqrt{N}" /> scaling on residual paths, and many Transformer
          implementations zero-initialize the output projection of attention layers. These are
          spiritual successors to Fixup and ReZero.
        </p>
      </NoteBlock>
    </div>
  )
}

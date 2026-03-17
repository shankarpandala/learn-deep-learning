import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function DropComparisonViz() {
  const [mode, setMode] = useState('dropout')
  const [seed, setSeed] = useState(0)
  const W = 360, H = 180

  const srcN = 4, dstN = 4
  const srcY = Array.from({ length: srcN }, (_, i) => 30 + i * 35)
  const dstY = Array.from({ length: dstN }, (_, i) => 30 + i * 35)
  const srcX = 80, dstX = 280

  const rng = (i, j) => Math.sin(seed * 997 + i * 131 + j * 67) * 0.5 + 0.5
  const nodeDropped = (i) => rng(i, 999) < 0.5
  const edgeDropped = (i, j) => rng(i, j) < 0.5

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Dropout vs DropConnect</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          <select value={mode} onChange={e => setMode(e.target.value)} className="rounded border px-2 py-1 text-sm dark:bg-gray-800 dark:border-gray-600">
            <option value="dropout">Dropout (nodes)</option>
            <option value="dropconnect">DropConnect (edges)</option>
          </select>
        </label>
        <button onClick={() => setSeed(s => s + 1)} className="rounded bg-violet-500 px-3 py-1 text-xs text-white hover:bg-violet-600">Resample</button>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        {srcY.map((sy, i) => dstY.map((dy, j) => {
          const hidden = mode === 'dropout' ? nodeDropped(i) : edgeDropped(i, j)
          return <line key={`e${i}${j}`} x1={srcX} y1={sy} x2={dstX} y2={dy}
            stroke={hidden ? '#e5e7eb' : '#8b5cf6'} strokeWidth={hidden ? 0.5 : 1.5} opacity={hidden ? 0.3 : 0.8} />
        }))}
        {srcY.map((y, i) => {
          const dropped = mode === 'dropout' && nodeDropped(i)
          return <circle key={`s${i}`} cx={srcX} cy={y} r={10} fill={dropped ? '#e5e7eb' : '#8b5cf6'} stroke={dropped ? '#9ca3af' : '#7c3aed'} strokeWidth={1.5} opacity={dropped ? 0.4 : 1} />
        })}
        {dstY.map((y, i) => (
          <circle key={`d${i}`} cx={dstX} cy={y} r={10} fill="#8b5cf6" stroke="#7c3aed" strokeWidth={1.5} />
        ))}
        <text x={srcX} y={H - 5} textAnchor="middle" fontSize={10} fill="#6b7280">Source</text>
        <text x={dstX} y={H - 5} textAnchor="middle" fontSize={10} fill="#6b7280">Dest</text>
      </svg>
    </div>
  )
}

export default function DropConnect() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        DropConnect and DropPath extend the dropout idea by operating at the connection
        or path level rather than the neuron level, offering finer-grained regularization.
      </p>

      <DefinitionBlock title="DropConnect">
        <p>Instead of zeroing activations, DropConnect zeros individual weights:</p>
        <BlockMath math="\tilde{W}_{ij} = \begin{cases} 0 & \text{with probability } p \\ W_{ij} & \text{with probability } 1-p \end{cases}" />
        <p className="mt-2">Each connection is independently dropped, giving <InlineMath math="2^{n \times m}" /> possible subnetworks for an <InlineMath math="n \times m" /> weight matrix.</p>
      </DefinitionBlock>

      <DropComparisonViz />

      <DefinitionBlock title="DropPath (Stochastic Depth)">
        <p>In residual networks, entire layers (paths) are randomly skipped during training:</p>
        <BlockMath math="x_{l+1} = x_l + b_l \cdot f_l(x_l), \quad b_l \sim \text{Bernoulli}(1 - p_l)" />
        <p className="mt-2">
          Typically <InlineMath math="p_l" /> increases linearly with depth: earlier layers are
          dropped less often since they learn fundamental features.
        </p>
      </DefinitionBlock>

      <TheoremBlock title="Linear Survival Schedule" id="linear-survival">
        <p>For a network with <InlineMath math="L" /> residual blocks, the survival probability of layer <InlineMath math="l" /> is:</p>
        <BlockMath math="p_{\text{survive}}(l) = 1 - \frac{l}{L}(1 - p_L)" />
        <p className="mt-2">where <InlineMath math="p_L" /> is the survival probability of the last layer (typically 0.8).</p>
      </TheoremBlock>

      <ExampleBlock title="Stochastic Depth in Practice">
        <p>
          Stochastic depth is essential in modern architectures like Vision Transformers (ViT)
          and ConvNeXt. For ViT-Large with 24 blocks, a typical drop path rate of 0.1-0.3
          significantly improves generalization while also reducing training time by ~25%.
        </p>
      </ExampleBlock>

      <PythonCode
        title="DropPath in PyTorch"
        code={`import torch
import torch.nn as nn

class DropPath(nn.Module):
    """Stochastic Depth: drop entire residual branches."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        # Shape: (batch, 1, 1, ...) for broadcasting
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
        return x * mask / keep_prob

class ResidualBlock(nn.Module):
    def __init__(self, dim, drop_path_rate=0.1):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        return x + self.drop_path(self.fc(x))

# Example: 12-block network with linear drop path schedule
depth = 12
dpr = [0.1 * i / (depth - 1) for i in range(depth)]
blocks = nn.Sequential(*[ResidualBlock(128, dp) for dp in dpr])
print(f"Drop rates: {[f'{r:.3f}' for r in dpr]}")`}
      />

      <NoteBlock type="note" title="Choosing Between Variants">
        <p>
          <strong>Dropout</strong>: general purpose, works well in MLPs and attention layers.
          <strong>DropConnect</strong>: finer-grained but more expensive. <strong>DropPath</strong>:
          specifically designed for residual networks and now standard in transformers.
        </p>
      </NoteBlock>
    </div>
  )
}

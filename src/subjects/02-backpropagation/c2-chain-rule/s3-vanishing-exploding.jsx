import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function GradientNormViz() {
  const [spectralNorm, setSpectralNorm] = useState(0.8)
  const layers = 20
  const W = 400, H = 180, padL = 40, padB = 30

  const data = Array.from({ length: layers }, (_, i) => Math.pow(spectralNorm, i + 1))
  const maxVal = Math.max(...data, 1)
  const xStep = (W - padL - 10) / layers
  const yScale = (H - padB - 10) / maxVal

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Gradient Norm vs Depth</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Spectral norm: {spectralNorm.toFixed(2)}
        <input type="range" min={0.3} max={1.5} step={0.01} value={spectralNorm} onChange={e => setSpectralNorm(parseFloat(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={padL} y1={0} x2={padL} y2={H - padB} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={padL} y1={H - padB} x2={W} y2={H - padB} stroke="#d1d5db" strokeWidth={0.5} />
        {data.map((v, i) => {
          const barH = Math.min(v * yScale, H - padB - 5)
          const color = spectralNorm < 1 ? '#8b5cf6' : spectralNorm > 1 ? '#ef4444' : '#22c55e'
          return (
            <rect key={i} x={padL + i * xStep + 2} y={H - padB - barH} width={Math.max(xStep - 4, 2)} height={barH} rx={2} fill={color} opacity={0.8} />
          )
        })}
        <text x={W / 2} y={H - 5} textAnchor="middle" fontSize={11} fill="#6b7280">Layer depth</text>
        <text x={12} y={H / 2} textAnchor="middle" fontSize={11} fill="#6b7280" transform={`rotate(-90, 12, ${H / 2})`}>Grad norm</text>
      </svg>
      <p className="text-center text-sm mt-2" style={{ color: spectralNorm < 1 ? '#7c3aed' : spectralNorm > 1 ? '#dc2626' : '#16a34a' }}>
        {spectralNorm < 1 ? 'Vanishing gradients' : spectralNorm > 1 ? 'Exploding gradients' : 'Stable gradients'}
      </p>
    </div>
  )
}

export default function VanishingExploding() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Deep networks suffer from vanishing or exploding gradients when the product of Jacobians
        across layers either shrinks or grows exponentially. This is the central challenge that
        modern architectures must address.
      </p>

      <TheoremBlock title="Gradient Scaling with Depth" id="gradient-scaling">
        <p>
          If each layer's Jacobian has spectral norm <InlineMath math="\sigma" />, then after{' '}
          <InlineMath math="L" /> layers:
        </p>
        <BlockMath math="\left\|\frac{\partial \mathcal{L}}{\partial x}\right\| \approx \sigma^L \left\|\frac{\partial \mathcal{L}}{\partial h_L}\right\|" />
        <p>
          For <InlineMath math="\sigma < 1" />: vanishing. For <InlineMath math="\sigma > 1" />: exploding.
        </p>
      </TheoremBlock>

      <GradientNormViz />

      <WarningBlock title="Sigmoid Activations in Deep Networks">
        <p>
          Sigmoid's maximum derivative is 0.25, guaranteeing vanishing gradients. A 20-layer sigmoid
          network has gradient factor <InlineMath math="0.25^{20} \approx 10^{-12}" />. This is why
          sigmoid is never used in hidden layers of deep networks.
        </p>
      </WarningBlock>

      <DefinitionBlock title="Gradient Clipping">
        <p>
          Gradient clipping limits the gradient norm to prevent explosion:
        </p>
        <BlockMath math="g \leftarrow g \cdot \min\left(1, \frac{\theta}{\|g\|}\right)" />
        <p>
          where <InlineMath math="\theta" /> is the maximum allowed norm. This rescales the gradient
          direction without changing it.
        </p>
      </DefinitionBlock>

      <ExampleBlock title="Residual Connections Preserve Gradients">
        <p>
          A residual block computes <InlineMath math="h_{l+1} = h_l + f(h_l)" />. Its Jacobian is:
        </p>
        <BlockMath math="\frac{\partial h_{l+1}}{\partial h_l} = I + \frac{\partial f}{\partial h_l}" />
        <p>
          The identity term <InlineMath math="I" /> ensures gradients always have a direct path,
          preventing vanishing. This is why ResNets can train hundreds of layers.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Detecting and Fixing Gradient Problems"
        code={`import torch
import torch.nn as nn

# Deep network with sigmoid — vanishing gradients
deep_sig = nn.Sequential(*[nn.Sequential(nn.Linear(32, 32), nn.Sigmoid()) for _ in range(10)])
x = torch.randn(1, 32)
y = deep_sig(x).sum()
y.backward()
print("Sigmoid grad norm layer 0:", deep_sig[0][0].weight.grad.norm().item())

# Same depth with ReLU + residual — stable gradients
class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)
    def forward(self, x):
        return x + torch.relu(self.fc(x))  # residual!

deep_res = nn.Sequential(*[ResBlock(32) for _ in range(10)])
y2 = deep_res(x).sum()
y2.backward()
print("ResBlock grad norm layer 0:", deep_res[0].fc.weight.grad.norm().item())

# Gradient clipping example
optimizer = torch.optim.SGD(deep_res.parameters(), lr=0.01)
nn.utils.clip_grad_norm_(deep_res.parameters(), max_norm=1.0)`}
      />

      <NoteBlock type="note" title="Modern Solutions Summary">
        <p>
          Key techniques to combat gradient pathologies: <strong>ReLU activations</strong> (avoid
          saturation), <strong>residual connections</strong> (direct gradient paths),{' '}
          <strong>batch/layer normalization</strong> (control activation scales),{' '}
          <strong>careful initialization</strong> (Xavier/He), and{' '}
          <strong>gradient clipping</strong> (prevent explosion in RNNs).
        </p>
      </NoteBlock>
    </div>
  )
}

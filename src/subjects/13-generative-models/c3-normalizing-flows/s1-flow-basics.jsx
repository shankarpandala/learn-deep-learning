import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ChangeOfVariablesViz() {
  const [scale, setScale] = useState(1.5)
  const [shift, setShift] = useState(0.5)
  const logDetJ = Math.log(Math.abs(scale)).toFixed(3)
  const W = 380, H = 120

  const gaussPoints = Array.from({ length: 80 }, (_, i) => {
    const x = -3 + i * 0.075
    const y = Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI)
    return { x, y }
  })

  const toSVG = (x, y, offsetX) => `${offsetX + x * 25 + 90},${H - 15 - y * 200}`

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Change of Variables</h3>
      <div className="flex gap-4 mb-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          scale: {scale.toFixed(1)}
          <input type="range" min={0.3} max={3} step={0.1} value={scale} onChange={e => setScale(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          shift: {shift.toFixed(1)}
          <input type="range" min={-2} max={2} step={0.1} value={shift} onChange={e => setShift(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <span className="text-xs text-violet-600">log|det J| = {logDetJ}</span>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <path d={gaussPoints.map((p, i) => `${i === 0 ? 'M' : 'L'}${toSVG(p.x, p.y, 0)}`).join(' ')} fill="none" stroke="#8b5cf6" strokeWidth={2} />
        <path d={gaussPoints.map((p, i) => {
          const tx = p.x * scale + shift
          const ty = p.y / Math.abs(scale)
          return `${i === 0 ? 'M' : 'L'}${toSVG(tx, ty, 0)}`
        }).join(' ')} fill="none" stroke="#f97316" strokeWidth={2} />
        <text x={50} y={12} className="text-[10px] fill-violet-500">z ~ N(0,1)</text>
        <text x={250} y={12} className="text-[10px] fill-orange-500">x = scale*z + shift</text>
      </svg>
    </div>
  )
}

export default function FlowBasics() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Normalizing flows transform a simple base distribution (e.g., Gaussian) into a complex
        data distribution through a chain of invertible transformations, allowing exact likelihood computation.
      </p>

      <TheoremBlock title="Change of Variables Formula" id="change-of-variables">
        <p>If <InlineMath math="f: \mathbb{R}^d \to \mathbb{R}^d" /> is an invertible, differentiable map and <InlineMath math="\mathbf{x} = f(\mathbf{z})" />:</p>
        <BlockMath math="\log p_X(\mathbf{x}) = \log p_Z(f^{-1}(\mathbf{x})) - \log\left|\det \frac{\partial f}{\partial \mathbf{z}}\right|" />
        <p className="mt-2">
          The Jacobian determinant accounts for volume change under the transformation.
        </p>
      </TheoremBlock>

      <DefinitionBlock title="Normalizing Flow">
        <p>A normalizing flow composes <InlineMath math="K" /> invertible transformations:</p>
        <BlockMath math="\mathbf{x} = f_K \circ f_{K-1} \circ \cdots \circ f_1(\mathbf{z}_0), \quad \mathbf{z}_0 \sim p_0(\mathbf{z})" />
        <p className="mt-2">Log-likelihood decomposes as:</p>
        <BlockMath math="\log p(\mathbf{x}) = \log p_0(\mathbf{z}_0) - \sum_{k=1}^{K} \log\left|\det J_{f_k}\right|" />
      </DefinitionBlock>

      <ChangeOfVariablesViz />

      <ExampleBlock title="Planar Flow">
        <p>A simple flow layer with a single hyperplane:</p>
        <BlockMath math="f(\mathbf{z}) = \mathbf{z} + \mathbf{u} \cdot h(\mathbf{w}^\top \mathbf{z} + b)" />
        <p className="mt-2">
          The Jacobian determinant is <InlineMath math="1 + \mathbf{u}^\top h'(\mathbf{w}^\top \mathbf{z} + b) \mathbf{w}" />,
          computable in <InlineMath math="O(d)" /> time.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Simple Planar Flow in PyTorch"
        code={`import torch
import torch.nn as nn

class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim))
        self.u = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        linear = z @ self.w + self.b          # (B,)
        h = torch.tanh(linear)                # (B,)
        f_z = z + self.u.unsqueeze(0) * h.unsqueeze(1)  # (B, D)
        # Log-det Jacobian
        h_prime = 1 - h ** 2                  # (B,)
        log_det = torch.log(torch.abs(1 + h_prime * (self.u @ self.w)) + 1e-8)
        return f_z, log_det

# Stack multiple planar flows
z = torch.randn(64, 2)  # 2D base distribution
log_prob_z = -0.5 * z.pow(2).sum(-1)  # log N(0,I)
total_log_det = 0
for _ in range(8):
    flow = PlanarFlow(dim=2)
    z, log_det = flow(z)
    total_log_det += log_det
log_prob_x = log_prob_z - total_log_det
print(f"Output shape: {z.shape}, mean log p(x): {log_prob_x.mean():.3f}")`}
      />

      <NoteBlock type="note" title="Key Trade-off: Expressiveness vs Efficiency">
        <p>
          The Jacobian determinant for a general <InlineMath math="d \times d" /> matrix
          costs <InlineMath math="O(d^3)" />. Practical flows use architectures with triangular
          Jacobians (coupling layers, autoregressive flows) for <InlineMath math="O(d)" /> computation.
        </p>
      </NoteBlock>
    </div>
  )
}

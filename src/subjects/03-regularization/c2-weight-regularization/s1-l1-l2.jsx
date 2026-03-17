import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function RegularizationGeometry() {
  const [lambda, setLambda] = useState(1.0)
  const [regType, setRegType] = useState('L2')
  const W = 300, H = 300, cx = W / 2, cy = H / 2, scale = 60

  const constraintPoints = (type, r) => {
    const pts = []
    for (let a = 0; a <= 2 * Math.PI; a += 0.02) {
      if (type === 'L2') {
        pts.push({ x: cx + r * Math.cos(a), y: cy - r * Math.sin(a) })
      } else {
        const t = a
        const x = r * Math.sign(Math.cos(t)) * Math.abs(Math.cos(t))
        const y = r * Math.sign(Math.sin(t)) * Math.abs(Math.sin(t))
        pts.push({ x: cx + (Math.cos(t)) * r, y: cy - (Math.sin(t)) * r })
      }
    }
    return pts
  }

  const l1Points = Array.from({ length: 201 }, (_, i) => {
    const t = (i / 200) * 2 * Math.PI
    const r = lambda * scale
    const abscos = Math.abs(Math.cos(t)), abssin = Math.abs(Math.sin(t))
    const s = r / (abscos + abssin || 1)
    return { x: cx + Math.cos(t) * s, y: cy - Math.sin(t) * s }
  })

  const l2Points = Array.from({ length: 201 }, (_, i) => {
    const t = (i / 200) * 2 * Math.PI
    const r = lambda * scale
    return { x: cx + Math.cos(t) * r, y: cy - Math.sin(t) * r }
  })

  const pts = regType === 'L1' ? l1Points : l2Points
  const pathD = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ') + 'Z'

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Constraint Region Geometry</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          <select value={regType} onChange={e => setRegType(e.target.value)} className="rounded border px-2 py-1 text-sm dark:bg-gray-800 dark:border-gray-600">
            <option value="L1">L1 (Lasso)</option>
            <option value="L2">L2 (Ridge)</option>
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          <InlineMath math="\lambda" /> = {lambda.toFixed(1)}
          <input type="range" min={0.2} max={2.0} step={0.1} value={lambda} onChange={e => setLambda(parseFloat(e.target.value))} className="w-32 accent-violet-500" />
        </label>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={0} y1={cy} x2={W} y2={cy} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={cx} y1={0} x2={cx} y2={H} stroke="#d1d5db" strokeWidth={0.5} />
        <path d={pathD} fill="rgba(139,92,246,0.1)" stroke="#8b5cf6" strokeWidth={2} />
        <ellipse cx={cx + 60} cy={cy - 40} rx={80} ry={50} fill="none" stroke="#f97316" strokeWidth={1.5} strokeDasharray="4,4" transform={`rotate(-30 ${cx + 60} ${cy - 40})`} />
      </svg>
      <p className="text-xs text-center text-gray-500 mt-2">Violet: constraint region. Orange: loss contours (ellipses).</p>
    </div>
  )
}

export default function L1L2Regularization() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Weight regularization adds a penalty on model parameters to the loss function,
        discouraging overly complex models and improving generalization.
      </p>

      <DefinitionBlock title="L2 Regularization (Ridge / Weight Decay)">
        <BlockMath math="\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{data}} + \frac{\lambda}{2} \|\mathbf{w}\|_2^2 = \mathcal{L}_{\text{data}} + \frac{\lambda}{2} \sum_i w_i^2" />
        <p className="mt-2">Penalizes large weights, encouraging small distributed values. Bayesian interpretation: Gaussian prior on weights.</p>
      </DefinitionBlock>

      <DefinitionBlock title="L1 Regularization (Lasso)">
        <BlockMath math="\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{data}} + \lambda \|\mathbf{w}\|_1 = \mathcal{L}_{\text{data}} + \lambda \sum_i |w_i|" />
        <p className="mt-2">Encourages sparsity (many weights become exactly zero). Bayesian interpretation: Laplace prior on weights.</p>
      </DefinitionBlock>

      <RegularizationGeometry />

      <TheoremBlock title="Why L1 Produces Sparsity" id="l1-sparsity">
        <p>
          The L1 constraint region is a diamond whose corners lie on the axes. Loss
          contour ellipses are more likely to touch the diamond at a corner where one
          coordinate is zero, producing sparse solutions:
        </p>
        <BlockMath math="\text{argmin}_{\|\mathbf{w}\|_1 \leq t} \mathcal{L}(\mathbf{w}) \text{ is more likely to have } w_i = 0" />
      </TheoremBlock>

      <PythonCode
        title="L1 & L2 Regularization in PyTorch"
        code={`import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
lam_l2, lam_l1 = 1e-4, 1e-5

x, y = torch.randn(200, 20), torch.randn(200, 1)

for epoch in range(100):
    pred = model(x)
    loss = criterion(pred, y)

    # L2 penalty
    l2_reg = sum(p.pow(2).sum() for p in model.parameters())
    # L1 penalty
    l1_reg = sum(p.abs().sum() for p in model.parameters())

    total_loss = loss + lam_l2 * l2_reg + lam_l1 * l1_reg
    optimizer.zero_grad(); total_loss.backward(); optimizer.step()

# Count near-zero weights (sparsity from L1)
n_sparse = sum((p.abs() < 1e-3).sum().item() for p in model.parameters())
n_total = sum(p.numel() for p in model.parameters())
print(f"Near-zero weights: {n_sparse}/{n_total} ({100*n_sparse/n_total:.1f}%)")`}
      />

      <NoteBlock type="note" title="Elastic Net">
        <p>
          Elastic Net combines both: <InlineMath math="\lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|_2^2" />.
          This provides sparsity from L1 while maintaining the grouping effect of L2 for correlated features.
        </p>
      </NoteBlock>
    </div>
  )
}

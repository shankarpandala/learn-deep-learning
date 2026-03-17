import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function GDVisualization() {
  const [lr, setLr] = useState(0.1)
  const [steps, setSteps] = useState(10)
  const f = (x, y) => x * x + 3 * y * y
  const W = 360, H = 280, cx = W / 2, cy = H / 2, scale = 30

  const trajectory = []
  let px = 3.0, py = 2.0
  for (let i = 0; i <= steps; i++) {
    trajectory.push([px, py])
    const gx = 2 * px, gy = 6 * py
    px -= lr * gx
    py -= lr * gy
  }

  const contourLevels = [1, 3, 6, 10, 15, 25]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Gradient Descent on f(x,y) = x^2 + 3y^2</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          LR: {lr.toFixed(3)}
          <input type="range" min={0.01} max={0.32} step={0.005} value={lr} onChange={e => setLr(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Steps: {steps}
          <input type="range" min={1} max={30} step={1} value={steps} onChange={e => setSteps(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        {contourLevels.map((lev, i) => {
          const rx = Math.sqrt(lev) * scale
          const ry = Math.sqrt(lev / 3) * scale
          return <ellipse key={i} cx={cx} cy={cy} rx={rx} ry={ry} fill="none" stroke="#c4b5fd" strokeWidth={0.8} opacity={0.6} />
        })}
        {trajectory.map((p, i) => {
          if (i === 0) return null
          const [x0, y0] = trajectory[i - 1]
          const [x1, y1] = p
          return <line key={i} x1={cx + x0 * scale} y1={cy - y0 * scale} x2={cx + x1 * scale} y2={cy - y1 * scale} stroke="#7c3aed" strokeWidth={1.5} />
        })}
        {trajectory.map(([x, y], i) => (
          <circle key={i} cx={cx + x * scale} cy={cy - y * scale} r={i === 0 ? 5 : 3} fill={i === 0 ? '#ef4444' : '#8b5cf6'} />
        ))}
        <circle cx={cx} cy={cy} r={3} fill="#22c55e" />
      </svg>
      <p className="text-center text-sm text-gray-500 mt-1">
        Final f = {f(trajectory[trajectory.length - 1][0], trajectory[trajectory.length - 1][1]).toFixed(4)}
      </p>
    </div>
  )
}

export default function GradientDescent() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Gradient descent iteratively updates parameters in the direction of steepest decrease of the loss.
        The choice of batch size and learning rate critically affects convergence speed and solution quality.
      </p>

      <DefinitionBlock title="Gradient Descent Update Rule">
        <BlockMath math="\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)" />
        <p>
          where <InlineMath math="\eta" /> is the learning rate. Three variants differ by how
          the gradient is estimated:
        </p>
        <p><strong>Batch GD:</strong> Full dataset — exact gradient, slow per step.</p>
        <p><strong>SGD:</strong> Single sample — noisy gradient, fast per step.</p>
        <p><strong>Mini-batch:</strong> Subset of <InlineMath math="B" /> samples — best trade-off.</p>
      </DefinitionBlock>

      <TheoremBlock title="SGD Convergence" id="sgd-convergence">
        <p>
          For convex <InlineMath math="\mathcal{L}" /> with Lipschitz gradients (<InlineMath math="L" />-smooth),
          SGD with learning rate <InlineMath math="\eta = O(1/\sqrt{T})" /> converges at rate:
        </p>
        <BlockMath math="\mathbb{E}[\mathcal{L}(\bar{\theta}_T)] - \mathcal{L}^* = O\left(\frac{1}{\sqrt{T}}\right)" />
        <p>Mini-batch reduces variance by factor <InlineMath math="1/B" /> but does not change the rate.</p>
      </TheoremBlock>

      <GDVisualization />

      <ExampleBlock title="Learning Rate Effects">
        <p>
          <strong>Too small:</strong> Slow convergence, can get trapped in poor local minima.{' '}
          <strong>Too large:</strong> Oscillation, divergence.{' '}
          <strong>Just right:</strong> Fast convergence to a good minimum. The optimal LR depends on
          the curvature (Hessian eigenvalues) of the loss surface.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Batch vs Mini-Batch vs SGD in PyTorch"
        code={`import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Synthetic data
X = torch.randn(1000, 10)
y = X @ torch.randn(10, 1) + 0.1 * torch.randn(1000, 1)

model = nn.Linear(10, 1)
criterion = nn.MSELoss()

# Mini-batch SGD (most common)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(5):
    total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: loss = {total_loss / len(loader):.4f}")`}
      />

      <NoteBlock type="note" title="Why Mini-Batch Works Best">
        <p>
          Mini-batch SGD provides a favorable trade-off: GPU parallelism makes batches nearly free
          up to a hardware-dependent size, and gradient noise acts as implicit regularization,
          helping escape sharp minima. Typical batch sizes range from 32 to 4096 depending on the
          task and available memory.
        </p>
      </NoteBlock>
    </div>
  )
}

import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function MomentumViz() {
  const [beta, setBeta] = useState(0.9)
  const [step, setStep] = useState(0)
  const W = 400, H = 250

  const trajectory = []
  let x = 3.5, y = 3.0, vx = 0, vy = 0
  const lr = 0.02
  for (let i = 0; i <= 60; i++) {
    trajectory.push({ x, y })
    const gx = 2 * x
    const gy = 20 * y
    vx = beta * vx + gx
    vy = beta * vy + gy
    x -= lr * vx
    y -= lr * vy
  }

  const vanillaTrajectory = []
  let vx2 = 3.5, vy2 = 3.0
  for (let i = 0; i <= 60; i++) {
    vanillaTrajectory.push({ x: vx2, y: vy2 })
    vx2 -= lr * 2 * vx2
    vy2 -= lr * 20 * vy2
  }

  const sx = W / 8, sy = H / 7, ox = W / 2, oy = H / 2
  const toSVG = (px, py) => `${ox + px * sx},${oy - py * sy}`
  const idx = Math.min(step, trajectory.length - 1)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Momentum vs Vanilla SGD</h3>
      <div className="flex items-center gap-4 mb-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          β = {beta.toFixed(2)}
          <input type="range" min={0} max={0.99} step={0.01} value={beta} onChange={e => setBeta(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Step: {step}
          <input type="range" min={0} max={60} step={1} value={step} onChange={e => setStep(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={0} y1={oy} x2={W} y2={oy} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={ox} y1={0} x2={ox} y2={H} stroke="#d1d5db" strokeWidth={0.5} />
        {vanillaTrajectory.slice(0, idx + 1).map((p, i, arr) => i > 0 && (
          <line key={`v-${i}`} x1={ox + arr[i - 1].x * sx} y1={oy - arr[i - 1].y * sy} x2={ox + p.x * sx} y2={oy - p.y * sy} stroke="#9ca3af" strokeWidth={1.5} opacity={0.6} />
        ))}
        {trajectory.slice(0, idx + 1).map((p, i, arr) => i > 0 && (
          <line key={`m-${i}`} x1={ox + arr[i - 1].x * sx} y1={oy - arr[i - 1].y * sy} x2={ox + p.x * sx} y2={oy - p.y * sy} stroke="#8b5cf6" strokeWidth={2} />
        ))}
        <circle cx={ox + trajectory[idx].x * sx} cy={oy - trajectory[idx].y * sy} r={4} fill="#8b5cf6" />
        <circle cx={ox + vanillaTrajectory[idx].x * sx} cy={oy - vanillaTrajectory[idx].y * sy} r={4} fill="#9ca3af" />
      </svg>
      <div className="mt-2 flex justify-center gap-6 text-xs">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-violet-500" /> Momentum</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-gray-400" /> Vanilla SGD</span>
      </div>
    </div>
  )
}

export default function Momentum() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Momentum accelerates SGD by accumulating an exponentially decaying moving average of past
        gradients, allowing the optimizer to build up velocity in consistent gradient directions
        and dampen oscillations.
      </p>

      <DefinitionBlock title="Classical Momentum">
        <BlockMath math="v_t = \beta \, v_{t-1} + \nabla_\theta \mathcal{L}(\theta_{t-1})" />
        <BlockMath math="\theta_t = \theta_{t-1} - \alpha \, v_t" />
        <p className="mt-2">
          Here <InlineMath math="\beta \in [0,1)" /> is the momentum coefficient (typically 0.9)
          and <InlineMath math="v_t" /> is the velocity vector.
        </p>
      </DefinitionBlock>

      <MomentumViz />

      <ExampleBlock title="Exponential Moving Average Intuition">
        <p>
          Expanding the velocity recursion for <InlineMath math="\beta = 0.9" />:
        </p>
        <BlockMath math="v_t = g_t + 0.9\,g_{t-1} + 0.81\,g_{t-2} + 0.729\,g_{t-3} + \cdots" />
        <p>
          The effective window is roughly <InlineMath math="1/(1-\beta) = 10" /> steps. Gradients
          from more than ~10 steps ago contribute negligibly.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Momentum SGD in PyTorch"
        code={`import torch
import torch.optim as optim
import torch.nn as nn

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

# Classical momentum with β=0.9
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training step
x = torch.randn(32, 784)
target = torch.randint(0, 10, (32,))
loss = nn.CrossEntropyLoss()(model(x), target)
loss.backward()
optimizer.step()
optimizer.zero_grad()

print(f"Loss: {loss.item():.4f}")`}
      />

      <WarningBlock title="Momentum Can Overshoot">
        <p>
          High momentum values (<InlineMath math="\beta > 0.95" />) can cause the optimizer to
          overshoot minima, especially with large learning rates. If training becomes unstable,
          try reducing <InlineMath math="\beta" /> or the learning rate.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Why Momentum Works">
        <p>
          In ravine-shaped loss landscapes (common in deep learning), gradients oscillate across the
          narrow dimension and are consistent along the long dimension. Momentum cancels out
          oscillations and amplifies the consistent direction, leading to faster convergence.
        </p>
      </NoteBlock>
    </div>
  )
}

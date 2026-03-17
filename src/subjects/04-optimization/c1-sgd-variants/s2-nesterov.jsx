import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function NesterovViz() {
  const [step, setStep] = useState(0)
  const W = 400, H = 250, ox = W / 2, oy = H / 2, sx = 40, sy = 40
  const beta = 0.9, lr = 0.02

  const classical = []
  let cx = 3, cy = 3, cvx = 0, cvy = 0
  for (let i = 0; i <= 50; i++) {
    classical.push({ x: cx, y: cy })
    const gx = 2 * cx; const gy = 18 * cy
    cvx = beta * cvx + gx; cvy = beta * cvy + gy
    cx -= lr * cvx; cy -= lr * cvy
  }

  const nesterov = []
  let nx = 3, ny = 3, nvx = 0, nvy = 0
  for (let i = 0; i <= 50; i++) {
    nesterov.push({ x: nx, y: ny })
    const lax = nx - lr * beta * nvx
    const lay = ny - lr * beta * nvy
    const gx = 2 * lax; const gy = 18 * lay
    nvx = beta * nvx + gx; nvy = beta * nvy + gy
    nx -= lr * nvx; ny -= lr * nvy
  }

  const idx = Math.min(step, 50)
  const toX = (v) => ox + v * sx
  const toY = (v) => oy - v * sy

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Nesterov vs Classical Momentum</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Step: {step}
        <input type="range" min={0} max={50} step={1} value={step} onChange={e => setStep(parseInt(e.target.value))} className="w-36 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={0} y1={oy} x2={W} y2={oy} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={ox} y1={0} x2={ox} y2={H} stroke="#d1d5db" strokeWidth={0.5} />
        {classical.slice(0, idx + 1).map((p, i, a) => i > 0 && (
          <line key={`c-${i}`} x1={toX(a[i-1].x)} y1={toY(a[i-1].y)} x2={toX(p.x)} y2={toY(p.y)} stroke="#9ca3af" strokeWidth={1.5} />
        ))}
        {nesterov.slice(0, idx + 1).map((p, i, a) => i > 0 && (
          <line key={`n-${i}`} x1={toX(a[i-1].x)} y1={toY(a[i-1].y)} x2={toX(p.x)} y2={toY(p.y)} stroke="#8b5cf6" strokeWidth={2} />
        ))}
        <circle cx={toX(classical[idx].x)} cy={toY(classical[idx].y)} r={4} fill="#9ca3af" />
        <circle cx={toX(nesterov[idx].x)} cy={toY(nesterov[idx].y)} r={4} fill="#8b5cf6" />
        <circle cx={toX(0)} cy={toY(0)} r={5} fill="#f97316" opacity={0.6} />
      </svg>
      <div className="mt-2 flex justify-center gap-6 text-xs">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-violet-500" /> Nesterov</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-gray-400" /> Classical</span>
        <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-full bg-orange-500" /> Minimum</span>
      </div>
    </div>
  )
}

export default function Nesterov() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Nesterov Accelerated Gradient (NAG) improves on classical momentum by evaluating the gradient
        at a &ldquo;look-ahead&rdquo; position, providing a corrective factor that reduces overshooting.
      </p>

      <DefinitionBlock title="Nesterov Accelerated Gradient">
        <BlockMath math="v_t = \beta \, v_{t-1} + \nabla_\theta \mathcal{L}(\theta_{t-1} - \alpha \beta \, v_{t-1})" />
        <BlockMath math="\theta_t = \theta_{t-1} - \alpha \, v_t" />
        <p className="mt-2">
          The key difference: the gradient is computed at the anticipated future
          position <InlineMath math="\theta - \alpha \beta v" /> rather than the current position.
        </p>
      </DefinitionBlock>

      <NesterovViz />

      <TheoremBlock title="Convergence Advantage" id="nag-convergence">
        <p>
          For <InlineMath math="L" />-smooth convex functions, Nesterov momentum achieves:
        </p>
        <BlockMath math="f(\theta_t) - f(\theta^*) \leq O\!\left(\frac{1}{t^2}\right)" />
        <p>
          compared to <InlineMath math="O(1/t)" /> for classical gradient descent, making it
          an optimal first-order method by Nesterov's lower bound.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Look-Ahead Intuition">
        <p>
          Think of a ball rolling downhill with momentum. Classical momentum checks the slope at
          the current position. Nesterov first rolls the ball forward by its momentum, then checks
          the slope at the new position. This look-ahead acts as a correction that prevents
          overshooting valleys.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Nesterov Momentum in PyTorch"
        code={`import torch
import torch.optim as optim
import torch.nn as nn

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

# Nesterov momentum — just add nesterov=True
optimizer = optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9, nesterov=True
)

x = torch.randn(32, 784)
target = torch.randint(0, 10, (32,))
loss = nn.CrossEntropyLoss()(model(x), target)
loss.backward()
optimizer.step()
optimizer.zero_grad()
print(f"Loss: {loss.item():.4f}")`}
      />

      <NoteBlock type="note" title="Practical Recommendations">
        <p>
          In practice, Nesterov momentum gives a modest but consistent improvement over classical
          momentum. It is the default choice for SGD in many frameworks. When using adaptive methods
          like Adam, the Nesterov variant (NAdam) can also be beneficial.
        </p>
      </NoteBlock>
    </div>
  )
}

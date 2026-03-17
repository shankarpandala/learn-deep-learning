import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function LossLandscapeViz() {
  const [surfaceType, setSurfaceType] = useState('saddle')
  const W = 380, H = 220, cx = W / 2, cy = H / 2

  const surfaces = {
    saddle: { label: 'Saddle Point', fn: (x, y) => x * x - y * y },
    localmin: { label: 'Local Minimum', fn: (x, y) => Math.sin(x * 1.5) * Math.cos(y * 1.5) + 0.05 * (x * x + y * y) },
    flat: { label: 'Flat Region', fn: (x, y) => 0.5 * Math.tanh(x * x + y * y - 4) },
  }

  const { fn } = surfaces[surfaceType]
  const gridSize = 20
  const scale = 8, zScale = 25
  const isoY = 0.3

  const points = []
  for (let i = -gridSize; i <= gridSize; i++) {
    const x = i * 0.2
    const z = fn(x, 0)
    points.push([cx + x * scale * 3, cy - z * zScale])
  }
  const linePathH = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0]},${p[1]}`).join(' ')

  const points2 = []
  for (let i = -gridSize; i <= gridSize; i++) {
    const y = i * 0.2
    const z = fn(0, y)
    points2.push([cx + y * scale * 3, cy - z * zScale])
  }
  const linePathV = points2.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0]},${p[1]}`).join(' ')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Loss Surface Cross-Sections</h3>
      <div className="flex gap-3 mb-3">
        {Object.entries(surfaces).map(([key, { label }]) => (
          <button key={key} onClick={() => setSurfaceType(key)}
            className={`px-3 py-1 rounded text-sm ${surfaceType === key ? 'bg-violet-500 text-white' : 'bg-violet-100 text-violet-700 dark:bg-violet-900 dark:text-violet-300'}`}>
            {label}
          </button>
        ))}
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={0} y1={cy} x2={W} y2={cy} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={cx} y1={0} x2={cx} y2={H} stroke="#d1d5db" strokeWidth={0.5} />
        <path d={linePathH} fill="none" stroke="#8b5cf6" strokeWidth={2.5} />
        <path d={linePathV} fill="none" stroke="#f97316" strokeWidth={2} strokeDasharray="5,3" />
        <circle cx={cx} cy={cy - fn(0, 0) * zScale} r={5} fill="#7c3aed" />
      </svg>
      <div className="mt-2 flex justify-center gap-6 text-xs">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-violet-500" /> f(x, 0) slice</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-orange-500" /> f(0, y) slice</span>
      </div>
    </div>
  )
}

export default function LossLandscapes() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The loss landscape of a neural network is the surface defined by the loss function over
        the parameter space. Understanding its geometry — saddle points, local minima, and flat
        regions — is crucial for designing effective optimizers.
      </p>

      <DefinitionBlock title="Critical Points">
        <p>
          A critical point satisfies <InlineMath math="\nabla \mathcal{L} = 0" />. It is classified by
          the Hessian eigenvalues:
        </p>
        <p><strong>Local minimum:</strong> All eigenvalues <InlineMath math="> 0" /> (positive definite)</p>
        <p><strong>Local maximum:</strong> All eigenvalues <InlineMath math="< 0" /> (negative definite)</p>
        <p><strong>Saddle point:</strong> Mixed positive and negative eigenvalues</p>
        <BlockMath math="\text{Index}(\theta^*) = \text{number of negative eigenvalues of } H(\theta^*)" />
      </DefinitionBlock>

      <TheoremBlock title="Saddle Point Dominance" id="saddle-dominance">
        <p>
          In high-dimensional loss surfaces, saddle points vastly outnumber local minima.
          For <InlineMath math="n" /> parameters, a random critical point is a local minimum with
          probability roughly <InlineMath math="2^{-n}" />. Most critical points are saddle points,
          and gradient descent naturally escapes them via noise.
        </p>
      </TheoremBlock>

      <LossLandscapeViz />

      <ExampleBlock title="Mode Connectivity">
        <p>
          Different local minima found by SGD are often connected by paths of nearly constant loss
          (mode connectivity). This means the loss landscape has a connected low-loss manifold
          rather than isolated basins. Techniques like <strong>linear mode connectivity</strong> test
          if the linear interpolation between two solutions stays low-loss.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Visualizing Loss Along a Random Direction"
        code={`import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
X = torch.randn(100, 10)
y = torch.randn(100, 1)
criterion = nn.MSELoss()

# Save current parameters
theta_star = torch.cat([p.flatten() for p in model.parameters()])

# Random direction
d = torch.randn_like(theta_star)
d = d / d.norm()

# Evaluate loss along direction
alphas = torch.linspace(-2, 2, 50)
losses = []
for alpha in alphas:
    # Perturb parameters
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data = (theta_star[offset:offset+numel] + alpha * d[offset:offset+numel]).reshape(p.shape)
        offset += numel
    losses.append(criterion(model(X), y).item())

print(f"Loss range: [{min(losses):.4f}, {max(losses):.4f}]")
print(f"Loss at origin: {losses[25]:.4f}")`}
      />

      <NoteBlock type="note" title="Sharp vs Flat Minima">
        <p>
          <strong>Flat minima</strong> (small Hessian eigenvalues) tend to generalize better than
          sharp minima. SGD with small batch sizes and large learning rates implicitly biases
          optimization toward flatter regions. This connection between optimization and generalization
          is an active area of research (PAC-Bayes bounds, sharpness-aware minimization).
        </p>
      </NoteBlock>
    </div>
  )
}

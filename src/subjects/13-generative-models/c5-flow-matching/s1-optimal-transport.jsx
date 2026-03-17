import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function OTPathViz() {
  const [t, setT] = useState(0.5)
  const W = 340, H = 160

  const pairs = [
    { x0: 40, y0: 30, x1: 280, y1: 40 },
    { x0: 60, y0: 80, x1: 300, y1: 90 },
    { x0: 30, y0: 130, x1: 260, y1: 120 },
    { x0: 70, y0: 50, x1: 290, y1: 60 },
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Optimal Transport Paths</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        t = {t.toFixed(2)}
        <input type="range" min={0} max={1} step={0.02} value={t} onChange={e => setT(parseFloat(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        {pairs.map((p, i) => {
          const cx = p.x0 + (p.x1 - p.x0) * t
          const cy = p.y0 + (p.y1 - p.y0) * t
          return (
            <g key={i}>
              <line x1={p.x0} y1={p.y0} x2={p.x1} y2={p.y1} stroke="#d1d5db" strokeWidth={0.8} strokeDasharray="3,3" />
              <circle cx={p.x0} cy={p.y0} r={4} fill="#8b5cf6" opacity={0.4} />
              <circle cx={p.x1} cy={p.y1} r={4} fill="#f97316" opacity={0.4} />
              <circle cx={cx} cy={cy} r={5} fill="#8b5cf6" />
            </g>
          )
        })}
        <text x={20} y={H - 5} className="text-[10px] fill-violet-500">noise (t=0)</text>
        <text x={W - 80} y={H - 5} className="text-[10px] fill-orange-500">data (t=1)</text>
      </svg>
    </div>
  )
}

export default function OptimalTransport() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Flow matching learns a vector field that transports a simple distribution to the data
        distribution. When paths are straight (via optimal transport), training is more efficient
        and sampling requires fewer steps than diffusion models.
      </p>

      <DefinitionBlock title="Flow Matching Objective">
        <p>Learn a time-dependent vector field <InlineMath math="v_\theta(\mathbf{x}, t)" /> that generates a probability path from noise to data:</p>
        <BlockMath math="\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, q(\mathbf{x}_1)}\left[\|v_\theta(\mathbf{x}_t, t) - u_t(\mathbf{x}_t | \mathbf{x}_1)\|^2\right]" />
        <p className="mt-2">
          where <InlineMath math="u_t" /> is the target vector field and <InlineMath math="\mathbf{x}_t" /> interpolates between noise and data.
        </p>
      </DefinitionBlock>

      <TheoremBlock title="Conditional OT Path" id="cot-path">
        <p>The optimal transport conditional path is a straight line:</p>
        <BlockMath math="\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\mathbf{x}_1, \quad \mathbf{x}_0 \sim \mathcal{N}(0, I),\; \mathbf{x}_1 \sim q(\mathbf{x})" />
        <p className="mt-2">The target vector field is constant along each path:</p>
        <BlockMath math="u_t(\mathbf{x}_t | \mathbf{x}_1) = \mathbf{x}_1 - \mathbf{x}_0" />
      </TheoremBlock>

      <OTPathViz />

      <ExampleBlock title="Flow Matching vs Diffusion">
        <p>
          Diffusion models use curved, stochastic paths (adding/removing noise gradually).
          Flow matching with OT uses straight paths from noise to data. Benefits:
          (1) simulation-free training (no ODE solving), (2) straighter trajectories need fewer
          integration steps at inference, (3) simpler implementation.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Flow Matching Training"
        code={`import torch
import torch.nn as nn

class VectorField(nn.Module):
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t.view(-1, 1).expand(-1, 1)], dim=-1))

def flow_matching_loss(model, x1):
    """OT conditional flow matching loss."""
    B = x1.shape[0]
    t = torch.rand(B)
    x0 = torch.randn_like(x1)  # noise samples

    # Straight-line interpolation
    x_t = (1 - t.view(-1, 1)) * x0 + t.view(-1, 1) * x1

    # Target: direction from noise to data
    target = x1 - x0

    # Predict vector field
    v_pred = model(x_t, t)
    return ((v_pred - target) ** 2).mean()

# Sampling: integrate the learned vector field
@torch.no_grad()
def sample(model, shape, steps=50):
    x = torch.randn(shape)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((shape[0],), i * dt)
        x = x + model(x, t) * dt  # Euler integration
    return x

model = VectorField(dim=2)
data = torch.randn(256, 2) + 3  # shifted Gaussian
loss = flow_matching_loss(model, data)
print(f"Flow matching loss: {loss.item():.4f}")`}
      />

      <NoteBlock type="note" title="Stable Diffusion 3 Uses Flow Matching">
        <p>
          Modern text-to-image models like Stable Diffusion 3 have moved from DDPM-style diffusion
          to rectified flow matching, benefiting from straighter sampling trajectories and
          better training stability. The core insight: straight paths are easier to learn and
          faster to sample.
        </p>
      </NoteBlock>
    </div>
  )
}

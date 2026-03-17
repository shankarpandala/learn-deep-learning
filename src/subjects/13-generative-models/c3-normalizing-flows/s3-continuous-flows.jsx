import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ODETrajectoryViz() {
  const [timeSteps, setTimeSteps] = useState(10)
  const W = 360, H = 160, cx = W / 2, cy = H / 2

  const trajectory = Array.from({ length: timeSteps + 1 }, (_, i) => {
    const t = i / timeSteps
    const x = cx + 60 * Math.cos(t * Math.PI * 1.2) * (1 - 0.3 * t)
    const y = cy - 50 * Math.sin(t * Math.PI * 1.5) * (0.3 + 0.7 * t)
    return { x, y }
  })

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Neural ODE Trajectory</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Integration steps: {timeSteps}
        <input type="range" min={3} max={30} step={1} value={timeSteps} onChange={e => setTimeSteps(parseInt(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        {trajectory.map((p, i) => i > 0 && (
          <line key={i} x1={trajectory[i - 1].x} y1={trajectory[i - 1].y} x2={p.x} y2={p.y} stroke="#8b5cf6" strokeWidth={1.5} opacity={0.6} />
        ))}
        {trajectory.map((p, i) => (
          <circle key={i} cx={p.x} cy={p.y} r={i === 0 ? 5 : i === timeSteps ? 5 : 2.5}
            fill={i === 0 ? '#8b5cf6' : i === timeSteps ? '#f97316' : '#a78bfa'} />
        ))}
        <text x={trajectory[0].x + 8} y={trajectory[0].y - 5} className="text-[10px] fill-violet-600">z(0)</text>
        <text x={trajectory[timeSteps].x + 8} y={trajectory[timeSteps].y - 5} className="text-[10px] fill-orange-600">z(1)=x</text>
      </svg>
    </div>
  )
}

export default function ContinuousFlows() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Continuous normalizing flows (CNFs) parameterize transformations as solutions to ordinary
        differential equations, replacing discrete flow steps with a continuous-time dynamics
        governed by a neural network.
      </p>

      <DefinitionBlock title="Neural ODE">
        <p>A neural ODE defines the dynamics of a hidden state via:</p>
        <BlockMath math="\frac{d\mathbf{z}(t)}{dt} = f_\theta(\mathbf{z}(t), t)" />
        <p className="mt-2">
          The output is obtained by integrating from <InlineMath math="t_0" /> to <InlineMath math="t_1" />:
        </p>
        <BlockMath math="\mathbf{z}(t_1) = \mathbf{z}(t_0) + \int_{t_0}^{t_1} f_\theta(\mathbf{z}(t), t)\,dt" />
      </DefinitionBlock>

      <TheoremBlock title="Instantaneous Change of Variables" id="inst-change-vars">
        <p>For a continuous normalizing flow, the log-density evolves as:</p>
        <BlockMath math="\frac{\partial \log p(\mathbf{z}(t))}{\partial t} = -\text{tr}\left(\frac{\partial f_\theta}{\partial \mathbf{z}(t)}\right)" />
        <p className="mt-2">
          This avoids computing full Jacobian determinants. The trace can be estimated stochastically
          via the Hutchinson trace estimator: <InlineMath math="\text{tr}(A) = \mathbb{E}_{\epsilon}[\epsilon^\top A \epsilon]" />.
        </p>
      </TheoremBlock>

      <ODETrajectoryViz />

      <PythonCode
        title="Neural ODE with torchdiffeq"
        code={`import torch
import torch.nn as nn
# from torchdiffeq import odeint  # pip install torchdiffeq

class ODEFunc(nn.Module):
    def __init__(self, dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t, z):
        # Concatenate time to input
        t_expand = t.expand(z.shape[0], 1)
        return self.net(torch.cat([z, t_expand], dim=-1))

# Simple Euler integration (in practice, use adaptive ODE solvers)
def euler_integrate(func, z0, t_span, steps=20):
    dt = (t_span[1] - t_span[0]) / steps
    z = z0
    t = t_span[0]
    for _ in range(steps):
        z = z + dt * func(torch.tensor([t]), z)
        t += dt
    return z

func = ODEFunc(dim=2)
z0 = torch.randn(16, 2)
z1 = euler_integrate(func, z0, t_span=(0.0, 1.0), steps=20)
print(f"z(0): {z0.shape} -> z(1): {z1.shape}")
print(f"Moved distance: {(z1 - z0).norm(dim=-1).mean():.3f}")`}
      />

      <ExampleBlock title="FFJORD: Free-Form Jacobian of Reversible Dynamics">
        <p>
          FFJORD combines the CNF with the Hutchinson trace estimator for unbiased, scalable
          log-likelihood computation. It avoids all architectural restrictions (coupling layers,
          autoregressive structure) — the vector field <InlineMath math="f_\theta" /> can be any
          neural network.
        </p>
      </ExampleBlock>

      <NoteBlock type="note" title="From CNFs to Flow Matching">
        <p>
          Training CNFs via maximum likelihood requires solving an ODE at every training step,
          which is expensive. Flow matching (covered in Chapter 5) provides a simulation-free
          alternative: directly regressing the vector field against a target, dramatically
          reducing training cost.
        </p>
      </NoteBlock>
    </div>
  )
}

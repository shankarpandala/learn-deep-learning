import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ConsistencyViz() {
  const [numSteps, setNumSteps] = useState(1)
  const W = 360, H = 120

  const trajectory = Array.from({ length: 6 }, (_, i) => ({
    x: 30 + i * 60,
    y: 60 + 30 * Math.sin(i * 0.8) * (1 - i / 6),
    t: (5 - i) / 5,
  }))

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Consistency Model: Any Point Maps to Origin</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Sampling steps: {numSteps}
        <input type="range" min={1} max={4} step={1} value={numSteps} onChange={e => setNumSteps(parseInt(e.target.value))} className="w-32 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        {trajectory.map((p, i) => i > 0 && (
          <line key={`l${i}`} x1={trajectory[i - 1].x} y1={trajectory[i - 1].y} x2={p.x} y2={p.y} stroke="#d1d5db" strokeWidth={1} strokeDasharray="3,3" />
        ))}
        {trajectory.map((p, i) => {
          const active = i === 0 || (numSteps >= 4 ? true : i >= 6 - numSteps - 1)
          return (
            <g key={i}>
              <circle cx={p.x} cy={p.y} r={active ? 5 : 3} fill={i === 5 ? '#f97316' : '#8b5cf6'} opacity={active ? 1 : 0.3} />
              {active && i < 5 && <line x1={p.x} y1={p.y} x2={trajectory[5].x} y2={trajectory[5].y} stroke="#8b5cf6" strokeWidth={1.5} opacity={0.4} />}
            </g>
          )
        })}
        <text x={trajectory[0].x} y={15} textAnchor="middle" className="text-[9px] fill-gray-500">t=T (noise)</text>
        <text x={trajectory[5].x} y={15} textAnchor="middle" className="text-[9px] fill-orange-500">t=0 (data)</text>
      </svg>
      <p className="text-xs text-gray-500 text-center mt-1">
        All points on the same trajectory map to the same output. {numSteps === 1 ? 'Single step: direct jump from any t to data.' : `${numSteps} steps: multi-step refinement.`}
      </p>
    </div>
  )
}

export default function ConsistencyModels() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Consistency models learn to map any point on a diffusion trajectory directly to the
        trajectory's origin (clean data), enabling one-step generation without iterative denoising.
      </p>

      <DefinitionBlock title="Consistency Function">
        <p>A consistency function <InlineMath math="f_\theta" /> satisfies the self-consistency property:</p>
        <BlockMath math="f_\theta(\mathbf{x}_t, t) = f_\theta(\mathbf{x}_{t'}, t') \quad \forall\, t, t' \in [\epsilon, T]" />
        <p className="mt-2">
          For any two points on the same PF-ODE trajectory, the model produces the same output.
          The boundary condition is <InlineMath math="f_\theta(\mathbf{x}_\epsilon, \epsilon) = \mathbf{x}_\epsilon \approx \mathbf{x}_0" />.
        </p>
      </DefinitionBlock>

      <ConsistencyViz />

      <TheoremBlock title="Consistency Training Loss" id="ct-loss">
        <p>Enforce consistency between adjacent timesteps using the target network <InlineMath math="\theta^{-}" /> (EMA):</p>
        <BlockMath math="\mathcal{L}_{\text{CT}} = \mathbb{E}\left[d\left(f_\theta(\mathbf{x}_{t_{n+1}}, t_{n+1}),\; f_{\theta^{-}}(\hat{\mathbf{x}}_{t_n}, t_n)\right)\right]" />
        <p className="mt-2">
          where <InlineMath math="\hat{\mathbf{x}}_{t_n}" /> is obtained by one step of a numerical ODE solver
          from <InlineMath math="\mathbf{x}_{t_{n+1}}" />, and <InlineMath math="d" /> is a distance metric (e.g., LPIPS).
        </p>
      </TheoremBlock>

      <ExampleBlock title="Consistency Distillation vs Training">
        <p>
          <strong>Consistency distillation</strong> (CD) requires a pre-trained diffusion model to generate
          ODE trajectories. <strong>Consistency training</strong> (CT) trains from scratch by estimating
          the ODE step with a single denoiser evaluation, removing the dependency on a teacher model.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Consistency Model Pseudocode"
        code={`import torch
import torch.nn as nn

class ConsistencyModel(nn.Module):
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        self.eps = 0.002  # boundary epsilon

    def forward(self, x, t):
        # Skip connection enforces boundary condition: f(x, eps) = x
        t_input = t.unsqueeze(-1)
        c_skip = self.eps / (t_input ** 2 + self.eps ** 2).sqrt()
        c_out = (t_input - self.eps) / (t_input ** 2 + self.eps ** 2).sqrt()
        return c_skip * x + c_out * self.net(torch.cat([x, t_input], dim=-1))

def consistency_loss(model, target_model, x0, noise_schedule):
    B = x0.shape[0]
    # Sample adjacent timestep pairs
    n = torch.randint(0, len(noise_schedule) - 1, (B,))
    t_next = noise_schedule[n + 1]
    t_curr = noise_schedule[n]

    noise = torch.randn_like(x0)
    x_next = x0 + t_next.unsqueeze(-1) * noise

    # One ODE step estimate (using pre-trained denoiser or self)
    x_curr = x0 + t_curr.unsqueeze(-1) * noise  # simplified

    pred = model(x_next, t_next)
    with torch.no_grad():
        target = target_model(x_curr, t_curr)
    return ((pred - target) ** 2).mean()

model = ConsistencyModel()
x = torch.randn(8, 2)
t = torch.ones(8) * 0.5
out = model(x, t)
print(f"One-step generation: input {x.shape} -> output {out.shape}")`}
      />

      <NoteBlock type="note" title="Improved Consistency Training (iCT)">
        <p>
          Improved consistency training removes the need for a pre-trained diffusion model entirely,
          using adaptive schedules for the number of discretization steps and the EMA decay rate.
          iCT achieves state-of-the-art FID for single-step generation on ImageNet, making it a
          compelling alternative to multi-step diffusion.
        </p>
      </NoteBlock>
    </div>
  )
}

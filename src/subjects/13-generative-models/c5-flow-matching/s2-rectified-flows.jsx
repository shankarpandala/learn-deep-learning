import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function RectificationViz() {
  const [iteration, setIteration] = useState(0)
  const W = 340, H = 150

  const curviness = Math.max(0, 1 - iteration * 0.35)

  const paths = [
    { x0: 30, y0: 130, x1: 310, y1: 30 },
    { x0: 50, y0: 100, x1: 280, y1: 50 },
    { x0: 40, y0: 60, x1: 300, y1: 110 },
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Rectification Iterations</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Reflow iteration: {iteration}
        <input type="range" min={0} max={3} step={1} value={iteration} onChange={e => setIteration(parseInt(e.target.value))} className="w-32 accent-violet-500" />
        <span className="text-xs text-violet-600">straightness: {(1 - curviness).toFixed(0) === '1' ? '1.00' : (1 - curviness).toFixed(2)}</span>
      </label>
      <svg width={W} height={H} className="mx-auto block">
        {paths.map((p, i) => {
          const mx = (p.x0 + p.x1) / 2 + curviness * (40 - i * 30)
          const my = (p.y0 + p.y1) / 2 + curviness * (20 * (i - 1))
          return (
            <g key={i}>
              <path d={`M${p.x0},${p.y0} Q${mx},${my} ${p.x1},${p.y1}`} fill="none" stroke="#8b5cf6" strokeWidth={2} />
              <circle cx={p.x0} cy={p.y0} r={4} fill="#8b5cf6" />
              <circle cx={p.x1} cy={p.y1} r={4} fill="#f97316" />
            </g>
          )
        })}
      </svg>
      <p className="text-xs text-gray-500 text-center mt-1">
        {iteration === 0 ? 'Initial: curved trajectories from diffusion training' :
         iteration === 1 ? 'After 1 reflow: noticeably straighter' :
         iteration === 2 ? 'After 2 reflows: nearly straight' :
         'After 3 reflows: almost perfectly straight (1-step generation possible)'}
      </p>
    </div>
  )
}

export default function RectifiedFlows() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Rectified flows iteratively straighten the transport paths between noise and data
        distributions. Straighter paths enable accurate generation with fewer integration steps,
        approaching single-step generation.
      </p>

      <DefinitionBlock title="Rectified Flow (Reflow)">
        <p>Starting with a learned flow <InlineMath math="v_\theta" />, generate coupled pairs <InlineMath math="(\mathbf{x}_0, \mathbf{x}_1)" /> by running the ODE, then retrain on straight-line interpolants:</p>
        <BlockMath math="\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1, \quad u_t = \mathbf{x}_1 - \mathbf{x}_0" />
        <p className="mt-2">Each reflow iteration produces straighter trajectories, reducing truncation error.</p>
      </DefinitionBlock>

      <RectificationViz />

      <TheoremBlock title="Straightness Bound" id="straightness">
        <p>
          The transport cost (path curvature) is non-increasing with each reflow iteration:
        </p>
        <BlockMath math="\mathbb{E}\left[\int_0^1 \|v_{\theta}^{(k+1)}(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0)\|^2 dt\right] \leq \mathbb{E}\left[\int_0^1 \|v_{\theta}^{(k)}(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0)\|^2 dt\right]" />
        <p className="mt-2">In the limit, trajectories become straight lines and one-step generation is exact.</p>
      </TheoremBlock>

      <ExampleBlock title="Distillation for One-Step Generation">
        <p>
          After rectification, further distill into a single-step model. The student network learns
          to predict <InlineMath math="\mathbf{x}_1" /> directly from <InlineMath math="\mathbf{x}_0" />:
        </p>
        <BlockMath math="\mathcal{L}_{\text{distill}} = \|\text{Student}(\mathbf{x}_0) - \text{ODE}(\mathbf{x}_0; v_\theta)\|^2" />
        <p>This gives single-step generation with quality approaching the multi-step teacher.</p>
      </ExampleBlock>

      <PythonCode
        title="Rectified Flow: Reflow Procedure"
        code={`import torch
import torch.nn as nn

class FlowModel(nn.Module):
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, t.unsqueeze(-1)], dim=-1))

@torch.no_grad()
def generate_pairs(model, n=1000, steps=100):
    """Generate (x0, x1) pairs by running the ODE."""
    x0 = torch.randn(n, 2)
    x = x0.clone()
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), i * dt)
        x = x + model(x, t) * dt
    x1 = x
    return x0, x1  # coupled noise-data pairs

def reflow_loss(model, x0, x1):
    """Train on straight-line interpolants of coupled pairs."""
    B = x0.shape[0]
    t = torch.rand(B)
    x_t = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
    target = x1 - x0  # straight-line direction
    v_pred = model(x_t, t)
    return ((v_pred - target) ** 2).mean()

model = FlowModel()
# Reflow: generate pairs -> retrain -> repeat
# x0, x1 = generate_pairs(model)
# loss = reflow_loss(new_model, x0, x1)
print("Reflow procedure:")
print("1. Train initial flow matching model")
print("2. Generate coupled (noise, data) pairs via ODE")
print("3. Retrain on straight-line interpolants")
print("4. Repeat 1-2 times for near-straight trajectories")`}
      />

      <NoteBlock type="note" title="InstaFlow and Practical Applications">
        <p>
          InstaFlow applies rectified flows to Stable Diffusion, achieving one-step text-to-image
          generation. The combination of flow matching + reflow + distillation has become a
          leading paradigm for fast generative models, powering systems like Stable Diffusion 3
          and FLUX.
        </p>
      </NoteBlock>
    </div>
  )
}

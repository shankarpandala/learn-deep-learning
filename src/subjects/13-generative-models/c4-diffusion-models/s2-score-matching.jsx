import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ScoreFieldViz() {
  const [noiseLevel, setNoiseLevel] = useState(0.5)
  const W = 300, H = 200, cx = W / 2, cy = H / 2

  const arrows = []
  for (let gx = 30; gx < W; gx += 40) {
    for (let gy = 30; gy < H; gy += 40) {
      const dx = cx - gx
      const dy = cy - gy
      const dist = Math.sqrt(dx * dx + dy * dy) + 1
      const scale = Math.min(15, 200 / (dist * noiseLevel + 10))
      const nx = (dx / dist) * scale
      const ny = (dy / dist) * scale
      arrows.push({ x: gx, y: gy, dx: nx, dy: ny })
    }
  }

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Score Field Visualization</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Noise level (sigma): {noiseLevel.toFixed(2)}
        <input type="range" min={0.1} max={2} step={0.05} value={noiseLevel} onChange={e => setNoiseLevel(parseFloat(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <circle cx={cx} cy={cy} r={8} fill="#8b5cf6" opacity={0.3} />
        {arrows.map((a, i) => (
          <line key={i} x1={a.x} y1={a.y} x2={a.x + a.dx} y2={a.y + a.dy} stroke="#8b5cf6" strokeWidth={1.5} markerEnd="url(#arrowhead)" />
        ))}
        <defs>
          <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
            <polygon points="0 0, 6 2, 0 4" fill="#8b5cf6" />
          </marker>
        </defs>
        <text x={cx} y={cy + 22} textAnchor="middle" className="text-[10px] fill-violet-600">data mode</text>
      </svg>
      <p className="text-xs text-gray-500 text-center mt-1">
        Arrows show the score (gradient of log-density) pointing toward high-density regions.
        {noiseLevel < 0.5 ? ' Low noise: sharp arrows near the mode.' : ' High noise: smoother, more spread field.'}
      </p>
    </div>
  )
}

export default function ScoreMatching() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Score-based generative models learn the gradient of the log-probability (the score function)
        and generate samples by following these gradients via Langevin dynamics. This perspective
        unifies diffusion models with stochastic differential equations.
      </p>

      <DefinitionBlock title="Score Function">
        <p>The score of a distribution <InlineMath math="p(\mathbf{x})" /> is the gradient of its log-density:</p>
        <BlockMath math="\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})" />
        <p className="mt-2">
          A score network <InlineMath math="\mathbf{s}_\theta(\mathbf{x}, \sigma)" /> is trained to approximate
          the score of the noise-perturbed distribution <InlineMath math="p_\sigma(\mathbf{x})" />.
        </p>
      </DefinitionBlock>

      <ScoreFieldViz />

      <TheoremBlock title="Denoising Score Matching" id="dsm">
        <p>
          Instead of directly matching the intractable true score, we match the score of
          noisy data <InlineMath math="q_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}; \mathbf{x}, \sigma^2 I)" />:
        </p>
        <BlockMath math="\mathcal{L}_{\text{DSM}} = \mathbb{E}_{\mathbf{x}, \tilde{\mathbf{x}}}\left[\|\mathbf{s}_\theta(\tilde{\mathbf{x}}, \sigma) - \nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}}|\mathbf{x})\|^2\right]" />
        <p className="mt-2">
          Since <InlineMath math="\nabla_{\tilde{\mathbf{x}}} \log q_\sigma = -(\tilde{\mathbf{x}} - \mathbf{x})/\sigma^2" />,
          this is equivalent to noise prediction (the DDPM objective).
        </p>
      </TheoremBlock>

      <ExampleBlock title="Langevin Dynamics Sampling">
        <p>Given the score, generate samples via annealed Langevin dynamics:</p>
        <BlockMath math="\mathbf{x}_{i+1} = \mathbf{x}_i + \frac{\eta}{2}\,\mathbf{s}_\theta(\mathbf{x}_i, \sigma) + \sqrt{\eta}\,\mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(0, I)" />
        <p className="mt-1">The noise levels <InlineMath math="\sigma" /> are annealed from large to small during sampling.</p>
      </ExampleBlock>

      <PythonCode
        title="Score Matching and Langevin Sampling"
        code={`import torch
import torch.nn as nn

class ScoreNet(nn.Module):
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, sigma):
        sigma_input = sigma.expand(x.shape[0], 1)
        return self.net(torch.cat([x, sigma_input], dim=-1))

# Denoising score matching loss
def dsm_loss(model, x, sigma=0.5):
    noise = torch.randn_like(x)
    x_noisy = x + sigma * noise
    score_pred = model(x_noisy, torch.tensor([sigma]))
    target = -noise / sigma  # true score of Gaussian perturbation
    return ((score_pred - target) ** 2).sum(dim=-1).mean()

# Langevin dynamics sampling
@torch.no_grad()
def langevin_sample(model, shape, sigmas, steps_per_sigma=100, lr=0.01):
    x = torch.randn(shape)
    for sigma in sigmas:
        for _ in range(steps_per_sigma):
            score = model(x, torch.tensor([sigma]))
            x = x + (lr / 2) * score + torch.sqrt(torch.tensor(lr)) * torch.randn_like(x)
    return x

model = ScoreNet(dim=2)
x_data = torch.randn(256, 2) * 0.5 + torch.tensor([2.0, 2.0])
loss = dsm_loss(model, x_data)
print(f"DSM loss: {loss.item():.4f}")`}
      />

      <NoteBlock type="note" title="SDE Framework: Unifying Diffusion and Score Models">
        <p>
          Song et al. showed that both DDPM and score-based models are discretizations of a
          continuous-time SDE: <InlineMath math="d\mathbf{x} = f(\mathbf{x},t)\,dt + g(t)\,d\mathbf{w}" />.
          The reverse SDE uses the score: <InlineMath math="d\mathbf{x} = [f - g^2 \nabla_x \log p_t]\,dt + g\,d\bar{\mathbf{w}}" />.
          This unification enables flexible solver choices (ODE for deterministic, SDE for stochastic sampling).
        </p>
      </NoteBlock>
    </div>
  )
}

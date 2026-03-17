import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function NoisingProcessViz() {
  const [timestep, setTimestep] = useState(500)
  const T = 1000
  const alphaBar = Math.exp(-0.0001 * timestep - 0.02 * (timestep / T) * (timestep / T) * T * 0.5)
  const noiseLevel = (1 - alphaBar).toFixed(3)
  const signalLevel = alphaBar.toFixed(3)

  const W = 360, H = 80
  const barW = 300

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Forward Noising Process</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        t = {timestep} / {T}
        <input type="range" min={0} max={T} step={10} value={timestep} onChange={e => setTimestep(parseInt(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <rect x={30} y={20} width={barW * alphaBar} height={24} rx={4} fill="#8b5cf6" />
        <rect x={30 + barW * alphaBar} y={20} width={barW * (1 - alphaBar)} height={24} rx={4} fill="#f97316" opacity={0.6} />
        <text x={30 + barW * alphaBar / 2} y={36} textAnchor="middle" className="text-[10px] fill-white font-semibold">signal ({signalLevel})</text>
        {(1 - alphaBar) > 0.15 && <text x={30 + barW * alphaBar + barW * (1 - alphaBar) / 2} y={36} textAnchor="middle" className="text-[10px] fill-white font-semibold">noise ({noiseLevel})</text>}
        <text x={30} y={60} className="text-[9px] fill-gray-500">clean image</text>
        <text x={barW + 10} y={60} textAnchor="end" className="text-[9px] fill-gray-500">pure noise</text>
      </svg>
    </div>
  )
}

export default function DDPM() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Denoising Diffusion Probabilistic Models (DDPM) generate data by learning to reverse a
        gradual noising process. Starting from pure Gaussian noise, the model iteratively denoises
        to produce high-quality samples.
      </p>

      <DefinitionBlock title="Forward Process (Noising)">
        <p>Gradually add Gaussian noise over <InlineMath math="T" /> steps with schedule <InlineMath math="\beta_1, \ldots, \beta_T" />:</p>
        <BlockMath math="q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\; \beta_t \mathbf{I})" />
        <p className="mt-2">The closed-form for any timestep with <InlineMath math="\bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)" />:</p>
        <BlockMath math="q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\; (1-\bar{\alpha}_t)\mathbf{I})" />
      </DefinitionBlock>

      <NoisingProcessViz />

      <TheoremBlock title="DDPM Training Objective" id="ddpm-loss">
        <p>The simplified training loss reduces to predicting the noise added at each step:</p>
        <BlockMath math="\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]" />
        <p className="mt-2">
          where <InlineMath math="\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}" /> and <InlineMath math="\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})" />.
        </p>
      </TheoremBlock>

      <PythonCode
        title="DDPM Training Loop Core"
        code={`import torch
import torch.nn as nn

T = 1000
# Linear beta schedule
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

def q_sample(x0, t, noise=None):
    """Forward process: add noise to x0 at timestep t."""
    if noise is None:
        noise = torch.randn_like(x0)
    ab_t = alpha_bar[t].view(-1, 1, 1, 1)  # for image shapes
    return torch.sqrt(ab_t) * x0 + torch.sqrt(1 - ab_t) * noise

def training_step(model, x0, optimizer):
    B = x0.shape[0]
    t = torch.randint(0, T, (B,))
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise)

    predicted_noise = model(x_t, t)
    loss = nn.MSELoss()(predicted_noise, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# The model (U-Net) predicts the noise epsilon
# Sampling: iteratively denoise from x_T ~ N(0,I) to x_0
print(f"alpha_bar at t=0: {alpha_bar[0]:.4f}")
print(f"alpha_bar at t=500: {alpha_bar[500]:.4f}")
print(f"alpha_bar at t=999: {alpha_bar[999]:.6f}")`}
      />

      <ExampleBlock title="Sampling (Reverse Process)">
        <p>Starting from <InlineMath math="\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})" />, iterate:</p>
        <BlockMath math="\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right) + \sigma_t \mathbf{z}" />
        <p className="mt-1">This requires <InlineMath math="T" /> forward passes (typically 1000), making sampling slow.</p>
      </ExampleBlock>

      <NoteBlock type="note" title="DDIM: Faster Sampling">
        <p>
          DDIM (Denoising Diffusion Implicit Models) uses a non-Markovian reverse process that
          allows skipping steps, reducing sampling from 1000 to as few as 20-50 steps with
          minimal quality loss. The key insight is that the same trained model can be sampled
          with different schedules.
        </p>
      </NoteBlock>
    </div>
  )
}

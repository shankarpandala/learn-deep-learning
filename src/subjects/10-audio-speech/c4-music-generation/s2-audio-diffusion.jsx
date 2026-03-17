import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function DiffusionStepsVisualizer() {
  const [steps, setSteps] = useState(50)
  const [cfgScale, setCfgScale] = useState(3.0)

  const snrValues = Array.from({ length: 5 }, (_, i) => {
    const t = (i / 4)
    return { t: (t * steps).toFixed(0), noise: (t * 100).toFixed(0), signal: ((1 - t) * 100).toFixed(0) }
  })

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Audio Diffusion Parameters</h3>
      <div className="flex flex-wrap gap-4 mb-4">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Denoising steps: {steps}
          <input type="range" min={10} max={200} step={10} value={steps} onChange={e => setSteps(Number(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          CFG scale: {cfgScale.toFixed(1)}
          <input type="range" min={1} max={10} step={0.5} value={cfgScale} onChange={e => setCfgScale(Number(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <div className="flex gap-2">
        {snrValues.map((v, i) => (
          <div key={i} className="flex-1 rounded-lg overflow-hidden">
            <div className="bg-violet-500 text-white text-center text-xs py-1" style={{ height: `${100 - Number(v.noise)}%`, minHeight: '20px' }}>
              {v.signal}%
            </div>
            <div className="bg-gray-300 dark:bg-gray-600 text-center text-xs py-1" style={{ height: `${Number(v.noise)}%`, minHeight: '20px' }}>
              {v.noise}%
            </div>
            <p className="text-xs text-center text-gray-500 mt-1">t={v.t}</p>
          </div>
        ))}
      </div>
      <p className="text-xs text-gray-500 mt-2">
        CFG strength {cfgScale.toFixed(1)}: {cfgScale < 3 ? 'More diverse, less adherent to prompt' : cfgScale < 6 ? 'Balanced quality and diversity' : 'High adherence, may reduce diversity'}
      </p>
    </div>
  )
}

export default function AudioDiffusionModels() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Diffusion models have emerged as a powerful paradigm for audio generation, producing
        high-quality sound effects, music, and speech by iteratively denoising latent
        representations conditioned on text descriptions.
      </p>

      <DefinitionBlock title="Latent Diffusion for Audio (AudioLDM)">
        <p>AudioLDM applies latent diffusion to mel spectrogram generation:</p>
        <BlockMath math="\mathcal{L} = \mathbb{E}_{z_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(z_t, t, c_\text{text})\|^2\right]" />
        <p className="mt-2">
          A VAE encodes mel spectrograms into latent space <InlineMath math="z_0" />, the diffusion
          model operates in this compressed space, and a vocoder (HiFi-GAN) converts the
          decoded mel back to audio. Text conditioning uses CLAP embeddings.
        </p>
      </DefinitionBlock>

      <DiffusionStepsVisualizer />

      <ExampleBlock title="Classifier-Free Guidance for Audio">
        <p>Audio diffusion models use CFG to balance quality and diversity:</p>
        <BlockMath math="\hat{\epsilon}_\theta(z_t, c) = \epsilon_\theta(z_t, \varnothing) + s \cdot [\epsilon_\theta(z_t, c) - \epsilon_\theta(z_t, \varnothing)]" />
        <p className="mt-1">
          where <InlineMath math="s" /> is the guidance scale. During training, the text condition
          <InlineMath math="c" /> is randomly dropped 10-20% of the time to enable unconditional generation.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Audio Generation with Diffusion (Simplified)"
        code={`import torch
import torch.nn as nn

class SimpleAudioUNet(nn.Module):
    """Simplified U-Net for audio latent diffusion."""
    def __init__(self, latent_dim=64, cond_dim=512, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.cond_proj = nn.Linear(cond_dim, time_dim)
        # Simplified encoder-decoder
        self.encoder = nn.Conv1d(latent_dim, 128, 3, padding=1)
        self.mid = nn.Conv1d(128, 128, 3, padding=1)
        self.decoder = nn.Conv1d(128, latent_dim, 3, padding=1)

    def forward(self, z_t, t, cond):
        t_emb = self.time_mlp(t.unsqueeze(-1)) + self.cond_proj(cond)
        h = self.encoder(z_t) + t_emb.unsqueeze(-1)
        h = torch.relu(self.mid(h))
        return self.decoder(h)

# Diffusion sampling loop
model = SimpleAudioUNet()
cond = torch.randn(1, 512)  # text embedding
z_t = torch.randn(1, 64, 100)  # start from noise
steps = 50

for i in range(steps, 0, -1):
    t = torch.tensor([i / steps])
    with torch.no_grad():
        noise_pred = model(z_t, t, cond)
    # Simplified DDPM step
    alpha = 1 - (i / steps) * 0.02
    z_t = (z_t - (1 - alpha) * noise_pred) / alpha**0.5

print(f"Denoised latent: {z_t.shape}")  # [1, 64, 100]`}
      />

      <WarningBlock title="Audio Diffusion Challenges">
        <p>
          Audio diffusion faces unique challenges: long sequences (30s audio = 1.5M samples),
          temporal coherence requirements, and the need for perceptually meaningful losses.
          Latent diffusion mitigates the computational cost, but generating long, coherent
          audio remains an open problem.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Stable Audio and Beyond">
        <p>
          Stable Audio from Stability AI applies latent diffusion with timing conditioning,
          enabling generation of variable-length audio up to 95 seconds. The model conditions
          on both text and timing embeddings (start time, total duration), providing fine-grained
          control over the generated audio's temporal structure.
        </p>
      </NoteBlock>
    </div>
  )
}

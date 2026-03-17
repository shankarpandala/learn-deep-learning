import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function DenoisingStepViz() {
  const [step, setStep] = useState(50)
  const [cfgScale, setCfgScale] = useState(7.5)
  const noiseLevel = Math.max(0, 1 - step / 50)
  const signalLevel = 1 - noiseLevel

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Latent Diffusion Denoising Process</h3>
      <div className="flex items-center gap-4 mb-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Denoising step: {step}/50
          <input type="range" min={0} max={50} step={1} value={step} onChange={e => setStep(parseInt(e.target.value))} className="w-32 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          CFG scale: {cfgScale.toFixed(1)}
          <input type="range" min={1} max={20} step={0.5} value={cfgScale} onChange={e => setCfgScale(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <span className="text-xs w-16 text-gray-500">Noise</span>
          <div className="flex-1 h-4 bg-gray-100 dark:bg-gray-800 rounded overflow-hidden">
            <div className="h-full bg-gray-400 transition-all duration-200" style={{ width: `${noiseLevel * 100}%` }} />
          </div>
          <span className="text-xs w-10 text-right font-mono">{(noiseLevel * 100).toFixed(0)}%</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs w-16 text-gray-500">Signal</span>
          <div className="flex-1 h-4 bg-gray-100 dark:bg-gray-800 rounded overflow-hidden">
            <div className="h-full bg-violet-500 transition-all duration-200" style={{ width: `${signalLevel * 100}%` }} />
          </div>
          <span className="text-xs w-10 text-right font-mono">{(signalLevel * 100).toFixed(0)}%</span>
        </div>
      </div>
      <p className="mt-2 text-xs text-gray-500">CFG scale {cfgScale.toFixed(1)}: higher values produce images more aligned with the text prompt but with less diversity.</p>
    </div>
  )
}

export default function StableDiffusionLatentDiffusion() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Stable Diffusion performs the diffusion process in a compressed latent space rather than
        pixel space, dramatically reducing compute. A text encoder (CLIP) provides conditioning,
        and a VAE decoder converts latents back to high-resolution images.
      </p>

      <DefinitionBlock title="Latent Diffusion Model (LDM)">
        <p>The key insight is performing diffusion in latent space <InlineMath math="z = \mathcal{E}(x)" /> where <InlineMath math="\mathcal{E}" /> is a pretrained encoder:</p>
        <BlockMath math="\mathcal{L}_{\text{LDM}} = \mathbb{E}_{z_0, \epsilon, t, c}\left[\|\epsilon - \epsilon_\theta(z_t, t, c)\|^2\right]" />
        <p className="mt-2">where <InlineMath math="z_t = \sqrt{\bar{\alpha}_t}z_0 + \sqrt{1-\bar{\alpha}_t}\epsilon" /> is the noised latent, <InlineMath math="c = f_\text{CLIP}(\text{text})" /> is the text conditioning, and <InlineMath math="\epsilon_\theta" /> is a U-Net with cross-attention.</p>
      </DefinitionBlock>

      <DenoisingStepViz />

      <DefinitionBlock title="Classifier-Free Guidance (CFG)">
        <p>CFG interpolates between conditional and unconditional predictions to strengthen text alignment:</p>
        <BlockMath math="\tilde{\epsilon}_\theta(z_t, c) = \epsilon_\theta(z_t, \varnothing) + s \cdot (\epsilon_\theta(z_t, c) - \epsilon_\theta(z_t, \varnothing))" />
        <p className="mt-2">where <InlineMath math="s" /> is the guidance scale (typically 7-12). During training, the text condition <InlineMath math="c" /> is randomly dropped (replaced with <InlineMath math="\varnothing" />) 10-20% of the time.</p>
      </DefinitionBlock>

      <PythonCode
        title="Stable Diffusion with Diffusers"
        code={`from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

# Load Stable Diffusion with optimized scheduler
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config
)  # 20-25 steps instead of 50 with DDPM
pipe = pipe.to("cuda")

# Text-to-image generation
image = pipe(
    prompt="A serene mountain landscape at sunset, oil painting style",
    negative_prompt="blurry, low quality, distorted",
    num_inference_steps=25,     # denoising steps
    guidance_scale=7.5,         # CFG strength
    height=512, width=512,
).images[0]
image.save("mountain.png")

# Key components of the pipeline:
# 1. Text encoder: CLIP ViT-L/14 (frozen) -> text embeddings
# 2. VAE encoder/decoder: 8x spatial compression (512->64)
# 3. U-Net: cross-attention conditioned denoising in latent space
print(f"Latent space: 4x64x64 = {4*64*64:,} values")
print(f"Pixel space:  3x512x512 = {3*512*512:,} values")
print(f"Compression ratio: {3*512*512 / (4*64*64):.0f}x")

# SDXL: latest version with dual text encoders + refiner
from diffusers import StableDiffusionXLPipeline
pipe_xl = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16",
)`}
      />

      <ExampleBlock title="Compute Savings from Latent Space">
        <p>For a 512x512 image with the Stable Diffusion VAE (downsampling factor 8):</p>
        <BlockMath math="\text{Pixel space: } 512 \times 512 \times 3 = 786{,}432 \text{ values}" />
        <BlockMath math="\text{Latent space: } 64 \times 64 \times 4 = 16{,}384 \text{ values}" />
        <p>This is a <strong>48x reduction</strong> in dimensionality, making training and inference dramatically cheaper.</p>
      </ExampleBlock>

      <NoteBlock type="note" title="Stable Diffusion Components">
        <p>
          The full pipeline has three pretrained components: (1) a VAE encoder/decoder for pixel-latent
          conversion, (2) a CLIP text encoder for conditioning, and (3) a U-Net with cross-attention
          that performs the actual denoising. Only the U-Net is trained for the diffusion task.
        </p>
      </NoteBlock>
    </div>
  )
}

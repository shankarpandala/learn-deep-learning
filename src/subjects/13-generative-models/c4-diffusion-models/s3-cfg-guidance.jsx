import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function GuidanceScaleViz() {
  const [w, setW] = useState(7.5)
  const diversity = Math.max(0, 1 - (w - 1) * 0.08)
  const quality = Math.min(1, 0.3 + w * 0.08)
  const saturation = w > 12 ? Math.min(1, (w - 12) * 0.1) : 0

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Classifier-Free Guidance Scale</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        w = {w.toFixed(1)}
        <input type="range" min={1} max={20} step={0.5} value={w} onChange={e => setW(parseFloat(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <div className="flex gap-4 justify-center items-end h-24">
        <div className="flex flex-col items-center">
          <div className="w-14 bg-violet-400 rounded-t transition-all" style={{ height: `${quality * 80}px` }} />
          <span className="text-xs text-gray-500 mt-1">Quality</span>
        </div>
        <div className="flex flex-col items-center">
          <div className="w-14 bg-violet-600 rounded-t transition-all" style={{ height: `${diversity * 80}px` }} />
          <span className="text-xs text-gray-500 mt-1">Diversity</span>
        </div>
        <div className="flex flex-col items-center">
          <div className="w-14 bg-red-400 rounded-t transition-all" style={{ height: `${saturation * 80}px` }} />
          <span className="text-xs text-gray-500 mt-1">Artifacts</span>
        </div>
      </div>
      <p className="text-xs text-gray-500 text-center mt-2">
        {w < 3 ? 'Low guidance: diverse but may not match the condition well' :
         w <= 10 ? 'Sweet spot: good balance of quality and condition adherence' :
         'High guidance: oversaturated, artifacts may appear'}
      </p>
    </div>
  )
}

export default function CFGGuidance() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Classifier-free guidance (CFG) enables conditional generation without a separate classifier,
        trading off sample diversity for stronger adherence to the conditioning signal. It has become
        the standard approach in text-to-image models like Stable Diffusion and DALL-E.
      </p>

      <DefinitionBlock title="Classifier-Free Guidance">
        <p>During training, randomly drop the condition <InlineMath math="c" /> (replace with null) with probability <InlineMath math="p_{\text{uncond}}" />. At inference, combine conditional and unconditional predictions:</p>
        <BlockMath math="\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, c) = (1 + w)\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, c) - w\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \varnothing)" />
        <p className="mt-2">
          where <InlineMath math="w" /> is the guidance scale. Equivalently in score form:
        </p>
        <BlockMath math="\tilde{\nabla} \log p(\mathbf{x}_t | c) = \nabla \log p(\mathbf{x}_t) + (1+w)\left(\nabla \log p(c | \mathbf{x}_t)\right)" />
      </DefinitionBlock>

      <GuidanceScaleViz />

      <ExampleBlock title="Classifier Guidance (Original Approach)">
        <p>
          The original approach by Dhariwal & Nichol uses a separate classifier <InlineMath math="p_\phi(c|\mathbf{x}_t)" /> trained on noisy data:
        </p>
        <BlockMath math="\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - s \cdot \sigma_t \nabla_{\mathbf{x}_t} \log p_\phi(c|\mathbf{x}_t)" />
        <p className="mt-1">
          CFG eliminates the need for this external classifier, simplifying the pipeline.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Classifier-Free Guidance Sampling"
        code={`import torch

def cfg_sample_step(model, x_t, t, condition, w=7.5, null_cond=None):
    """Single CFG denoising step."""
    # Conditional prediction
    eps_cond = model(x_t, t, condition)
    # Unconditional prediction (condition dropped)
    eps_uncond = model(x_t, t, null_cond)
    # Guided prediction
    eps_guided = (1 + w) * eps_cond - w * eps_uncond
    return eps_guided

# Training with random condition dropout
def training_step(model, x0, condition, p_uncond=0.1):
    t = torch.randint(0, 1000, (x0.shape[0],))
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise)  # forward noising

    # Randomly drop condition
    mask = torch.rand(x0.shape[0]) < p_uncond
    cond_input = condition.clone()
    cond_input[mask] = 0  # null condition (e.g., zero embedding)

    eps_pred = model(x_t, t, cond_input)
    loss = ((eps_pred - noise) ** 2).mean()
    return loss

# Typical guidance scales:
# w=1.0: minimal guidance
# w=7.5: default for Stable Diffusion
# w=15+: very strong guidance (risk of artifacts)
print("CFG: two forward passes per step (conditional + unconditional)")
print("Doubles inference cost but dramatically improves condition adherence")`}
      />

      <WarningBlock title="Guidance Scale Pitfalls">
        <p>
          Very high guidance scales (<InlineMath math="w > 15" />) can cause color saturation,
          loss of fine detail, and unrealistic artifacts. Dynamic guidance (varying <InlineMath math="w" /> across
          timesteps) and guidance rescaling can mitigate these issues.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Beyond Text Conditioning">
        <p>
          CFG works with any conditioning signal: text embeddings (Stable Diffusion), class labels
          (ImageNet generation), spatial maps (ControlNet), or even image references (IP-Adapter).
          The same principle applies: train with dropout, guide at inference.
        </p>
      </NoteBlock>
    </div>
  )
}

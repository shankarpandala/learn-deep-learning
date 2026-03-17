import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function BatchNormViz() {
  const [batchSize, setBatchSize] = useState(8)
  const [gamma, setGamma] = useState(1.0)
  const [betaP, setBetaP] = useState(0.0)
  const W = 380, H = 180

  const raw = Array.from({ length: batchSize }, (_, i) => 2 * Math.sin(i * 1.3) + 3)
  const mean = raw.reduce((a, b) => a + b, 0) / raw.length
  const variance = raw.reduce((a, b) => a + (b - mean) ** 2, 0) / raw.length
  const normed = raw.map(x => gamma * (x - mean) / Math.sqrt(variance + 1e-5) + betaP)

  const allVals = [...raw, ...normed]
  const minV = Math.min(...allVals) - 0.5
  const maxV = Math.max(...allVals) + 0.5
  const sx = W / (batchSize + 1), sy = (H - 30) / (maxV - minV)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Batch Norm Effect</h3>
      <div className="flex items-center gap-4 mb-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          γ = {gamma.toFixed(1)}
          <input type="range" min={0.1} max={3} step={0.1} value={gamma} onChange={e => setGamma(parseFloat(e.target.value))} className="w-24 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          β = {betaP.toFixed(1)}
          <input type="range" min={-2} max={2} step={0.1} value={betaP} onChange={e => setBetaP(parseFloat(e.target.value))} className="w-24 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Batch: {batchSize}
          <input type="range" min={4} max={16} step={1} value={batchSize} onChange={e => setBatchSize(parseInt(e.target.value))} className="w-24 accent-violet-500" />
        </label>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={0} y1={H - 20} x2={W} y2={H - 20} stroke="#d1d5db" strokeWidth={0.5} />
        {raw.map((v, i) => (
          <g key={`r-${i}`}>
            <circle cx={(i + 0.7) * sx} cy={H - 20 - (v - minV) * sy} r={5} fill="#9ca3af" opacity={0.6} />
            <circle cx={(i + 0.7) * sx} cy={H - 20 - (normed[i] - minV) * sy} r={5} fill="#8b5cf6" />
          </g>
        ))}
      </svg>
      <div className="mt-2 flex justify-center gap-6 text-xs">
        <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-full bg-gray-400" /> Raw</span>
        <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-full bg-violet-500" /> Normalized</span>
      </div>
      <div className="mt-1 text-center text-xs text-gray-500">
        μ = {mean.toFixed(2)}, σ² = {variance.toFixed(2)}
      </div>
    </div>
  )
}

export default function BatchNorm() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Batch Normalization (Ioffe & Szegedy, 2015) normalizes activations across the mini-batch,
        enabling faster training with higher learning rates and reducing sensitivity to initialization.
      </p>

      <DefinitionBlock title="Batch Normalization">
        <BlockMath math="\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}" />
        <BlockMath math="y_i = \gamma \hat{x}_i + \beta" />
        <p className="mt-2">
          where <InlineMath math="\mu_B" /> and <InlineMath math="\sigma_B^2" /> are computed
          over the mini-batch, and <InlineMath math="\gamma, \beta" /> are learned affine parameters.
        </p>
      </DefinitionBlock>

      <BatchNormViz />

      <TheoremBlock title="Smoothing Effect" id="bn-smoothing">
        <p>
          Rather than reducing &ldquo;internal covariate shift&rdquo; as originally claimed,
          BatchNorm's main benefit is making the loss landscape significantly smoother:
        </p>
        <BlockMath math="\|\nabla \mathcal{L}_{\text{BN}}\| \leq \|\nabla \mathcal{L}\| \cdot \frac{\gamma}{\sqrt{\sigma_B^2 + \epsilon}}" />
        <p>
          This allows larger learning rates without divergence (Santurkar et al., 2018).
        </p>
      </TheoremBlock>

      <ExampleBlock title="Train vs Eval Mode">
        <p>
          During training, BN uses mini-batch statistics. During evaluation, it uses running
          averages accumulated during training. Forgetting to switch modes
          (<code>model.eval()</code>) is a common source of bugs that causes inference
          performance to degrade.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Batch Normalization in PyTorch"
        code={`import torch
import torch.nn as nn

# BN for fully connected layers
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),  # normalizes across batch dim
    nn.ReLU(),
    nn.Linear(256, 10)
)

# BN for convolutional layers
conv_model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.BatchNorm2d(64),   # normalizes across batch, H, W
    nn.ReLU(),
)

# Training vs evaluation mode matters!
model.train()   # uses batch statistics
x_train = torch.randn(32, 784)
out = model(x_train)

model.eval()    # uses running mean/var
x_test = torch.randn(1, 784)
out = model(x_test)  # works even with batch_size=1
print(f"Output shape: {out.shape}")`}
      />

      <WarningBlock title="Small Batch Sizes">
        <p>
          BatchNorm degrades with small batch sizes (below ~16) because the mini-batch statistics
          become noisy. For small batches, use GroupNorm or LayerNorm instead. This is particularly
          relevant for object detection and segmentation models with large inputs.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="BatchNorm's Legacy">
        <p>
          BatchNorm was transformative for CNNs and remains the default normalization for
          vision models. However, Transformers and RNNs typically use LayerNorm due to
          variable sequence lengths and the desire for batch-independent normalization.
        </p>
      </NoteBlock>
    </div>
  )
}

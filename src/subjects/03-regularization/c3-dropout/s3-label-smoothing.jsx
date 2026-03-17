import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function LabelSmoothingViz() {
  const [epsilon, setEpsilon] = useState(0.1)
  const K = 5
  const W = 380, H = 160, barW = 50, gap = 15

  const hardLabel = [0, 0, 1, 0, 0]
  const smoothLabel = hardLabel.map(y => y * (1 - epsilon) + epsilon / K)

  const startX = (W - K * (barW + gap)) / 2

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Label Smoothing Visualization</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        <InlineMath math="\epsilon" /> = {epsilon.toFixed(2)}
        <input type="range" min={0} max={0.5} step={0.01} value={epsilon} onChange={e => setEpsilon(parseFloat(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        {smoothLabel.map((v, i) => {
          const x = startX + i * (barW + gap)
          const barH = v * 100
          return (
            <g key={i}>
              <rect x={x} y={H - 30 - barH} width={barW} height={barH} fill="#8b5cf6" rx={3} opacity={0.8} />
              <rect x={x} y={H - 30 - hardLabel[i] * 100} width={barW} height={hardLabel[i] * 100} fill="none" stroke="#f97316" strokeWidth={1.5} strokeDasharray="3,3" rx={3} />
              <text x={x + barW / 2} y={H - 12} textAnchor="middle" fontSize={10} fill="#6b7280">Class {i}</text>
              <text x={x + barW / 2} y={H - 34 - barH} textAnchor="middle" fontSize={9} fill="#8b5cf6">{v.toFixed(3)}</text>
            </g>
          )
        })}
      </svg>
      <div className="mt-1 flex justify-center gap-4 text-xs text-gray-500">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-2 bg-violet-500 rounded-sm" /> Smoothed</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-2 border border-orange-500 rounded-sm" /> Hard</span>
      </div>
    </div>
  )
}

export default function LabelSmoothing() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Label smoothing and mixup are regularization techniques that soften the training
        signal, preventing the model from becoming overconfident and improving calibration.
      </p>

      <DefinitionBlock title="Label Smoothing">
        <p>Replace hard one-hot targets <InlineMath math="y" /> with smoothed targets:</p>
        <BlockMath math="y_{\text{smooth}} = (1 - \epsilon) \cdot y + \frac{\epsilon}{K}" />
        <p className="mt-2">
          where <InlineMath math="\epsilon" /> is the smoothing parameter (typically 0.1) and
          <InlineMath math="K" /> is the number of classes.
        </p>
      </DefinitionBlock>

      <LabelSmoothingViz />

      <DefinitionBlock title="Mixup">
        <p>Create virtual training examples by interpolating pairs:</p>
        <BlockMath math="\tilde{x} = \lambda x_i + (1 - \lambda) x_j, \quad \tilde{y} = \lambda y_i + (1 - \lambda) y_j" />
        <p className="mt-2">where <InlineMath math="\lambda \sim \text{Beta}(\alpha, \alpha)" /> and <InlineMath math="\alpha" /> controls interpolation strength (typically 0.2-0.4).</p>
      </DefinitionBlock>

      <TheoremBlock title="CutMix" id="cutmix">
        <p>CutMix replaces a rectangular region of one image with a patch from another:</p>
        <BlockMath math="\tilde{x} = \mathbf{M} \odot x_i + (1 - \mathbf{M}) \odot x_j" />
        <p className="mt-2">
          where <InlineMath math="\mathbf{M}" /> is a binary mask. The label is mixed proportionally to
          the area ratio: <InlineMath math="\tilde{y} = \lambda y_i + (1-\lambda) y_j" /> where
          <InlineMath math="\lambda" /> is the fraction of the unmasked area.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Benefits of Soft Targets">
        <ul className="list-disc ml-4 space-y-1">
          <li><strong>Better calibration</strong>: model probabilities reflect true uncertainty</li>
          <li><strong>Reduced overconfidence</strong>: logits don't grow unboundedly</li>
          <li><strong>Knowledge distillation</strong>: dark knowledge in soft targets carries inter-class similarities</li>
          <li><strong>Label noise robustness</strong>: smoothing reduces impact of mislabeled examples</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Label Smoothing & Mixup in PyTorch"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

# Label Smoothing Cross-Entropy (built-in since PyTorch 1.10)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

logits = torch.randn(8, 10)  # batch=8, classes=10
targets = torch.randint(0, 10, (8,))
loss = criterion(logits, targets)
print(f"Label smoothed loss: {loss:.4f}")

# Mixup implementation
def mixup_data(x, y, alpha=0.2):
    lam = torch.distributions.Beta(alpha, alpha).sample()
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def mixup_criterion(pred, y_a, y_b, lam):
    return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)

# Usage in training loop
x = torch.randn(32, 3, 32, 32)
y = torch.randint(0, 10, (32,))
mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
print(f"Mixup lambda: {lam:.4f}")`}
      />

      <NoteBlock type="note" title="Combining Techniques">
        <p>
          Label smoothing and mixup are complementary but using both together requires care.
          CutMix generally outperforms vanilla mixup for image classification. Modern training
          recipes (e.g., for ViT) often combine label smoothing
          (<InlineMath math="\epsilon = 0.1" />) with mixup (<InlineMath math="\alpha = 0.8" />)
          and CutMix (<InlineMath math="\alpha = 1.0" />).
        </p>
      </NoteBlock>
    </div>
  )
}

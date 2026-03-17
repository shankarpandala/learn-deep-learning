import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function bce(p, y) { return y === 1 ? -Math.log(Math.max(p, 1e-7)) : -Math.log(Math.max(1 - p, 1e-7)) }
function focal(p, y, gamma) {
  const pt = y === 1 ? p : 1 - p
  return -Math.pow(1 - pt, gamma) * Math.log(Math.max(pt, 1e-7))
}

function LossCurvePlot() {
  const [gamma, setGamma] = useState(2.0)
  const [label, setLabel] = useState(1)
  const W = 420, H = 250, padL = 30, padB = 25
  const plotW = W - padL, plotH = H - padB

  const range = Array.from({ length: 100 }, (_, i) => 0.01 + i * 0.0098)
  const toSVG = (p, l) => `${padL + p * plotW},${plotH - Math.min(l, 5) * (plotH / 5.5)}`

  const bcePath = range.map((p, i) => `${i === 0 ? 'M' : 'L'}${toSVG(p, bce(p, label))}`).join(' ')
  const focalPath = range.map((p, i) => `${i === 0 ? 'M' : 'L'}${toSVG(p, focal(p, label, gamma))}`).join(' ')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Cross-Entropy vs Focal Loss</h3>
      <div className="flex flex-wrap items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          True label:
          <select value={label} onChange={e => setLabel(Number(e.target.value))} className="rounded border px-1 dark:bg-gray-800 dark:border-gray-600">
            <option value={1}>y = 1</option>
            <option value={0}>y = 0</option>
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Focal gamma = {gamma.toFixed(1)}
          <input type="range" min={0} max={5} step={0.5} value={gamma} onChange={e => setGamma(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={padL} y1={plotH} x2={W} y2={plotH} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={padL} y1={0} x2={padL} y2={plotH} stroke="#d1d5db" strokeWidth={0.5} />
        <text x={W / 2} y={H - 3} fontSize={10} fill="#9ca3af" textAnchor="middle">predicted probability</text>
        <text x={12} y={plotH / 2} fontSize={10} fill="#9ca3af" textAnchor="middle" transform={`rotate(-90,12,${plotH / 2})`}>loss</text>
        <path d={bcePath} fill="none" stroke="#8b5cf6" strokeWidth={2.5} />
        <path d={focalPath} fill="none" stroke="#f97316" strokeWidth={2.5} />
      </svg>
      <div className="mt-2 flex justify-center gap-6 text-xs">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-violet-500" /> Cross-Entropy</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-orange-500" /> Focal (gamma={gamma})</span>
      </div>
    </div>
  )
}

export default function ClassificationLosses() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Classification losses measure how well predicted probability distributions match true
        class labels. Cross-entropy is the workhorse of classification, while focal loss
        addresses the critical problem of class imbalance.
      </p>

      <DefinitionBlock title="Binary Cross-Entropy (BCE)">
        <BlockMath math="\mathcal{L}_{\text{BCE}} = -\frac{1}{n}\sum_{i=1}^{n}\bigl[y_i \log(\hat{p}_i) + (1 - y_i)\log(1 - \hat{p}_i)\bigr]" />
        <p className="mt-2">
          Where <InlineMath math="\hat{p}_i = \sigma(z_i)" /> is the sigmoid output and <InlineMath math="y_i \in \{0, 1\}" />.
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="Softmax & Categorical Cross-Entropy">
        <BlockMath math="\text{softmax}(z_k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}" />
        <BlockMath math="\mathcal{L}_{\text{CE}} = -\sum_{k=1}^{K} y_k \log\bigl(\text{softmax}(z_k)\bigr)" />
        <p className="mt-2">
          For one-hot encoded targets this simplifies to the negative log-likelihood of
          the correct class: <InlineMath math="\mathcal{L} = -\log(\hat{p}_{c})" />.
        </p>
      </DefinitionBlock>

      <TheoremBlock title="Cross-Entropy as KL Divergence" id="ce-kl">
        <p>
          Minimizing cross-entropy <InlineMath math="H(p, q)" /> is equivalent to minimizing
          the KL divergence from the true distribution <InlineMath math="p" /> to the model
          distribution <InlineMath math="q" />:
        </p>
        <BlockMath math="H(p, q) = H(p) + D_{\text{KL}}(p \| q)" />
        <p>
          Since <InlineMath math="H(p)" /> is fixed, minimizing cross-entropy minimizes
          the KL divergence.
        </p>
      </TheoremBlock>

      <DefinitionBlock title="Focal Loss (Lin et al., 2017)">
        <BlockMath math="\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)" />
        <p className="mt-2">
          Where <InlineMath math="p_t" /> is the predicted probability for the true class
          and <InlineMath math="\gamma \geq 0" /> is the focusing parameter.
          When <InlineMath math="\gamma = 0" /> this reduces to standard cross-entropy.
          Higher <InlineMath math="\gamma" /> down-weights easy examples, focusing training
          on hard, misclassified samples.
        </p>
      </DefinitionBlock>

      <LossCurvePlot />

      <WarningBlock title="Numerical Stability">
        <p>
          Never compute <InlineMath math="\log(\text{softmax}(z))" /> in two separate steps.
          Use <code>log_softmax</code> which is numerically stable via the log-sum-exp trick:
          <InlineMath math="\log \text{softmax}(z_k) = z_k - \log\sum_j e^{z_j}" />.
          In PyTorch, prefer <code>nn.CrossEntropyLoss</code> which combines both.
        </p>
      </WarningBlock>

      <PythonCode
        title="Classification Losses in PyTorch"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

# Binary cross-entropy
logits = torch.tensor([1.5, -0.5, 2.0, -1.0])
labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
bce = nn.BCEWithLogitsLoss()
print(f"BCE: {bce(logits, labels):.4f}")

# Multi-class cross-entropy (expects raw logits)
logits_mc = torch.randn(4, 5)       # batch=4, classes=5
targets_mc = torch.tensor([2, 0, 4, 1])
ce = nn.CrossEntropyLoss()
print(f"CE:  {ce(logits_mc, targets_mc):.4f}")

# Focal loss (custom implementation)
def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none'
    )
    p_t = torch.exp(-bce_loss)  # p_t = sigmoid when y=1
    loss = alpha * (1 - p_t) ** gamma * bce_loss
    return loss.mean()

print(f"Focal: {focal_loss(logits, labels):.4f}")`}
      />

      <ExampleBlock title="Impact of Focal Loss on Class Imbalance">
        <p>
          Consider a detector where 99.9% of anchors are background. A well-classified
          background with <InlineMath math="p_t = 0.99" /> contributes to standard CE:
        </p>
        <BlockMath math="-\log(0.99) \approx 0.01" />
        <p>
          With focal loss (<InlineMath math="\gamma = 2" />): <InlineMath math="(1 - 0.99)^2 \times 0.01 = 10^{-6}" />.
          The easy negatives are effectively silenced, letting the model focus on hard positives.
        </p>
      </ExampleBlock>

      <NoteBlock type="tip" title="Choosing a Classification Loss">
        <p>
          <strong>BCE / CE</strong>: Default for balanced datasets.
          <strong> Focal Loss</strong>: Use when facing severe class imbalance (detection, medical imaging).
          <strong> Label Smoothing</strong>: Regularization technique that replaces hard targets
          with <InlineMath math="y_{\text{smooth}} = (1-\epsilon)y + \epsilon/K" /> to prevent overconfidence.
        </p>
      </NoteBlock>
    </div>
  )
}

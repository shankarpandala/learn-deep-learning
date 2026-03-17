import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function SegmentationDemo() {
  const [dilation, setDilation] = useState(1)
  const gridSize = 7
  const center = 3
  const W = 220, cellSize = W / gridSize

  const isActive = (r, c) => {
    const dr = Math.abs(r - center)
    const dc = Math.abs(c - center)
    return (dr === 0 && dc <= dilation) || (dc === 0 && dr <= dilation) ||
      (dr === dilation && dc === dilation) || (dr <= 1 && dc <= 1)
  }

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Dilated Convolution Receptive Field</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Dilation rate: {dilation}
        <input type="range" min={1} max={3} step={1} value={dilation}
          onChange={e => setDilation(parseInt(e.target.value))} className="w-32 accent-violet-500" />
      </label>
      <svg width={W} height={W} className="mx-auto block">
        {Array.from({ length: gridSize }).map((_, r) =>
          Array.from({ length: gridSize }).map((_, c) => (
            <rect key={`${r}-${c}`} x={c * cellSize} y={r * cellSize} width={cellSize - 1} height={cellSize - 1}
              fill={r === center && c === center ? '#8b5cf6' : isActive(r, c) ? '#c4b5fd' : '#f3f4f6'}
              stroke="#d1d5db" strokeWidth={0.5} rx={2} />
          ))
        )}
      </svg>
      <p className="mt-2 text-center text-xs text-gray-500">
        Receptive field: {(2 * dilation + 1)}x{(2 * dilation + 1)} | No resolution loss
      </p>
    </div>
  )
}

export default function SemanticSeg() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Semantic segmentation assigns a class label to every pixel in an image. Encoder-decoder
        architectures and dilated convolutions are the foundational techniques for this task.
      </p>

      <DefinitionBlock title="Semantic Segmentation">
        <p>
          Given an image <InlineMath math="I \in \mathbb{R}^{H \times W \times 3}" />, predict a
          label map <InlineMath math="Y \in \{1, \ldots, C\}^{H \times W}" />:
        </p>
        <BlockMath math="Y_{ij} = \arg\max_c \; f_\theta(I)_{ijc}" />
        <p className="mt-2">
          The per-pixel cross-entropy loss sums over all spatial locations:
        </p>
        <BlockMath math="\mathcal{L} = -\frac{1}{HW}\sum_{i,j}\sum_{c} y_{ijc}\log\hat{y}_{ijc}" />
      </DefinitionBlock>

      <SegmentationDemo />

      <TheoremBlock title="Dilated (Atrous) Convolution" id="dilated-conv">
        <p>
          Dilated convolution with rate <InlineMath math="r" /> expands the kernel without adding parameters:
        </p>
        <BlockMath math="(f *_r k)(p) = \sum_{s+rt=p} f(s) \cdot k(t)" />
        <p className="mt-1">
          Effective receptive field for kernel size <InlineMath math="k" /> with dilation <InlineMath math="r" />:
          <InlineMath math="\; k_{\text{eff}} = k + (k-1)(r-1)" />. This captures multi-scale context
          without downsampling.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Key Architectures">
        <ul className="list-disc ml-5 space-y-1">
          <li><strong>FCN</strong>: First fully convolutional approach, upsamples via transposed convolutions</li>
          <li><strong>U-Net</strong>: Skip connections between encoder and decoder at each scale</li>
          <li><strong>DeepLabv3+</strong>: ASPP module with parallel dilated convolutions (rates 6, 12, 18)</li>
          <li><strong>PSPNet</strong>: Pyramid pooling at multiple scales for global context</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="DeepLabv3+ Segmentation"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""
    def __init__(self, in_ch=2048, out_ch=256, rates=[6, 12, 18]):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, 1)
        self.atrous = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r)
            for r in rates
        ])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1),
        )
        self.project = nn.Conv2d(out_ch * (len(rates) + 2), out_ch, 1)

    def forward(self, x):
        h, w = x.shape[2:]
        feats = [self.conv1x1(x)]
        feats += [conv(x) for conv in self.atrous]
        feats.append(F.interpolate(self.pool(x), (h, w),
                                   mode='bilinear'))
        return self.project(torch.cat(feats, dim=1))

# Mean IoU evaluation metric
def mean_iou(pred, target, num_classes):
    ious = []
    for c in range(num_classes):
        inter = ((pred == c) & (target == c)).sum().float()
        union = ((pred == c) | (target == c)).sum().float()
        ious.append((inter / union.clamp(min=1)).item())
    return sum(ious) / len(ious)`}
      />

      <NoteBlock type="note" title="Class Imbalance in Segmentation">
        <p>
          Pixel-level class imbalance is severe (e.g., road vs sign in driving scenes). Solutions
          include weighted cross-entropy, focal loss, and Dice loss:
          <InlineMath math="\mathcal{L}_{\text{Dice}} = 1 - \frac{2|P \cap G|}{|P| + |G|}" />.
          Combining cross-entropy with Dice loss often gives the best results.
        </p>
      </NoteBlock>
    </div>
  )
}

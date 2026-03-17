import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ASPPVisualization() {
  const [showRates, setShowRates] = useState(true)
  const rates = [1, 6, 12, 18]
  const cellSize = 5, gridN = 25, padX = 10, padY = 30, gridGap = 20
  const gridW = gridN * cellSize

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Atrous Spatial Pyramid Pooling (ASPP)</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        <input type="checkbox" checked={showRates} onChange={e => setShowRates(e.target.checked)} className="accent-violet-500" />
        Highlight kernel sampling positions
      </label>
      <div className="flex justify-center gap-4">
        {rates.map((rate, ri) => {
          const center = Math.floor(gridN / 2)
          return (
            <div key={ri} className="text-center">
              <svg width={gridW + 2 * padX} height={gridW + padY + 10}>
                <text x={padX + gridW / 2} y={14} textAnchor="middle" fontSize={10} fill="#6b7280">rate={rate}</text>
                {Array.from({ length: gridN }, (_, r) =>
                  Array.from({ length: gridN }, (_, c) => {
                    const isSample = showRates && Math.abs(r - center) <= rate && Math.abs(c - center) <= rate &&
                      (r - center) % rate === 0 && (c - center) % rate === 0
                    return (
                      <rect key={`${r}-${c}`} x={padX + c * cellSize} y={padY + r * cellSize}
                        width={cellSize - 0.5} height={cellSize - 0.5}
                        fill={isSample ? '#8b5cf6' : '#f3f4f6'} stroke="#e5e7eb" strokeWidth={0.3} />
                    )
                  })
                )}
              </svg>
              <p className="text-xs text-gray-500">RF: {1 + 2 * rate}x{1 + 2 * rate}</p>
            </div>
          )
        })}
      </div>
      <p className="text-xs text-center text-gray-500 mt-2">
        3x3 kernels with different dilation rates capture multi-scale context
      </p>
    </div>
  )
}

export default function DeepLab() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        DeepLab (Chen et al.) introduced atrous (dilated) convolutions and Atrous Spatial Pyramid
        Pooling (ASPP) for semantic segmentation. By using dilated convolutions, DeepLab maintains
        high-resolution feature maps while capturing multi-scale context without extra parameters.
      </p>

      <DefinitionBlock title="Atrous (Dilated) Convolution for Segmentation">
        <p>
          Standard CNNs reduce spatial resolution by <InlineMath math="32\times" /> through pooling.
          DeepLab replaces pooling in later stages with atrous convolutions of increasing rates,
          maintaining <InlineMath math="8\times" /> or <InlineMath math="16\times" /> output stride:
        </p>
        <BlockMath math="y[i] = \sum_k x[i + r \cdot k] \cdot w[k]" />
        <p className="mt-2">Where <InlineMath math="r" /> is the dilation rate.</p>
      </DefinitionBlock>

      <TheoremBlock title="ASPP Module" id="aspp">
        <p>ASPP applies multiple parallel atrous convolutions at different rates and concatenates:</p>
        <BlockMath math="\text{ASPP}(x) = \text{Cat}\left[f_{1 \times 1}(x), \; f^{r=6}_{3 \times 3}(x), \; f^{r=12}_{3 \times 3}(x), \; f^{r=18}_{3 \times 3}(x), \; \text{GAP}(x)\right]" />
        <p className="mt-2">
          The global average pooling branch captures image-level context. All branches are fused
          with a <InlineMath math="1 \times 1" /> convolution.
        </p>
      </TheoremBlock>

      <ASPPVisualization />

      <ExampleBlock title="DeepLab Evolution">
        <p>
          DeepLab v1: Atrous convolutions + CRF post-processing.{' '}
          DeepLab v2: Added ASPP for multi-scale.{' '}
          DeepLab v3: Improved ASPP with batch norm and image pooling.{' '}
          DeepLab v3+: Added decoder module with skip connections from low-level features.
        </p>
      </ExampleBlock>

      <PythonCode
        title="ASPP Module in PyTorch"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256, rates=[6, 12, 18]):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.atrous = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
            for r in rates])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        n_branches = 2 + len(rates)
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * n_branches, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        h, w = x.shape[2:]
        branches = [self.conv1x1(x)]
        branches += [m(x) for m in self.atrous]
        branches.append(F.interpolate(
            self.pool(x), size=(h, w), mode='bilinear', align_corners=False))
        return self.project(torch.cat(branches, dim=1))

aspp = ASPP(2048)
x = torch.randn(1, 2048, 32, 32)
print(f"ASPP output: {aspp(x).shape}")  # [1, 256, 32, 32]`}
      />

      <NoteBlock type="note" title="Output Stride and CRF">
        <p>
          DeepLab uses output stride 16 (or 8 for higher resolution) and bilinear upsampling for
          final predictions. Early versions used Conditional Random Fields (CRF) as post-processing
          to refine boundaries, but DeepLab v3+ dropped CRF in favor of the encoder-decoder design.
        </p>
      </NoteBlock>
    </div>
  )
}

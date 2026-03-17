import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function CostComparison() {
  const [channels, setChannels] = useState(64)
  const [kernelSize, setKernelSize] = useState(3)
  const [spatialSize, setSpatialSize] = useState(32)

  const standardOps = channels * channels * kernelSize * kernelSize * spatialSize * spatialSize
  const depthwiseOps = channels * kernelSize * kernelSize * spatialSize * spatialSize
  const pointwiseOps = channels * channels * spatialSize * spatialSize
  const separableOps = depthwiseOps + pointwiseOps
  const ratio = (separableOps / standardOps * 100).toFixed(1)

  const fmt = (n) => n > 1e9 ? (n / 1e9).toFixed(2) + 'G' : n > 1e6 ? (n / 1e6).toFixed(2) + 'M' : (n / 1e3).toFixed(0) + 'K'

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Computational Cost Comparison</h3>
      <div className="grid grid-cols-3 gap-3 mb-4">
        {[
          ['Channels', channels, setChannels, 8, 512],
          ['Kernel', kernelSize, setKernelSize, 1, 7],
          ['Spatial', spatialSize, setSpatialSize, 4, 64],
        ].map(([label, val, setter, min, max]) => (
          <label key={label} className="text-sm text-gray-600 dark:text-gray-400">
            {label}: <strong>{val}</strong>
            <input type="range" min={min} max={max} step={label === 'Kernel' ? 2 : 1} value={val} onChange={e => setter(parseInt(e.target.value))} className="w-full accent-violet-500" />
          </label>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div className="p-3 rounded bg-gray-50 dark:bg-gray-800">
          <p className="font-semibold text-gray-700 dark:text-gray-300">Standard Conv</p>
          <p className="text-violet-600 dark:text-violet-400 font-mono">{fmt(standardOps)} ops</p>
        </div>
        <div className="p-3 rounded bg-gray-50 dark:bg-gray-800">
          <p className="font-semibold text-gray-700 dark:text-gray-300">Depthwise Separable</p>
          <p className="text-violet-600 dark:text-violet-400 font-mono">{fmt(separableOps)} ops ({ratio}%)</p>
        </div>
      </div>
    </div>
  )
}

export default function DepthwiseSeparable() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Depthwise separable convolutions, introduced in MobileNet, factorize a standard convolution
        into two steps to dramatically reduce computation while maintaining accuracy.
      </p>

      <DefinitionBlock title="Depthwise Separable Convolution">
        <p>Factorized into two operations:</p>
        <p className="mt-2"><strong>1. Depthwise:</strong> One <InlineMath math="K \times K" /> filter per input channel (no cross-channel mixing).</p>
        <p><strong>2. Pointwise:</strong> A <InlineMath math="1 \times 1" /> convolution to combine channels.</p>
      </DefinitionBlock>

      <TheoremBlock title="Computational Savings" id="dw-savings">
        <p>Standard convolution cost for <InlineMath math="C_{in} = C_{out} = C" />:</p>
        <BlockMath math="\text{Standard} = K^2 \cdot C^2 \cdot H \cdot W" />
        <p>Depthwise separable cost:</p>
        <BlockMath math="\text{Separable} = C \cdot (K^2 + C) \cdot H \cdot W" />
        <p className="mt-2">Ratio:</p>
        <BlockMath math="\frac{\text{Separable}}{\text{Standard}} = \frac{1}{C} + \frac{1}{K^2}" />
        <p className="mt-2">For <InlineMath math="K = 3, C = 64" />: reduction to ~12% of the original cost.</p>
      </TheoremBlock>

      <CostComparison />

      <ExampleBlock title="MobileNet V1 Block">
        <p>Each MobileNet block consists of:</p>
        <p className="mt-1">1. <InlineMath math="3 \times 3" /> depthwise conv + BatchNorm + ReLU6</p>
        <p>2. <InlineMath math="1 \times 1" /> pointwise conv + BatchNorm + ReLU6</p>
        <p className="mt-2">MobileNetV2 adds inverted residuals with linear bottlenecks.</p>
      </ExampleBlock>

      <PythonCode
        title="Depthwise Separable Conv in PyTorch"
        code={`import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size,
                                    padding=padding, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

x = torch.randn(1, 64, 32, 32)
dw_sep = DepthwiseSeparableConv(64, 128)
standard = nn.Conv2d(64, 128, 3, padding=1)

print(f"Separable params: {sum(p.numel() for p in dw_sep.parameters()):,}")
print(f"Standard params:  {sum(p.numel() for p in standard.parameters()):,}")
print(f"Output: {dw_sep(x).shape}")  # [1, 128, 32, 32]`}
      />

      <NoteBlock type="note" title="Beyond MobileNet">
        <p>
          Depthwise separable convolutions are now standard in efficient architectures including
          <strong> EfficientNet</strong>, <strong>MnasNet</strong>, and on-device models. They achieve
          near-equivalent accuracy at a fraction of the cost, making them essential for mobile and
          edge deployment.
        </p>
      </NoteBlock>
    </div>
  )
}

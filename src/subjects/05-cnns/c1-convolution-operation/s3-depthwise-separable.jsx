import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function SavingsCalculator() {
  const [cIn, setCIn] = useState(64)
  const [cOut, setCOut] = useState(128)
  const [k, setK] = useState(3)

  const standardOps = cIn * cOut * k * k
  const depthwiseOps = cIn * k * k + cIn * cOut
  const ratio = (depthwiseOps / standardOps * 100).toFixed(1)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Computational Savings Calculator</h3>
      <div className="grid grid-cols-3 gap-3 mb-4">
        {[
          ['C_in', cIn, setCIn, 8, 512],
          ['C_out', cOut, setCOut, 8, 512],
          ['Kernel', k, setK, 1, 7],
        ].map(([label, val, setter, min, max]) => (
          <label key={label} className="text-sm text-gray-600 dark:text-gray-400">
            {label}: <strong>{val}</strong>
            <input type="range" min={min} max={max} value={val} onChange={e => setter(parseInt(e.target.value))} className="w-full accent-violet-500" />
          </label>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3 text-center">
        <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-800">
          <p className="text-xs text-gray-500">Standard Conv</p>
          <p className="text-lg font-bold text-gray-700 dark:text-gray-300">{standardOps.toLocaleString()}</p>
        </div>
        <div className="p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20">
          <p className="text-xs text-gray-500">Depthwise Separable</p>
          <p className="text-lg font-bold text-violet-700 dark:text-violet-300">{depthwiseOps.toLocaleString()}</p>
        </div>
      </div>
      <p className="text-center mt-2 text-sm text-gray-600 dark:text-gray-400">
        Only <strong className="text-violet-600">{ratio}%</strong> of the standard cost
      </p>
    </div>
  )
}

export default function DepthwiseSeparable() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Depthwise separable convolutions, popularized by MobileNet, factorize a standard convolution
        into two steps: a depthwise convolution that filters each channel independently, and a
        pointwise convolution that combines channels. This dramatically reduces computation.
      </p>

      <DefinitionBlock title="Depthwise Separable Convolution">
        <p>A standard convolution with kernel <InlineMath math="k \times k" />, <InlineMath math="C_{\text{in}}" /> input
          channels and <InlineMath math="C_{\text{out}}" /> output channels is factored into:</p>
        <p className="mt-2"><strong>1. Depthwise:</strong> One <InlineMath math="k \times k" /> filter per input channel</p>
        <p><strong>2. Pointwise:</strong> A <InlineMath math="1 \times 1" /> convolution mapping <InlineMath math="C_{\text{in}} \to C_{\text{out}}" /></p>
      </DefinitionBlock>

      <TheoremBlock title="Computational Reduction" id="dw-reduction">
        <p>Standard convolution cost per spatial position:</p>
        <BlockMath math="\text{Standard} = C_{\text{in}} \cdot C_{\text{out}} \cdot k^2" />
        <p>Depthwise separable cost:</p>
        <BlockMath math="\text{DW-Sep} = C_{\text{in}} \cdot k^2 + C_{\text{in}} \cdot C_{\text{out}}" />
        <p className="mt-2">Reduction ratio:</p>
        <BlockMath math="\frac{\text{DW-Sep}}{\text{Standard}} = \frac{1}{C_{\text{out}}} + \frac{1}{k^2}" />
      </TheoremBlock>

      <SavingsCalculator />

      <ExampleBlock title="MobileNet v1 Savings">
        <p>With <InlineMath math="k = 3" />, the depthwise separable conv uses roughly:</p>
        <BlockMath math="\frac{1}{C_{\text{out}}} + \frac{1}{9} \approx \frac{1}{9} \approx 11\%" />
        <p>of the computation, enabling real-time inference on mobile devices.</p>
      </ExampleBlock>

      <PythonCode
        title="Depthwise Separable Conv in PyTorch"
        code={`import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, padding=1):
        super().__init__()
        # Depthwise: groups=c_in means one filter per channel
        self.depthwise = nn.Conv2d(c_in, c_in, kernel_size,
                                    padding=padding, groups=c_in, bias=False)
        self.pointwise = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))

x = torch.randn(1, 64, 56, 56)
dw_sep = DepthwiseSeparableConv(64, 128)
standard = nn.Conv2d(64, 128, 3, padding=1)

print(f"DW-Sep params:  {sum(p.numel() for p in dw_sep.parameters()):,}")
print(f"Standard params: {sum(p.numel() for p in standard.parameters()):,}")
print(f"Output shape: {dw_sep(x).shape}")  # [1, 128, 56, 56]`}
      />

      <NoteBlock type="note" title="Evolution: MobileNet v1 to v3">
        <p>
          MobileNet v2 introduced inverted residuals with linear bottlenecks, expanding channels
          before the depthwise conv. MobileNet v3 added squeeze-and-excitation blocks and
          hardware-aware NAS for further efficiency improvements.
        </p>
      </NoteBlock>
    </div>
  )
}

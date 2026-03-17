import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function KernelSlidingDemo() {
  const [pos, setPos] = useState(0)
  const input = [1, 0, 2, 3, 1, 0, 1]
  const kernel = [1, 0, -1]
  const kSize = kernel.length
  const outSize = input.length - kSize + 1
  const maxPos = outSize - 1

  const computeOutput = (p) => {
    let sum = 0
    for (let k = 0; k < kSize; k++) sum += input[p + k] * kernel[k]
    return sum
  }

  const outputs = Array.from({ length: outSize }, (_, i) => computeOutput(i))
  const cellW = 44, cellH = 40

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Interactive Kernel Sliding Demo</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Position: {pos}
        <input type="range" min={0} max={maxPos} step={1} value={pos} onChange={e => setPos(parseInt(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <svg width={input.length * cellW + 20} height={160} className="mx-auto block">
        {input.map((v, i) => (
          <g key={`in-${i}`}>
            <rect x={10 + i * cellW} y={10} width={cellW} height={cellH} fill={i >= pos && i < pos + kSize ? '#ddd6fe' : '#f3f4f6'} stroke="#9ca3af" strokeWidth={1} rx={4} />
            <text x={10 + i * cellW + cellW / 2} y={35} textAnchor="middle" fontSize={14} fill="#374151">{v}</text>
          </g>
        ))}
        {kernel.map((v, i) => (
          <g key={`k-${i}`}>
            <rect x={10 + (pos + i) * cellW} y={60} width={cellW} height={cellH} fill="#c4b5fd" stroke="#7c3aed" strokeWidth={1.5} rx={4} />
            <text x={10 + (pos + i) * cellW + cellW / 2} y={85} textAnchor="middle" fontSize={14} fill="#4c1d95">{v}</text>
          </g>
        ))}
        {outputs.map((v, i) => (
          <g key={`o-${i}`}>
            <rect x={10 + (i + 1) * cellW} y={115} width={cellW} height={cellH} fill={i === pos ? '#a78bfa' : '#f3f4f6'} stroke="#9ca3af" strokeWidth={1} rx={4} />
            <text x={10 + (i + 1) * cellW + cellW / 2} y={140} textAnchor="middle" fontSize={14} fill={i === pos ? '#ffffff' : '#374151'}>{v}</text>
          </g>
        ))}
        <text x={5} y={35} fontSize={11} fill="#6b7280">Input</text>
        <text x={5} y={85} fontSize={11} fill="#6b7280">Kernel</text>
        <text x={5} y={140} fontSize={11} fill="#6b7280">Output</text>
      </svg>
      <p className="text-xs text-center text-gray-500 mt-2">
        {kernel.map((k, i) => `${input[pos + i]}*${k >= 0 ? k : `(${k})`}`).join(' + ')} = {computeOutput(pos)}
      </p>
    </div>
  )
}

export default function DiscreteConvolution() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Convolution is the fundamental operation in CNNs. It slides a small learnable kernel across the
        input, computing element-wise products and summing them to produce a feature map. In practice,
        deep learning frameworks implement cross-correlation rather than true convolution.
      </p>

      <DefinitionBlock title="1D Discrete Convolution">
        <BlockMath math="(f * g)[n] = \sum_{k} f[k] \, g[n - k]" />
        <p className="mt-2">
          In deep learning, we typically use <strong>cross-correlation</strong> (no kernel flip):
        </p>
        <BlockMath math="(f \star g)[n] = \sum_{k} f[n + k] \, g[k]" />
      </DefinitionBlock>

      <DefinitionBlock title="2D Convolution (Feature Map)">
        <BlockMath math="Y[i, j] = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} X[i+m,\, j+n] \cdot W[m, n] + b" />
        <p className="mt-2">
          Where <InlineMath math="W" /> is the <InlineMath math="k_h \times k_w" /> kernel,{' '}
          <InlineMath math="X" /> is the input, and <InlineMath math="b" /> is the bias.
        </p>
      </DefinitionBlock>

      <KernelSlidingDemo />

      <ExampleBlock title="Output Size Computation">
        <p>For an input of size <InlineMath math="n" /> with kernel size <InlineMath math="k" /> and no padding or stride:</p>
        <BlockMath math="n_{\text{out}} = n - k + 1" />
        <p>Example: input size 7, kernel size 3 gives output size <InlineMath math="7 - 3 + 1 = 5" />.</p>
      </ExampleBlock>

      <PythonCode
        title="2D Convolution in PyTorch"
        code={`import torch
import torch.nn as nn

# Input: batch=1, channels=1, height=5, width=5
x = torch.randn(1, 1, 5, 5)

# Conv2d: in_channels=1, out_channels=4, kernel_size=3
conv = nn.Conv2d(1, 4, kernel_size=3, padding=0)
y = conv(x)
print(f"Input shape:  {x.shape}")   # [1, 1, 5, 5]
print(f"Output shape: {y.shape}")   # [1, 4, 3, 3]
print(f"Parameters:   {sum(p.numel() for p in conv.parameters())}")  # 4*(1*3*3+1) = 40`}
      />

      <NoteBlock type="note" title="Convolution vs Cross-Correlation">
        <p>
          True convolution flips the kernel before sliding. Deep learning calls the operation
          "convolution" but implements cross-correlation. Since kernels are learned, the flip is
          absorbed into the weights, making the distinction irrelevant in practice.
        </p>
      </NoteBlock>
    </div>
  )
}

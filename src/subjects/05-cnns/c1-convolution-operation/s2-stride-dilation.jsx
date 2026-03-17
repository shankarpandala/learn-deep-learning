import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function OutputSizeCalculator() {
  const [inputSize, setInputSize] = useState(28)
  const [kernelSize, setKernelSize] = useState(3)
  const [stride, setStride] = useState(1)
  const [padding, setPadding] = useState(0)
  const [dilation, setDilation] = useState(1)

  const effectiveK = dilation * (kernelSize - 1) + 1
  const outputSize = Math.floor((inputSize + 2 * padding - effectiveK) / stride) + 1

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Output Size Calculator</h3>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-4">
        {[
          ['Input', inputSize, setInputSize, 1, 64],
          ['Kernel', kernelSize, setKernelSize, 1, 11],
          ['Stride', stride, setStride, 1, 5],
          ['Padding', padding, setPadding, 0, 10],
          ['Dilation', dilation, setDilation, 1, 5],
        ].map(([label, val, setter, min, max]) => (
          <label key={label} className="text-sm text-gray-600 dark:text-gray-400">
            {label}: <strong>{val}</strong>
            <input type="range" min={min} max={max} value={val} onChange={e => setter(parseInt(e.target.value))} className="w-full accent-violet-500" />
          </label>
        ))}
      </div>
      <div className="text-center p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20">
        <p className="text-sm text-gray-600 dark:text-gray-400">Effective kernel size: <strong>{effectiveK}</strong></p>
        <p className="text-lg font-bold text-violet-700 dark:text-violet-300">
          Output size: {outputSize > 0 ? outputSize : <span className="text-red-500">Invalid</span>}
        </p>
      </div>
    </div>
  )
}

export default function StrideDilation() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Stride and dilation are two key mechanisms that control how convolution kernels traverse
        the input. Stride controls the step size, while dilation inserts gaps between kernel elements
        to increase the receptive field without adding parameters.
      </p>

      <TheoremBlock title="General Output Size Formula" id="conv-output-size">
        <BlockMath math="n_{\text{out}} = \left\lfloor \frac{n_{\text{in}} + 2p - d(k - 1) - 1}{s} \right\rfloor + 1" />
        <p className="mt-2">
          Where <InlineMath math="p" /> is padding, <InlineMath math="s" /> is stride,{' '}
          <InlineMath math="d" /> is dilation, and <InlineMath math="k" /> is the kernel size.
        </p>
      </TheoremBlock>

      <OutputSizeCalculator />

      <DefinitionBlock title="Dilated (Atrous) Convolution">
        <p>
          Dilation inserts <InlineMath math="d - 1" /> zeros between kernel elements, giving an
          effective kernel size of <InlineMath math="d(k - 1) + 1" /> with only{' '}
          <InlineMath math="k^2" /> parameters. This exponentially increases receptive field size
          in stacked layers.
        </p>
        <BlockMath math="Y[i, j] = \sum_{m} \sum_{n} X[i + d \cdot m,\; j + d \cdot n] \cdot W[m, n]" />
      </DefinitionBlock>

      <DefinitionBlock title="Transposed Convolution">
        <p>
          Also called "deconvolution" or fractionally-strided convolution. It upsamples the feature
          map by inserting zeros between input elements and performing a standard convolution:
        </p>
        <BlockMath math="n_{\text{out}} = (n_{\text{in}} - 1) \cdot s - 2p + k" />
      </DefinitionBlock>

      <ExampleBlock title="Stride 2 Downsampling">
        <p>A <InlineMath math="3 \times 3" /> conv with stride 2 on a <InlineMath math="32 \times 32" /> input:</p>
        <BlockMath math="n_{\text{out}} = \left\lfloor \frac{32 + 2(1) - 3}{2} \right\rfloor + 1 = 16" />
        <p>This halves the spatial dimensions, acting as a learned downsampling alternative to pooling.</p>
      </ExampleBlock>

      <PythonCode
        title="Stride, Dilation & Transposed Conv in PyTorch"
        code={`import torch
import torch.nn as nn

x = torch.randn(1, 1, 32, 32)

# Strided convolution (downsampling)
conv_stride = nn.Conv2d(1, 16, 3, stride=2, padding=1)
print(f"Strided:    {x.shape} -> {conv_stride(x).shape}")  # [1,16,16,16]

# Dilated convolution (larger receptive field)
conv_dilated = nn.Conv2d(1, 16, 3, dilation=2, padding=2)
print(f"Dilated:    {x.shape} -> {conv_dilated(x).shape}")  # [1,16,32,32]

# Transposed convolution (upsampling)
x_small = torch.randn(1, 16, 16, 16)
conv_t = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)
print(f"Transposed: {x_small.shape} -> {conv_t(x_small).shape}")  # [1,1,32,32]`}
      />

      <NoteBlock type="note" title="Checkerboard Artifacts">
        <p>
          Transposed convolutions with stride {">"} 1 can produce checkerboard artifacts due to
          uneven overlap. A common fix is to use nearest-neighbor upsampling followed by a regular
          convolution instead.
        </p>
      </NoteBlock>
    </div>
  )
}

import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function OutputSizeCalculator() {
  const [inputSize, setInputSize] = useState(32)
  const [kernelSize, setKernelSize] = useState(3)
  const [stride, setStride] = useState(2)
  const [padding, setPadding] = useState(1)
  const [dilation, setDilation] = useState(1)

  const effectiveK = dilation * (kernelSize - 1) + 1
  const outSize = Math.floor((inputSize - effectiveK + 2 * padding) / stride) + 1

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Output Size Calculator</h3>
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-4">
        {[
          ['Input', inputSize, setInputSize, 1, 128],
          ['Kernel', kernelSize, setKernelSize, 1, 11],
          ['Stride', stride, setStride, 1, 5],
          ['Padding', padding, setPadding, 0, 10],
          ['Dilation', dilation, setDilation, 1, 5],
        ].map(([label, val, setter, min, max]) => (
          <label key={label} className="text-sm text-gray-600 dark:text-gray-400">
            {label}: <strong>{val}</strong>
            <input type="range" min={min} max={max} step={1} value={val} onChange={e => setter(parseInt(e.target.value))} className="w-full accent-violet-500" />
          </label>
        ))}
      </div>
      <p className="text-sm text-gray-700 dark:text-gray-300">
        Effective kernel: <strong className="text-violet-600 dark:text-violet-400">{effectiveK}</strong>
        {' | '}Output size: <strong className="text-violet-600 dark:text-violet-400">{outSize > 0 ? outSize : 'invalid'}</strong>
      </p>
    </div>
  )
}

export default function StrideDilation() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Stride and dilation give convolutions control over spatial resolution and receptive field
        without increasing parameter count.
      </p>

      <DefinitionBlock title="Strided Convolution">
        <p>A convolution with stride <InlineMath math="s > 1" /> skips positions, reducing spatial dimensions:</p>
        <BlockMath math="W_{out} = \left\lfloor \frac{W_{in} - K + 2P}{s} \right\rfloor + 1" />
        <p className="mt-2">Stride-2 convolutions are commonly used instead of pooling for downsampling.</p>
      </DefinitionBlock>

      <DefinitionBlock title="Dilated (Atrous) Convolution">
        <p>Dilation rate <InlineMath math="d" /> inserts <InlineMath math="d - 1" /> zeros between kernel elements:</p>
        <BlockMath math="K_{eff} = d \cdot (K - 1) + 1" />
        <p className="mt-2">This expands the receptive field exponentially without increasing parameters or reducing resolution.</p>
      </DefinitionBlock>

      <OutputSizeCalculator />

      <TheoremBlock title="Transposed Convolution Output Size" id="transposed-conv">
        <p>Transposed (deconvolution) upsamples feature maps:</p>
        <BlockMath math="W_{out} = (W_{in} - 1) \cdot s - 2P + K + \text{output\_padding}" />
        <p className="mt-2">
          Used in decoder networks, generators (GANs), and segmentation for spatial upsampling.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Stride vs Dilation Comparison">
        <p>Input <InlineMath math="8 \times 8" /> with <InlineMath math="3 \times 3" /> kernel:</p>
        <p className="mt-1">Stride 2, dilation 1: output <InlineMath math="3 \times 3" /> (downsampled)</p>
        <p>Stride 1, dilation 2: output <InlineMath math="4 \times 4" /> (larger receptive field, same params)</p>
      </ExampleBlock>

      <PythonCode
        title="Strided, Dilated & Transposed Convolutions"
        code={`import torch
import torch.nn as nn

x = torch.randn(1, 1, 8, 8)

# Strided convolution (downsampling)
conv_stride = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
print(f"Stride 2: {x.shape} -> {conv_stride(x).shape}")  # [1,1,4,4]

# Dilated convolution (larger receptive field)
conv_dilated = nn.Conv2d(1, 1, kernel_size=3, dilation=2, padding=2)
print(f"Dilation 2: {x.shape} -> {conv_dilated(x).shape}")  # [1,1,8,8]

# Transposed convolution (upsampling)
x_small = torch.randn(1, 1, 4, 4)
conv_t = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
print(f"Transposed: {x_small.shape} -> {conv_t(x_small).shape}")  # [1,1,8,8]`}
      />

      <NoteBlock type="note" title="Checkerboard Artifacts">
        <p>
          Transposed convolutions with non-divisible kernel/stride combinations produce
          <strong> checkerboard artifacts</strong>. The recommended alternative is nearest-neighbor
          or bilinear upsampling followed by a regular convolution.
        </p>
      </NoteBlock>
    </div>
  )
}

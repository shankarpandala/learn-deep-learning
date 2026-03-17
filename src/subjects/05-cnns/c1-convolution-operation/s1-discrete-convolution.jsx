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
  const outLen = input.length - kSize + 1
  const clampedPos = Math.min(pos, outLen - 1)

  const outputVal = input.slice(clampedPos, clampedPos + kSize).reduce((s, v, i) => s + v * kernel[i], 0)
  const outputs = Array.from({ length: outLen }, (_, p) =>
    input.slice(p, p + kSize).reduce((s, v, i) => s + v * kernel[i], 0)
  )

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">1D Kernel Sliding Demo</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Position: {clampedPos}
        <input type="range" min={0} max={outLen - 1} step={1} value={clampedPos} onChange={e => setPos(parseInt(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <div className="flex gap-1 mb-2">
        {input.map((v, i) => (
          <div key={i} className={`w-10 h-10 flex items-center justify-center rounded text-sm font-mono border ${i >= clampedPos && i < clampedPos + kSize ? 'bg-violet-100 border-violet-400 dark:bg-violet-900/40 dark:border-violet-500' : 'bg-gray-50 border-gray-300 dark:bg-gray-800 dark:border-gray-600'}`}>
            {v}
          </div>
        ))}
        <span className="ml-2 text-xs text-gray-500 self-center">input</span>
      </div>
      <div className="flex gap-1 mb-2" style={{ marginLeft: clampedPos * 44 }}>
        {kernel.map((v, i) => (
          <div key={i} className="w-10 h-10 flex items-center justify-center rounded text-sm font-mono bg-violet-200 border border-violet-500 dark:bg-violet-800/50 dark:border-violet-400">
            {v}
          </div>
        ))}
        <span className="ml-2 text-xs text-gray-500 self-center">kernel</span>
      </div>
      <p className="text-sm text-gray-700 dark:text-gray-300 mt-3">
        Output at position {clampedPos}: <strong className="text-violet-600 dark:text-violet-400">{outputVal}</strong>
        <span className="ml-4">Full output: [{outputs.join(', ')}]</span>
      </p>
    </div>
  )
}

export default function DiscreteConvolution() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Convolution is the core operation in CNNs. It applies a learned kernel (filter) across
        the input to produce a feature map, detecting local patterns such as edges, textures, and shapes.
      </p>

      <DefinitionBlock title="1D Discrete Convolution">
        <BlockMath math="(f * g)[n] = \sum_{k} f[k] \, g[n - k]" />
        <p className="mt-2">
          In practice, deep learning frameworks compute <strong>cross-correlation</strong> (no kernel flip):
        </p>
        <BlockMath math="(f \star g)[n] = \sum_{k} f[k] \, g[n + k]" />
      </DefinitionBlock>

      <DefinitionBlock title="2D Convolution">
        <BlockMath math="(I * K)[i, j] = \sum_{m}\sum_{n} K[m, n] \cdot I[i + m, j + n]" />
        <p className="mt-2">
          A kernel of size <InlineMath math="k \times k" /> slides over the 2D input. Each position
          produces one value in the output feature map.
        </p>
      </DefinitionBlock>

      <KernelSlidingDemo />

      <ExampleBlock title="Output Size Calculation">
        <p>For input size <InlineMath math="W" />, kernel size <InlineMath math="K" />, padding <InlineMath math="P" />, and stride <InlineMath math="S" />:</p>
        <BlockMath math="W_{out} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1" />
        <p className="mt-2">Example: input 32, kernel 5, padding 0, stride 1 gives <InlineMath math="\lfloor(32 - 5)/1\rfloor + 1 = 28" /></p>
      </ExampleBlock>

      <PythonCode
        title="2D Convolution in PyTorch"
        code={`import torch
import torch.nn as nn

# Single channel 5x5 input, 3x3 kernel
x = torch.randn(1, 1, 5, 5)
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=0)

out = conv(x)
print(f"Input shape:  {x.shape}")      # [1, 1, 5, 5]
print(f"Output shape: {out.shape}")     # [1, 1, 3, 3]
print(f"Kernel shape: {conv.weight.shape}")  # [1, 1, 3, 3]

# Cross-correlation vs convolution
# PyTorch uses cross-correlation by default (no kernel flip)
kernel = conv.weight.data[0, 0]
print(f"Kernel values:\\n{kernel}")`}
      />

      <NoteBlock type="note" title="Convolution vs Cross-Correlation">
        <p>
          True convolution flips the kernel before sliding. In deep learning, because kernels are
          <strong> learned</strong>, flipping is unnecessary — the network simply learns the flipped
          version. All major frameworks (PyTorch, TensorFlow) implement cross-correlation but call
          it "convolution" by convention.
        </p>
      </NoteBlock>
    </div>
  )
}

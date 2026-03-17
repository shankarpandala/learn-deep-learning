import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ReceptiveFieldCalculator() {
  const [numLayers, setNumLayers] = useState(3)
  const [kernelSize, setKernelSize] = useState(3)
  const [stride, setStride] = useState(1)
  const [dilation, setDilation] = useState(1)

  const computeRF = () => {
    let rf = 1, jump = 1
    for (let i = 0; i < numLayers; i++) {
      const effectiveK = dilation * (kernelSize - 1) + 1
      rf = rf + (effectiveK - 1) * jump
      jump = jump * stride
    }
    return rf
  }

  const rf = computeRF()
  const gridSize = Math.min(rf + 4, 25)
  const center = Math.floor(gridSize / 2)
  const halfRf = Math.floor(rf / 2)
  const cellSize = Math.max(12, Math.min(22, 300 / gridSize))

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Receptive Field Calculator</h3>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        {[
          ['Layers', numLayers, setNumLayers, 1, 8],
          ['Kernel', kernelSize, setKernelSize, 2, 7],
          ['Stride', stride, setStride, 1, 3],
          ['Dilation', dilation, setDilation, 1, 4],
        ].map(([label, val, setter, min, max]) => (
          <label key={label} className="text-sm text-gray-600 dark:text-gray-400">
            {label}: <strong>{val}</strong>
            <input type="range" min={min} max={max} value={val} onChange={e => setter(parseInt(e.target.value))} className="w-full accent-violet-500" />
          </label>
        ))}
      </div>
      <div className="flex items-center gap-6 justify-center">
        <svg width={gridSize * cellSize + 4} height={gridSize * cellSize + 4} className="block">
          {Array.from({ length: gridSize }, (_, r) =>
            Array.from({ length: gridSize }, (_, c) => {
              const inRf = Math.abs(r - center) <= halfRf && Math.abs(c - center) <= halfRf
              return (
                <rect key={`${r}-${c}`} x={2 + c * cellSize} y={2 + r * cellSize} width={cellSize - 1} height={cellSize - 1}
                  fill={r === center && c === center ? '#7c3aed' : inRf ? '#ddd6fe' : '#f3f4f6'} stroke="#d1d5db" strokeWidth={0.5} rx={1} />
              )
            })
          )}
        </svg>
        <div className="text-center">
          <p className="text-3xl font-bold text-violet-700 dark:text-violet-300">{rf} x {rf}</p>
          <p className="text-sm text-gray-500 mt-1">Receptive field size</p>
        </div>
      </div>
    </div>
  )
}

export default function ReceptiveField() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The receptive field of a neuron in a CNN is the region of the input image that influences
        that neuron's output. Understanding receptive fields is crucial for designing architectures
        that can capture features at the appropriate spatial scale.
      </p>

      <DefinitionBlock title="Receptive Field">
        <p>
          The <strong>theoretical receptive field</strong> of a unit at layer <InlineMath math="l" /> is
          the set of input pixels that can affect its value. In practice, the{' '}
          <strong>effective receptive field</strong> is much smaller, concentrated in a Gaussian-like
          pattern at the center.
        </p>
      </DefinitionBlock>

      <TheoremBlock title="Receptive Field Recurrence" id="rf-recurrence">
        <p>For a stack of convolutional layers, the receptive field grows as:</p>
        <BlockMath math="r_l = r_{l-1} + (k_l - 1) \cdot j_{l-1} \cdot d_l" />
        <BlockMath math="j_l = j_{l-1} \cdot s_l" />
        <p className="mt-2">
          Where <InlineMath math="r_l" /> is the RF at layer <InlineMath math="l" />,{' '}
          <InlineMath math="j_l" /> is the jump (product of strides), <InlineMath math="k_l" /> is kernel size,{' '}
          <InlineMath math="s_l" /> is stride, and <InlineMath math="d_l" /> is dilation. Base case: <InlineMath math="r_0 = 1, j_0 = 1" />.
        </p>
      </TheoremBlock>

      <ReceptiveFieldCalculator />

      <ExampleBlock title="VGG-16 Receptive Field">
        <p>VGG-16 uses thirteen <InlineMath math="3 \times 3" /> conv layers with five <InlineMath math="2 \times 2" /> max-pooling layers (stride 2). The final conv layer has a receptive field of:</p>
        <BlockMath math="r = 212 \times 212 \text{ pixels}" />
        <p>This covers most of the <InlineMath math="224 \times 224" /> input, enabling the network to capture global context.</p>
      </ExampleBlock>

      <PythonCode
        title="Computing Receptive Field Programmatically"
        code={`def compute_receptive_field(layers):
    """Compute RF for a list of (kernel_size, stride, dilation) tuples."""
    rf, jump = 1, 1
    for k, s, d in layers:
        effective_k = d * (k - 1) + 1
        rf = rf + (effective_k - 1) * jump
        jump = jump * s
    return rf

# VGG-style: 3x3 convs with 2x2 max-pooling
vgg_block = [(3, 1, 1)] * 2 + [(2, 2, 1)]  # 2 convs + pool
vgg_layers = vgg_block * 2 + [(3, 1, 1)] * 3 + [(2, 2, 1)]  # repeat pattern
vgg_layers += [(3, 1, 1)] * 3 + [(2, 2, 1)] + [(3, 1, 1)] * 3 + [(2, 2, 1)]
print(f"VGG-like RF: {compute_receptive_field(vgg_layers)}")

# Dilated stack: exponentially growing RF
dilated = [(3, 1, 2**i) for i in range(5)]
print(f"Dilated RF (5 layers): {compute_receptive_field(dilated)}")`}
      />

      <NoteBlock type="note" title="Effective vs Theoretical Receptive Field">
        <p>
          Research by Luo et al. (2016) showed that the effective receptive field is significantly
          smaller than the theoretical one, following a Gaussian distribution. Techniques like
          dilated convolutions and attention mechanisms help increase the effective receptive field.
        </p>
      </NoteBlock>
    </div>
  )
}

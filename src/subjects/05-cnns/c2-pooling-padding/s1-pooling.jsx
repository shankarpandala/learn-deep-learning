import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function PoolingDemo() {
  const [mode, setMode] = useState('max')
  const input = [
    [1, 3, 2, 4],
    [5, 6, 1, 2],
    [3, 2, 7, 8],
    [4, 1, 5, 3],
  ]
  const [highlight, setHighlight] = useState([0, 0])

  const pooled = (r, c) => {
    const block = [input[r * 2][c * 2], input[r * 2][c * 2 + 1], input[r * 2 + 1][c * 2], input[r * 2 + 1][c * 2 + 1]]
    return mode === 'max' ? Math.max(...block) : (block.reduce((a, b) => a + b, 0) / 4).toFixed(1)
  }

  const cellW = 48, cellH = 40

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Interactive Pooling Demo</h3>
      <div className="flex items-center gap-4 mb-3">
        {['max', 'avg'].map(m => (
          <label key={m} className="flex items-center gap-1.5 text-sm text-gray-600 dark:text-gray-400">
            <input type="radio" name="pool" checked={mode === m} onChange={() => setMode(m)} className="accent-violet-500" />
            {m === 'max' ? 'Max' : 'Average'} Pooling
          </label>
        ))}
      </div>
      <div className="flex items-center gap-8 justify-center">
        <svg width={4 * cellW + 10} height={4 * cellH + 10}>
          {input.map((row, r) => row.map((v, c) => {
            const isActive = Math.floor(r / 2) === highlight[0] && Math.floor(c / 2) === highlight[1]
            return (
              <g key={`${r}-${c}`}>
                <rect x={5 + c * cellW} y={5 + r * cellH} width={cellW} height={cellH} fill={isActive ? '#ddd6fe' : '#f9fafb'} stroke="#9ca3af" strokeWidth={1} rx={3} />
                <text x={5 + c * cellW + cellW / 2} y={5 + r * cellH + cellH / 2 + 5} textAnchor="middle" fontSize={14} fill="#374151">{v}</text>
              </g>
            )
          }))}
        </svg>
        <span className="text-2xl text-gray-400">&rarr;</span>
        <svg width={2 * cellW + 10} height={2 * cellH + 10}>
          {[0, 1].map(r => [0, 1].map(c => (
            <g key={`o-${r}-${c}`} onMouseEnter={() => setHighlight([r, c])} style={{ cursor: 'pointer' }}>
              <rect x={5 + c * cellW} y={5 + r * cellH} width={cellW} height={cellH} fill={r === highlight[0] && c === highlight[1] ? '#a78bfa' : '#ede9fe'} stroke="#7c3aed" strokeWidth={1.5} rx={3} />
              <text x={5 + c * cellW + cellW / 2} y={5 + r * cellH + cellH / 2 + 5} textAnchor="middle" fontSize={14} fill="#4c1d95">{pooled(r, c)}</text>
            </g>
          )))}
        </svg>
      </div>
      <p className="text-xs text-center text-gray-500 mt-2">Hover over output cells to see which input region they correspond to.</p>
    </div>
  )
}

export default function Pooling() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Pooling layers reduce spatial dimensions of feature maps, providing translation invariance
        and reducing computational cost. Max and average pooling are the two most common variants.
      </p>

      <DefinitionBlock title="Max Pooling">
        <BlockMath math="Y[i, j] = \max_{(m,n) \in \mathcal{R}_{ij}} X[m, n]" />
        <p className="mt-2">
          Selects the maximum value in each <InlineMath math="k \times k" /> pooling window.
          This preserves the strongest activations and provides robustness to small translations.
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="Average Pooling">
        <BlockMath math="Y[i, j] = \frac{1}{k^2} \sum_{(m,n) \in \mathcal{R}_{ij}} X[m, n]" />
        <p className="mt-2">
          Computes the mean of all values in the pooling window. Smoother than max pooling but
          may lose fine-grained spatial detail.
        </p>
      </DefinitionBlock>

      <PoolingDemo />

      <ExampleBlock title="Spatial Downsampling">
        <p>A <InlineMath math="2 \times 2" /> pooling with stride 2 on a <InlineMath math="224 \times 224" /> feature map:</p>
        <BlockMath math="224 \times 224 \xrightarrow{\text{pool}} 112 \times 112" />
        <p>Each pooling operation halves spatial dimensions and reduces computation by 4x in subsequent layers.</p>
      </ExampleBlock>

      <PythonCode
        title="Pooling Layers in PyTorch"
        code={`import torch
import torch.nn as nn

x = torch.randn(1, 64, 32, 32)

# Max pooling 2x2 with stride 2
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
print(f"MaxPool: {x.shape} -> {max_pool(x).shape}")  # [1, 64, 16, 16]

# Average pooling 2x2 with stride 2
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
print(f"AvgPool: {x.shape} -> {avg_pool(x).shape}")  # [1, 64, 16, 16]

# Max pooling with return_indices (useful for unpooling)
pool_idx = nn.MaxPool2d(2, stride=2, return_indices=True)
output, indices = pool_idx(x)
print(f"Indices shape: {indices.shape}")  # [1, 64, 16, 16]`}
      />

      <NoteBlock type="note" title="Pooling vs Strided Convolutions">
        <p>
          Modern architectures like ResNet and ConvNeXt often replace pooling with strided
          convolutions. Strided convolutions are learnable and can preserve more information,
          though they add parameters. Max pooling remains popular for its simplicity and
          translation invariance properties.
        </p>
      </NoteBlock>
    </div>
  )
}

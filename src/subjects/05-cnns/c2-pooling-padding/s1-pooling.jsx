import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function PoolingDemo() {
  const [poolType, setPoolType] = useState('max')
  const grid = [
    [1, 3, 2, 4],
    [5, 6, 1, 2],
    [7, 2, 3, 8],
    [0, 4, 5, 1],
  ]
  const [highlight, setHighlight] = useState(0)

  const regions = [
    { r: 0, c: 0 }, { r: 0, c: 2 }, { r: 2, c: 0 }, { r: 2, c: 2 },
  ]
  const region = regions[highlight]
  const vals = [grid[region.r][region.c], grid[region.r][region.c + 1], grid[region.r + 1][region.c], grid[region.r + 1][region.c + 1]]
  const pooled = regions.map(rg => {
    const v = [grid[rg.r][rg.c], grid[rg.r][rg.c + 1], grid[rg.r + 1][rg.c], grid[rg.r + 1][rg.c + 1]]
    return poolType === 'max' ? Math.max(...v) : (v.reduce((a, b) => a + b) / 4).toFixed(1)
  })

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">2x2 Pooling Demo</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          <select value={poolType} onChange={e => setPoolType(e.target.value)} className="rounded border px-2 py-1 text-sm dark:bg-gray-800 dark:border-gray-600">
            <option value="max">Max Pooling</option>
            <option value="avg">Average Pooling</option>
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Region: {highlight}
          <input type="range" min={0} max={3} step={1} value={highlight} onChange={e => setHighlight(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <div className="flex items-center gap-6">
        <div className="grid grid-cols-4 gap-1">
          {grid.flat().map((v, i) => {
            const r = Math.floor(i / 4), c = i % 4
            const active = r >= region.r && r < region.r + 2 && c >= region.c && c < region.c + 2
            return (
              <div key={i} className={`w-10 h-10 flex items-center justify-center rounded text-sm font-mono border ${active ? 'bg-violet-100 border-violet-400 dark:bg-violet-900/40 dark:border-violet-500' : 'bg-gray-50 border-gray-300 dark:bg-gray-800 dark:border-gray-600'}`}>
                {v}
              </div>
            )
          })}
        </div>
        <span className="text-gray-400 text-xl">&rarr;</span>
        <div className="grid grid-cols-2 gap-1">
          {pooled.map((v, i) => (
            <div key={i} className={`w-10 h-10 flex items-center justify-center rounded text-sm font-mono border ${i === highlight ? 'bg-violet-200 border-violet-500 dark:bg-violet-800/50 dark:border-violet-400' : 'bg-gray-50 border-gray-300 dark:bg-gray-800 dark:border-gray-600'}`}>
              {v}
            </div>
          ))}
        </div>
      </div>
      <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
        Region values: [{vals.join(', ')}] &rarr; {poolType === 'max' ? 'max' : 'avg'} = <strong className="text-violet-600 dark:text-violet-400">{pooled[highlight]}</strong>
      </p>
    </div>
  )
}

export default function Pooling() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Pooling layers reduce spatial dimensions, decrease computation, and provide a degree of
        translation invariance by summarizing local regions.
      </p>

      <DefinitionBlock title="Max Pooling">
        <BlockMath math="y_{i,j} = \max_{(m,n) \in R_{i,j}} x_{m,n}" />
        <p className="mt-2">Selects the maximum value in each pooling region. Preserves dominant features.</p>
      </DefinitionBlock>

      <DefinitionBlock title="Average Pooling">
        <BlockMath math="y_{i,j} = \frac{1}{|R|}\sum_{(m,n) \in R_{i,j}} x_{m,n}" />
        <p className="mt-2">Computes the mean of the pooling region. Smoother but may dilute strong activations.</p>
      </DefinitionBlock>

      <PoolingDemo />

      <ExampleBlock title="Translation Invariance">
        <p>
          If an edge feature shifts by 1 pixel within a <InlineMath math="2 \times 2" /> pooling window,
          the max-pooled output remains the same, providing robustness to small spatial translations.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Pooling Layers in PyTorch"
        code={`import torch
import torch.nn as nn

x = torch.randn(1, 64, 32, 32)

# Max pooling: 2x2 window, stride 2
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
print(f"Max pool: {x.shape} -> {max_pool(x).shape}")  # [1,64,16,16]

# Average pooling: 2x2 window, stride 2
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
print(f"Avg pool: {x.shape} -> {avg_pool(x).shape}")  # [1,64,16,16]

# Adaptive pooling: specify output size, not kernel size
adaptive = nn.AdaptiveAvgPool2d((1, 1))
print(f"Adaptive: {x.shape} -> {adaptive(x).shape}")  # [1,64,1,1]`}
      />

      <WarningBlock title="Information Loss">
        <p>
          Pooling discards spatial information irreversibly. In tasks requiring precise localization
          (segmentation, detection), aggressive pooling can harm performance. Modern architectures
          often use strided convolutions as a learnable alternative.
        </p>
      </WarningBlock>
    </div>
  )
}

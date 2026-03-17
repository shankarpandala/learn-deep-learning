import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function NormComparisonViz() {
  const [mode, setMode] = useState('layer')
  const B = 4, C = 6
  const data = Array.from({ length: B * C }, (_, i) => 0.3 * Math.sin(i * 0.7) + 0.5 * Math.cos(i * 0.3))

  const cellW = 44, cellH = 32, pad = 2

  const getGroup = (b, c) => {
    if (mode === 'batch') return `batch-${c}`
    if (mode === 'layer') return `layer-${b}`
    return `instance-${b}-${c}`
  }

  const colors = {
    batch: ['#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe', '#ede9fe', '#f5f3ff'],
    layer: ['#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe'],
    instance: '#8b5cf6',
  }

  const getColor = (b, c) => {
    if (mode === 'batch') return colors.batch[c % colors.batch.length]
    if (mode === 'layer') return colors.layer[b % colors.layer.length]
    return '#8b5cf6'
  }

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-2 text-base font-bold text-gray-800 dark:text-gray-200">Normalization Dimensions</h3>
      <div className="flex gap-2 mb-3">
        {['batch', 'layer', 'instance'].map(m => (
          <button key={m} onClick={() => setMode(m)}
            className={`px-3 py-1 rounded text-xs font-medium ${mode === m ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'}`}>
            {m === 'batch' ? 'BatchNorm' : m === 'layer' ? 'LayerNorm' : 'InstanceNorm'}
          </button>
        ))}
      </div>
      <div className="flex justify-center">
        <svg width={(C + 1) * (cellW + pad) + 40} height={(B + 1) * (cellH + pad) + 10}>
          <text x={0} y={15} fill="#6b7280" fontSize={10}>B \ C</text>
          {Array.from({ length: C }, (_, c) => (
            <text key={`h-${c}`} x={40 + c * (cellW + pad) + cellW / 2} y={15} textAnchor="middle" fill="#6b7280" fontSize={9}>c{c}</text>
          ))}
          {Array.from({ length: B }, (_, b) => (
            <g key={`row-${b}`}>
              <text x={15} y={30 + b * (cellH + pad) + cellH / 2 + 4} textAnchor="middle" fill="#6b7280" fontSize={9}>b{b}</text>
              {Array.from({ length: C }, (_, c) => (
                <rect key={`cell-${b}-${c}`} x={40 + c * (cellW + pad)} y={22 + b * (cellH + pad)} width={cellW} height={cellH} rx={4} fill={getColor(b, c)} opacity={0.5} stroke={getColor(b, c)} strokeWidth={1.5} />
              ))}
            </g>
          ))}
        </svg>
      </div>
      <p className="text-center text-xs text-gray-500 mt-2">
        {mode === 'batch' && 'Same color = normalized together (across batch, per channel)'}
        {mode === 'layer' && 'Same color = normalized together (across channels, per instance)'}
        {mode === 'instance' && 'Each cell normalized independently (per instance, per channel)'}
      </p>
    </div>
  )
}

export default function LayerNorm() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Layer Normalization (Ba et al., 2016) normalizes across features within each instance,
        making it independent of batch size. It is the standard normalization for Transformers.
      </p>

      <DefinitionBlock title="Layer Normalization">
        <BlockMath math="\mu = \frac{1}{H}\sum_{i=1}^{H} x_i, \quad \sigma^2 = \frac{1}{H}\sum_{i=1}^{H}(x_i - \mu)^2" />
        <BlockMath math="y_i = \gamma \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta" />
        <p className="mt-2">
          Statistics are computed over the feature dimension <InlineMath math="H" /> independently
          for each sample. No dependence on other samples in the batch.
        </p>
      </DefinitionBlock>

      <NormComparisonViz />

      <TheoremBlock title="LayerNorm vs BatchNorm" id="ln-vs-bn">
        <p>Key differences that make LayerNorm preferred for sequence models:</p>
        <BlockMath math="\text{BN: normalize over } (B, H, W) \quad \text{LN: normalize over } (C)" />
        <p>
          LayerNorm computes identical results regardless of batch size (even 1). It handles
          variable-length sequences naturally and behaves identically at train and eval time.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Pre-Norm vs Post-Norm Transformers">
        <p>
          <strong>Post-Norm</strong> (original): <InlineMath math="x + \text{LN}(\text{Attn}(x))" />.
          <strong> Pre-Norm</strong> (GPT-2+): <InlineMath math="x + \text{Attn}(\text{LN}(x))" />.
          Pre-Norm is more stable for training deep Transformers because gradients flow through
          the residual path without being normalized.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Layer Normalization in PyTorch"
        code={`import torch
import torch.nn as nn

# LayerNorm for a Transformer with d_model=512
ln = nn.LayerNorm(512)

# Works the same regardless of batch size
x1 = torch.randn(1, 10, 512)   # batch=1, seq=10
x32 = torch.randn(32, 10, 512) # batch=32, seq=10

# Identical normalization per-instance
out1 = ln(x1)
out32 = ln(x32)

# Pre-norm Transformer block pattern
class PreNormBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ffn(self.ln2(x))
        return x

block = PreNormBlock(512, 8)
print(f"Output: {block(x32).shape}")`}
      />

      <NoteBlock type="note" title="When to Use Which Norm">
        <p>
          <strong>BatchNorm</strong>: CNNs with batch size &ge; 16. <strong>LayerNorm</strong>:
          Transformers, RNNs, any model needing batch-independent normalization.
          <strong> GroupNorm</strong>: CNNs with small batch sizes. The trend in modern
          architectures is toward LayerNorm and its variants (RMSNorm).
        </p>
      </NoteBlock>
    </div>
  )
}

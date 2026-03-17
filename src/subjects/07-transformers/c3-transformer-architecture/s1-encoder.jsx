import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

function EncoderBlockDiagram() {
  const [showResidual, setShowResidual] = useState(true)
  const boxH = 40
  const layers = [
    { label: 'Input Embeddings + Positional Encoding', color: '#ddd6fe' },
    { label: 'Multi-Head Self-Attention', color: '#c4b5fd' },
    { label: 'Add & Layer Norm', color: '#a78bfa', isResidual: true },
    { label: 'Feed-Forward Network (FFN)', color: '#c4b5fd' },
    { label: 'Add & Layer Norm', color: '#a78bfa', isResidual: true },
    { label: 'Encoder Output', color: '#ddd6fe' },
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Transformer Encoder Block</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        <input type="checkbox" checked={showResidual} onChange={e => setShowResidual(e.target.checked)} className="accent-violet-500" />
        Show residual connections
      </label>
      <svg width={320} height={layers.length * (boxH + 16) + 8} className="mx-auto block">
        {layers.map((l, i) => {
          const y = i * (boxH + 16) + 4
          return (
            <g key={i}>
              <rect x={40} y={y} width={240} height={boxH} rx={8} fill={l.color} stroke="#7c3aed" strokeWidth={1} />
              <text x={160} y={y + boxH / 2 + 5} textAnchor="middle" fontSize={11} fill="#3b0764">{l.label}</text>
              {showResidual && l.isResidual && (
                <path d={`M 35 ${y - boxH - 12} C 15 ${y - boxH - 12}, 15 ${y + boxH / 2}, 35 ${y + boxH / 2}`} fill="none" stroke="#8b5cf6" strokeWidth={1.5} strokeDasharray="4,3" markerEnd="url(#arrowhead)" />
              )}
            </g>
          )
        })}
        <defs>
          <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
            <polygon points="0 0, 6 2, 0 4" fill="#8b5cf6" />
          </marker>
        </defs>
      </svg>
    </div>
  )
}

export default function TransformerEncoder() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The transformer encoder processes the entire input sequence in parallel through a stack
        of identical blocks. Each block contains multi-head self-attention and a position-wise
        feed-forward network, with residual connections and layer normalization stabilizing training.
      </p>

      <DefinitionBlock title="Encoder Block">
        <p>Each encoder block applies two sub-layers with residual connections:</p>
        <BlockMath math="h = \text{LayerNorm}(x + \text{MultiHeadAttn}(x, x, x))" />
        <BlockMath math="\text{out} = \text{LayerNorm}(h + \text{FFN}(h))" />
        <p className="mt-2">
          The FFN is a two-layer MLP: <InlineMath math="\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2" /> with
          inner dimension typically <InlineMath math="4 \cdot d_{\text{model}}" />.
        </p>
      </DefinitionBlock>

      <EncoderBlockDiagram />

      <TheoremBlock title="Why Residual Connections Matter" id="residual-encoder">
        <p>Residual connections ensure the gradient flows directly through the network:</p>
        <BlockMath math="\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \prod_{i=l}^{L-1}\left(1 + \frac{\partial F_i}{\partial x_i}\right)" />
        <p className="mt-2">
          The additive &quot;1&quot; term prevents gradient vanishing even in very deep stacks
          (BERT-large uses 24 encoder layers).
        </p>
      </TheoremBlock>

      <ExampleBlock title="Layer Normalization vs Batch Normalization">
        <p>
          Transformers use <strong>Layer Norm</strong> (normalize across features for each token)
          rather than Batch Norm (normalize across the batch). Layer Norm is invariant to batch size
          and works naturally with variable-length sequences:
        </p>
        <BlockMath math="\text{LN}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta, \quad \mu = \frac{1}{d}\sum_i x_i" />
      </ExampleBlock>

      <PythonCode
        title="Transformer Encoder Block in PyTorch"
        code={`import torch
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.drop1(attn_out))
        ff_out = self.ffn(x)
        x = self.norm2(x + self.drop2(ff_out))
        return x

block = TransformerEncoderBlock(d_model=512, num_heads=8, d_ff=2048)
x = torch.randn(4, 20, 512)  # batch=4, seq=20
out = block(x)
print(f"Output: {out.shape}")  # (4, 20, 512)`}
      />

      <NoteBlock type="note" title="Pre-Norm vs Post-Norm">
        <p>
          The original Transformer uses <strong>Post-Norm</strong> (normalize after residual addition).
          Many modern models use <strong>Pre-Norm</strong> (normalize before the sub-layer), which
          is more stable for training deep models but may have slightly lower performance at convergence.
        </p>
      </NoteBlock>
    </div>
  )
}

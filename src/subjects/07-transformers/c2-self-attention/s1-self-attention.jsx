import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

function SelfAttentionDemo() {
  const [queryIdx, setQueryIdx] = useState(1)
  const tokens = ['The', 'bank', 'of', 'the', 'river']
  const scores = [
    [0.35, 0.15, 0.10, 0.10, 0.30],
    [0.08, 0.30, 0.07, 0.05, 0.50],
    [0.20, 0.25, 0.15, 0.20, 0.20],
    [0.30, 0.10, 0.15, 0.25, 0.20],
    [0.10, 0.45, 0.10, 0.05, 0.30],
  ]

  const row = scores[queryIdx]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Self-Attention: Which tokens does each word attend to?</h3>
      <div className="flex gap-2 mb-4 mt-3">
        {tokens.map((t, i) => (
          <button key={i} onClick={() => setQueryIdx(i)} className={`px-3 py-1 rounded-lg text-sm font-medium transition ${queryIdx === i ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {t}
          </button>
        ))}
      </div>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
        Query: <strong className="text-violet-600 dark:text-violet-400">{tokens[queryIdx]}</strong> attends to:
      </p>
      <div className="flex gap-3 justify-center">
        {tokens.map((t, i) => (
          <div key={i} className="flex flex-col items-center gap-1">
            <div className="w-14 rounded" style={{ height: `${Math.max(4, row[i] * 100)}px`, backgroundColor: `rgba(139, 92, 246, ${0.2 + row[i] * 0.8})` }} />
            <span className="text-xs font-mono text-gray-700 dark:text-gray-300">{t}</span>
            <span className="text-xs text-violet-600 dark:text-violet-400">{row[i].toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function SelfAttention() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Self-attention lets every position in a sequence attend to every other position, enabling the
        model to capture long-range dependencies in a single operation. Unlike RNNs, the path length
        between any two positions is O(1), not O(n).
      </p>

      <DefinitionBlock title="Self-Attention">
        <p>In self-attention, the queries, keys, and values all come from the same sequence:</p>
        <BlockMath math="Q = XW^Q, \quad K = XW^K, \quad V = XW^V" />
        <BlockMath math="\text{SelfAttn}(X) = \text{softmax}\!\left(\frac{(XW^Q)(XW^K)^\top}{\sqrt{d_k}}\right)(XW^V)" />
      </DefinitionBlock>

      <SelfAttentionDemo />

      <ExampleBlock title="Disambiguation via Self-Attention">
        <p>
          Consider &quot;The <strong>bank</strong> of the river&quot; vs &quot;The <strong>bank</strong> approved the loan.&quot;
          Self-attention allows &quot;bank&quot; to attend to surrounding context (river vs. loan),
          enabling different representations of the same word depending on context.
        </p>
      </ExampleBlock>

      <TheoremBlock title="Self-Attention Complexity" id="self-attn-complexity">
        <p>Self-attention has:</p>
        <BlockMath math="\text{Time: } O(n^2 \cdot d), \quad \text{Memory: } O(n^2 + n \cdot d)" />
        <p className="mt-2">
          The <InlineMath math="O(n^2)" /> term comes from computing all pairwise attention scores.
          This quadratic cost is the primary bottleneck for long sequences.
        </p>
      </TheoremBlock>

      <PythonCode
        title="Self-Attention from Scratch"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)
        self.scale = d_k ** 0.5

    def forward(self, x):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V), attn

# Example: batch=2, seq_len=5, d_model=16, d_k=8
x = torch.randn(2, 5, 16)
sa = SelfAttention(d_model=16, d_k=8)
output, attn_weights = sa(x)
print(f"Output: {output.shape}")          # (2, 5, 8)
print(f"Attn weights: {attn_weights.shape}")  # (2, 5, 5)`}
      />

      <NoteBlock type="note" title="Self-Attention vs Convolution vs Recurrence">
        <p>
          Self-attention connects all positions with O(1) path length and O(n) sequential operations
          (parallelizable). Convolutions have O(n/k) path length and are also parallel, but with limited
          receptive field. Recurrence has O(n) path length and O(n) sequential steps (not parallelizable).
        </p>
      </NoteBlock>
    </div>
  )
}

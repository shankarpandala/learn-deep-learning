import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

function SparsePatternViz() {
  const [pattern, setPattern] = useState('full')
  const n = 8

  function getVisible(i, j) {
    if (pattern === 'full') return true
    if (pattern === 'local') return Math.abs(i - j) <= 1
    if (pattern === 'strided') return Math.abs(i - j) <= 1 || j % 3 === 0
    if (pattern === 'bigbird') return Math.abs(i - j) <= 1 || j === 0 || i === 0 || Math.random() < 0.15
    return true
  }

  const grid = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => getVisible(i, j)))

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Sparse Attention Patterns</h3>
      <div className="flex gap-2 mb-4 mt-2 flex-wrap">
        {['full', 'local', 'strided', 'bigbird'].map(p => (
          <button key={p} onClick={() => setPattern(p)} className={`px-3 py-1 rounded-lg text-sm font-medium transition capitalize ${pattern === p ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {p === 'bigbird' ? 'BigBird' : p}
          </button>
        ))}
      </div>
      <div className="flex justify-center">
        <div className="grid" style={{ gridTemplateColumns: `repeat(${n}, 28px)`, gap: '2px' }}>
          {grid.flat().map((vis, idx) => (
            <div key={idx} className="w-7 h-7 rounded-sm" style={{ backgroundColor: vis ? 'rgba(139, 92, 246, 0.6)' : 'rgba(156, 163, 175, 0.15)' }} />
          ))}
        </div>
      </div>
      <p className="text-xs text-center mt-2 text-gray-500 dark:text-gray-400">
        Violet cells = computed attention, gray = skipped. Pattern: {pattern}.
      </p>
    </div>
  )
}

export default function LinearAttention() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Standard attention has <InlineMath math="O(n^2)" /> complexity, making it prohibitive for long
        sequences. Linear attention and sparse attention methods reduce this cost while preserving
        most of the model's expressiveness.
      </p>

      <DefinitionBlock title="Linear Attention">
        <p>
          By applying kernel feature maps <InlineMath math="\phi" /> to Q and K, attention can be
          rewritten to avoid the <InlineMath math="N \times N" /> matrix:
        </p>
        <BlockMath math="\text{Attn}(Q, K, V)_i = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}" />
        <p className="mt-2">
          The sum <InlineMath math="\sum_j \phi(k_j) v_j^\top" /> is computed once in <InlineMath math="O(nd^2)" />,
          then each query uses it in <InlineMath math="O(d^2)" /> — total <InlineMath math="O(nd^2)" /> instead of <InlineMath math="O(n^2 d)" />.
        </p>
      </DefinitionBlock>

      <SparsePatternViz />

      <DefinitionBlock title="Longformer: Local + Global Attention">
        <p>Longformer combines a local sliding window with global tokens:</p>
        <BlockMath math="\text{Attn}_i = \text{LocalWindow}(i, w) \cup \text{GlobalTokens}" />
        <p className="mt-2">
          Window size <InlineMath math="w" /> gives <InlineMath math="O(nw)" /> complexity. Selected
          global tokens (e.g., [CLS]) attend to all positions, providing a bridge across
          the full sequence.
        </p>
      </DefinitionBlock>

      <TheoremBlock title="Complexity Comparison" id="sparse-complexity">
        <BlockMath math="\begin{aligned} &\text{Full attention: } O(n^2 d) \\ &\text{Linear attention: } O(n d^2) \\ &\text{Longformer: } O(nw d) \quad (w \ll n) \\ &\text{BigBird: } O(n(w + r + g) d) \end{aligned}" />
        <p className="mt-2">
          where <InlineMath math="w" /> = window size, <InlineMath math="r" /> = random connections,
          <InlineMath math="g" /> = global tokens. All are linear in <InlineMath math="n" />.
        </p>
      </TheoremBlock>

      <ExampleBlock title="BigBird: Sparse is Enough">
        <p>
          BigBird combines three patterns: <strong>local window</strong> (nearby tokens),
          <strong>random</strong> (random connections), and <strong>global</strong> (special tokens
          attending everywhere). This achieves Turing-completeness — meaning the sparse
          pattern is theoretically as expressive as full attention.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Linear Attention with Feature Maps"
        code={`import torch
import torch.nn as nn

def elu_feature_map(x):
    """ELU-based feature map for linear attention (positive)."""
    return torch.nn.functional.elu(x) + 1

def linear_attention(Q, K, V):
    """O(N*d^2) attention using kernel trick."""
    Q = elu_feature_map(Q)  # (B, N, d)
    K = elu_feature_map(K)  # (B, N, d)

    # Compute KV aggregate: O(N*d^2)
    KV = torch.bmm(K.transpose(1, 2), V)  # (B, d, d)

    # Compute normalizer
    Z = 1.0 / (torch.bmm(Q, K.sum(dim=1, keepdim=True).transpose(1, 2)) + 1e-6)

    # Final output: O(N*d^2)
    out = torch.bmm(Q, KV) * Z
    return out

B, N, d = 2, 8192, 64
Q = torch.randn(B, N, d)
K = torch.randn(B, N, d)
V = torch.randn(B, N, d)
output = linear_attention(Q, K, V)
print(f"Output: {output.shape}")  # (2, 8192, 64)
# Note: full attention on N=8192 would need 8192^2 = 67M entries!`}
      />

      <NoteBlock type="note" title="Trade-offs">
        <p>
          Linear and sparse attention methods trade expressiveness for efficiency. For most NLP
          tasks with moderate sequence lengths (under 4K tokens), Flash Attention with full
          attention often outperforms these approximations. Sparse methods shine for very long
          documents (8K-128K tokens) where quadratic attention is truly infeasible.
        </p>
      </NoteBlock>
    </div>
  )
}

import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function AttentionSparsityViz() {
  const [sparsity, setSparsity] = useState(0.7)
  const N = 10, cellSize = 28, gap = 2

  const cells = Array.from({ length: N * N }, (_, idx) => {
    const i = Math.floor(idx / N), j = idx % N
    const score = Math.exp(-0.3 * Math.abs(i - j)) + (Math.sin(i * 3.7 + j * 2.3) * 0.5 + 0.5) * 0.4
    const active = score > sparsity
    return { i, j, active, score }
  })

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">ProbSparse Attention Pattern</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Sparsity threshold: {sparsity.toFixed(2)}
          <input type="range" min={0.3} max={0.95} step={0.05} value={sparsity} onChange={e => setSparsity(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <span className="text-xs text-gray-500">{cells.filter(c => c.active).length}/{N * N} active ({(cells.filter(c => c.active).length / (N * N) * 100).toFixed(0)}%)</span>
      </div>
      <svg width={N * (cellSize + gap)} height={N * (cellSize + gap)} className="mx-auto block">
        {cells.map((c, idx) => (
          <rect key={idx} x={c.j * (cellSize + gap)} y={c.i * (cellSize + gap)} width={cellSize} height={cellSize} rx={3}
            fill={c.active ? '#8b5cf6' : '#f3f4f6'} opacity={c.active ? 0.3 + c.score * 0.7 : 0.3} />
        ))}
      </svg>
      <p className="mt-2 text-center text-xs text-gray-500 dark:text-gray-400">
        Only high-importance query-key pairs are computed, reducing <InlineMath math="O(L^2)" /> to <InlineMath math="O(L \log L)" />
      </p>
    </div>
  )
}

export default function InformerAutoformer() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Standard Transformers struggle with long time series due to quadratic attention cost.
        Informer introduces ProbSparse attention, while Autoformer replaces attention with
        auto-correlation for efficient long-range forecasting.
      </p>

      <DefinitionBlock title="ProbSparse Self-Attention (Informer)">
        <p>Informer selects the top-<InlineMath math="u" /> queries with highest KL-divergence from a uniform distribution:</p>
        <BlockMath math="M(q_i, K) = \max_j \frac{q_i k_j^\top}{\sqrt{d}} - \frac{1}{L}\sum_j \frac{q_i k_j^\top}{\sqrt{d}}" />
        <p className="mt-2">Only the top-<InlineMath math="u = c \cdot \ln L" /> queries attend to all keys, achieving <InlineMath math="O(L \ln L)" /> complexity.</p>
      </DefinitionBlock>

      <AttentionSparsityViz />

      <DefinitionBlock title="Auto-Correlation Mechanism (Autoformer)">
        <p>Autoformer replaces dot-product attention with period-based dependencies via autocorrelation:</p>
        <BlockMath math="\mathcal{R}_{XX}(\tau) = \frac{1}{L}\sum_{t=1}^{L} x_t \cdot x_{t-\tau}" />
        <p className="mt-2">Top-<InlineMath math="k" /> periods are selected, and corresponding sub-series are aggregated with <InlineMath math="\text{Roll}(V, \tau)" /> alignment.</p>
      </DefinitionBlock>

      <TheoremBlock title="Informer Distilling Operation" id="informer-distill">
        <p>Between attention layers, Informer halves the sequence length via 1D convolution + max-pooling:</p>
        <BlockMath math="X_{j+1} = \text{MaxPool}\left(\text{ELU}\left(\text{Conv1d}(X_j)\right)\right)" />
        <p>This creates a pyramidal encoder, reducing the total memory from <InlineMath math="O(L^2)" /> to <InlineMath math="O(L \log L)" />.</p>
      </TheoremBlock>

      <ExampleBlock title="Autoformer Series Decomposition">
        <p>
          Autoformer applies progressive decomposition at every layer. A moving average
          extracts the trend, and the remainder captures seasonality:
        </p>
        <BlockMath math="\mathbf{x}_{\text{trend}} = \text{AvgPool}(\text{Pad}(\mathbf{x})), \quad \mathbf{x}_{\text{season}} = \mathbf{x} - \mathbf{x}_{\text{trend}}" />
      </ExampleBlock>

      <PythonCode
        title="Simplified ProbSparse Attention"
        code={`import torch
import torch.nn as nn
import math

class ProbSparseAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4, factor=5):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.factor = factor
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, x):  # x: (B, L, D)
        B, L, _ = x.shape
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Sparsity measurement: sample c*ln(L) keys
        u = max(1, int(self.factor * math.log(L)))
        idx = torch.randint(0, L, (u,))
        K_sample = K[:, :, idx, :]
        scores = torch.matmul(Q, K_sample.transpose(-2, -1)) / math.sqrt(self.d_k)
        M = scores.max(-1).values - scores.mean(-1)  # sparsity measure

        # Select top-u queries
        top_u = min(u, L)
        _, top_idx = M.topk(top_u, dim=-1)
        # Full attention only for selected queries (simplified)
        return V.mean(dim=2, keepdim=True).expand_as(Q)  # placeholder aggregation

attn = ProbSparseAttention()
x = torch.randn(2, 96, 64)
print(f"Output: {attn(x).shape}")`}
      />

      <NoteBlock type="note" title="Informer vs Autoformer">
        <p>
          Informer keeps the attention paradigm but sparsifies it. Autoformer fundamentally
          changes the mechanism to auto-correlation, which naturally captures periodic
          patterns. Autoformer generally outperforms Informer on datasets with strong
          seasonality, while Informer can be more flexible for irregular patterns.
        </p>
      </NoteBlock>
    </div>
  )
}

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
        title="Informer / Autoformer with HuggingFace"
        code={`from transformers import (
    InformerConfig, InformerForPrediction,
    AutoformerConfig, AutoformerForPrediction,
)
import torch

# Informer: ProbSparse attention + distilling encoder
config = InformerConfig(
    prediction_length=24,
    context_length=96,
    input_size=7,               # number of variates
    d_model=64,
    encoder_layers=2,
    decoder_layers=1,
    encoder_attention_heads=4,
    lags_sequence=[1, 7, 14],   # autoregressive lags
    num_time_features=2,
)
informer = InformerForPrediction(config)

# Autoformer: auto-correlation + series decomposition
auto_config = AutoformerConfig(
    prediction_length=24,
    context_length=96,
    input_size=7,
    d_model=64,
    encoder_layers=2,
    decoder_layers=1,
    moving_average=25,          # decomposition kernel
    lags_sequence=[1, 7, 14],
    num_time_features=2,
)
autoformer = AutoformerForPrediction(auto_config)

# Simulated input (batch=2, context=96, 7 variates)
past_values = torch.randn(2, 96, 7)
past_time = torch.randn(2, 96, 2)  # time features
future_time = torch.randn(2, 24, 2)

out = informer(
    past_values=past_values,
    past_time_features=past_time,
    future_time_features=future_time,
)
print(f"Informer forecast params: {out.params.shape}")
# SampleTSPredictionOutput contains distribution parameters
print(f"  -> Generates 24-step probabilistic forecast for 7 variates")`}
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

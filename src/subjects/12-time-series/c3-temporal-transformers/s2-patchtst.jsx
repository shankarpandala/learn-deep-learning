import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function PatchingVisualizer() {
  const [patchLen, setPatchLen] = useState(8)
  const [stride, setStride] = useState(8)
  const T = 48
  const W = 420, H = 100, barH = 24

  const patches = []
  for (let i = 0; i + patchLen <= T; i += stride) {
    patches.push({ start: i, end: i + patchLen })
  }
  const colors = ['#8b5cf6', '#f97316', '#06b6d4', '#ec4899', '#10b981', '#f59e0b']
  const cellW = W / T

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Time Series Patching</h3>
      <div className="flex flex-wrap gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Patch length: {patchLen}
          <input type="range" min={4} max={16} step={2} value={patchLen} onChange={e => setPatchLen(parseInt(e.target.value))} className="w-24 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Stride: {stride}
          <input type="range" min={2} max={16} step={2} value={stride} onChange={e => setStride(parseInt(e.target.value))} className="w-24 accent-violet-500" />
        </label>
        <span className="text-xs text-gray-500 self-center">{patches.length} patches (tokens to Transformer)</span>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        {Array.from({ length: T }, (_, i) => (
          <rect key={i} x={i * cellW + 0.5} y={10} width={cellW - 1} height={barH} rx={2} fill="#e5e7eb" />
        ))}
        {patches.map((p, pi) => (
          <g key={pi}>
            <rect x={p.start * cellW} y={50} width={(p.end - p.start) * cellW - 1} height={barH} rx={4}
              fill={colors[pi % colors.length]} opacity={0.7} />
            <text x={(p.start + (p.end - p.start) / 2) * cellW} y={67} textAnchor="middle" className="text-[9px] fill-white font-bold">P{pi + 1}</text>
            {Array.from({ length: p.end - p.start }, (_, j) => (
              <line key={j} x1={(p.start + j + 0.5) * cellW} y1={10 + barH} x2={(p.start + j + 0.5) * cellW} y2={50}
                stroke={colors[pi % colors.length]} strokeWidth={0.5} opacity={0.4} />
            ))}
          </g>
        ))}
        <text x={2} y={8} className="text-[9px] fill-gray-400">Raw time steps (T={T})</text>
        <text x={2} y={88} className="text-[9px] fill-gray-400">Patches as Transformer tokens</text>
      </svg>
    </div>
  )
}

export default function PatchTSTChannelIndependence() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        PatchTST applies Vision Transformer-inspired patching to time series, dramatically
        reducing the token count fed to the Transformer while preserving local semantic
        information. Combined with channel independence, it achieves state-of-the-art
        long-term forecasting results.
      </p>

      <DefinitionBlock title="Time Series Patching">
        <p>A univariate series <InlineMath math="\mathbf{x} \in \mathbb{R}^L" /> is segmented into patches of length <InlineMath math="P" /> with stride <InlineMath math="S" />:</p>
        <BlockMath math="\mathbf{p}_i = \mathbf{x}_{iS : iS+P}, \quad i = 0, \ldots, \left\lfloor\frac{L-P}{S}\right\rfloor" />
        <p className="mt-2">Each patch is linearly projected to dimension <InlineMath math="d" />, producing <InlineMath math="N = \lfloor(L-P)/S\rfloor + 1" /> tokens.</p>
      </DefinitionBlock>

      <PatchingVisualizer />

      <TheoremBlock title="Complexity Reduction via Patching" id="patch-complexity">
        <p>Standard Transformer on raw time steps: <InlineMath math="O(L^2)" />. With patching:</p>
        <BlockMath math="O(N^2) = O\!\left(\left(\frac{L}{S}\right)^2\right)" />
        <p>For <InlineMath math="L=512, P=S=16" />: reduces from <InlineMath math="262{,}144" /> to <InlineMath math="1{,}024" /> attention computations (256x speedup).</p>
      </TheoremBlock>

      <DefinitionBlock title="Channel Independence">
        <p>
          For multivariate series <InlineMath math="\mathbf{X} \in \mathbb{R}^{C \times L}" />,
          channel independence processes each variable separately through the same Transformer:
        </p>
        <BlockMath math="\hat{\mathbf{y}}_c = f_\theta(\mathbf{x}_c), \quad c = 1, \ldots, C" />
        <p className="mt-2">Shared weights across channels act as implicit regularization, preventing overfitting to spurious cross-channel correlations.</p>
      </DefinitionBlock>

      <ExampleBlock title="Channel-Independent vs Channel-Mixing">
        <p>
          Counterintuitively, channel independence often outperforms models that explicitly
          model cross-variate dependencies (like full multivariate attention). The shared
          backbone learns universal temporal patterns while avoiding overfitting to
          dataset-specific inter-variable relationships.
        </p>
      </ExampleBlock>

      <PythonCode
        title="PatchTST with HuggingFace Transformers"
        code={`from transformers import PatchTSTConfig, PatchTSTForPrediction
import torch

# PatchTST: patching + channel independence + Transformer
config = PatchTSTConfig(
    num_input_channels=7,       # multivariate: 7 channels
    context_length=96,          # lookback window
    prediction_length=24,       # forecast horizon
    patch_length=16,            # each patch covers 16 time steps
    patch_stride=8,             # 50% overlap -> (96-16)//8 + 1 = 11 patches
    d_model=128,
    num_attention_heads=4,
    num_hidden_layers=3,
    feedforward_dim=256,
    dropout=0.1,
    channel_attention=False,    # channel independence (key design choice)
)
model = PatchTSTForPrediction(config)

# Input: (batch, seq_len, num_channels)
past_values = torch.randn(8, 96, 7)
outputs = model(past_values=past_values)

print(f"Input: {past_values.shape}")           # [8, 96, 7]
print(f"Forecast: {outputs.prediction_outputs.shape}")  # [8, 24, 7]
print(f"Patches per channel: {(96 - 16) // 8 + 1}")  # 11 tokens
print(f"Attention cost: 11^2 = 121 vs raw 96^2 = 9216 (76x reduction)")

# Self-supervised pre-training: mask random patches
config_ssl = PatchTSTConfig(
    num_input_channels=7, context_length=96, patch_length=16,
    patch_stride=8, d_model=128, num_attention_heads=4,
    num_hidden_layers=3, mask_type="random", random_mask_ratio=0.4,
)
# Use PatchTSTForPretraining for masked patch reconstruction`}
      />

      <NoteBlock type="note" title="Self-Supervised Pre-Training">
        <p>
          PatchTST supports masked patch pre-training analogous to BERT: randomly mask patches,
          reconstruct them, then fine-tune on the forecasting objective. This can improve
          performance by 2-5% on benchmarks, especially with limited labeled data.
        </p>
      </NoteBlock>
    </div>
  )
}

import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function MultivariateChannelViz() {
  const [strategy, setStrategy] = useState('fused')
  const C = 4, T = 30, W = 400, H = 160
  const channelH = H / C

  const channels = Array.from({ length: C }, (_, c) =>
    Array.from({ length: T }, (_, t) => Math.sin(t * 0.3 * (c + 1)) + Math.cos(t * 0.15) * (c * 0.3))
  )
  const colors = ['#8b5cf6', '#f97316', '#06b6d4', '#10b981']

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Multivariate Strategies</h3>
      <div className="flex gap-2 mb-3">
        {[
          { key: 'fused', label: 'Early Fusion' },
          { key: 'independent', label: 'Channel Independent' },
          { key: 'attention', label: 'Cross-Channel Attention' },
        ].map(s => (
          <button key={s.key} onClick={() => setStrategy(s.key)}
            className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${strategy === s.key ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'}`}>
            {s.label}
          </button>
        ))}
      </div>
      <svg width={W} height={H} className="mx-auto block">
        {channels.map((ch, c) => {
          const yOff = c * channelH
          const path = ch.map((v, t) => {
            const x = (t / T) * W
            const y = yOff + channelH / 2 - v * 6
            return `${t === 0 ? 'M' : 'L'}${x},${y}`
          }).join(' ')
          const opacity = strategy === 'independent' ? (c === 0 ? 1 : 0.3) : 1
          return <path key={c} d={path} fill="none" stroke={colors[c]} strokeWidth={1.5} opacity={opacity} />
        })}
        {strategy === 'attention' && (
          <>
            <line x1={W * 0.5} y1={channelH * 0.5} x2={W * 0.5} y2={channelH * 1.5} stroke="#9ca3af" strokeWidth={1} strokeDasharray="3,2" />
            <line x1={W * 0.5} y1={channelH * 1.5} x2={W * 0.5} y2={channelH * 2.5} stroke="#9ca3af" strokeWidth={1} strokeDasharray="3,2" />
            <line x1={W * 0.5} y1={channelH * 2.5} x2={W * 0.5} y2={channelH * 3.5} stroke="#9ca3af" strokeWidth={1} strokeDasharray="3,2" />
            <text x={W * 0.5 + 4} y={channelH * 2} className="text-[8px] fill-gray-400">attn</text>
          </>
        )}
        {strategy === 'fused' && (
          <rect x={0} y={0} width={W} height={H} fill="#8b5cf6" opacity={0.04} rx={4} />
        )}
      </svg>
      <p className="mt-2 text-center text-xs text-gray-500 dark:text-gray-400">
        {strategy === 'fused' && 'Stack all channels as input features — captures cross-channel patterns from the start'}
        {strategy === 'independent' && 'Process each channel separately, merge later — avoids spurious correlations'}
        {strategy === 'attention' && 'Cross-channel attention learns which inter-variable dependencies matter'}
      </p>
    </div>
  )
}

export default function MultivariateClassification() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Multivariate time series classification (MTSC) operates on <InlineMath math="C" /> correlated
        channels simultaneously. The key challenge is modeling both temporal dynamics within
        each channel and dependencies across channels effectively.
      </p>

      <DefinitionBlock title="Multivariate TS Classification">
        <p>Given input <InlineMath math="\mathbf{X} \in \mathbb{R}^{C \times T}" /> with <InlineMath math="C" /> variables and <InlineMath math="T" /> time steps:</p>
        <BlockMath math="f_\theta : \mathbb{R}^{C \times T} \to \Delta^K" />
        <p className="mt-2">Three strategies for handling multiple channels: early fusion, channel independence, and cross-channel attention.</p>
      </DefinitionBlock>

      <MultivariateChannelViz />

      <TheoremBlock title="Cross-Variable Attention" id="cross-var-attention">
        <p>Given per-channel representations <InlineMath math="\mathbf{H} \in \mathbb{R}^{C \times d}" />, cross-variable attention computes:</p>
        <BlockMath math="\text{CVAttn}(\mathbf{H}) = \text{Softmax}\!\left(\frac{\mathbf{H}\mathbf{H}^\top}{\sqrt{d}}\right)\mathbf{H}" />
        <p>This <InlineMath math="C \times C" /> attention matrix captures pairwise channel dependencies, scaling to hundreds of variables when <InlineMath math="C \ll T" />.</p>
      </TheoremBlock>

      <ExampleBlock title="ROCKET: Random Convolutional Kernels">
        <p>
          ROCKET generates thousands of random 1D convolutional kernels (varying lengths,
          dilations, biases) and extracts two features per kernel: max value and proportion
          of positive values. These features are fed to a simple linear classifier,
          achieving near state-of-the-art accuracy at a fraction of the training cost.
        </p>
      </ExampleBlock>

      <WarningBlock title="Curse of Dimensionality in MTSC">
        <p>
          Adding more variables does not always improve classification. Irrelevant channels
          add noise that can degrade performance. Use channel selection (dropout, learned
          gating, or mutual information) to identify which variables contain discriminative
          information for each class.
        </p>
      </WarningBlock>

      <PythonCode
        title="Multivariate TS Classifier with Cross-Channel Attention"
        code={`import torch
import torch.nn as nn

class MultivariateTSClassifier(nn.Module):
    def __init__(self, n_vars=6, seq_len=128, d_model=64, n_classes=4, n_heads=4):
        super().__init__()
        # Per-channel temporal encoder (shared weights)
        self.temporal_enc = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=8, padding=3),
            nn.ReLU(), nn.BatchNorm1d(d_model),
            nn.AdaptiveAvgPool1d(1)  # global average pooling
        )
        # Cross-variable attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(n_vars * d_model, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):  # x: (B, C, T)
        B, C, T = x.shape
        # Encode each channel independently
        h = torch.stack([self.temporal_enc(x[:, c:c+1, :]).squeeze(-1) for c in range(C)], dim=1)
        # Cross-channel attention: (B, C, d_model)
        h_attn, _ = self.cross_attn(h, h, h)
        h = self.norm(h + h_attn)
        # Flatten and classify
        return self.classifier(h.reshape(B, -1))

model = MultivariateTSClassifier(n_vars=6, seq_len=128, n_classes=4)
x = torch.randn(16, 6, 128)
print(f"Output: {model(x).shape}")  # (16, 4)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")`}
      />

      <NoteBlock type="note" title="Benchmarking: UEA Archive">
        <p>
          The UEA Multivariate Time Series Archive contains 30 benchmark datasets spanning
          domains like motion capture, medical sensors, and audio. When evaluating MTSC
          models, report critical difference diagrams across datasets rather than cherry-picking
          individual results. No single method dominates all datasets.
        </p>
      </NoteBlock>
    </div>
  )
}

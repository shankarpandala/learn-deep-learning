import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function AssociationDiscrepancyViz() {
  const [anomalyPos, setAnomalyPos] = useState(5)
  const N = 10, cellSize = 26, gap = 2

  const priorW = Array.from({ length: N }, (_, i) =>
    Array.from({ length: N }, (_, j) => {
      const dist = Math.abs(i - j)
      return Math.exp(-dist * 0.5)
    })
  )
  const seriesW = Array.from({ length: N }, (_, i) =>
    Array.from({ length: N }, (_, j) => {
      if (i === anomalyPos || j === anomalyPos) return 0.1 + Math.random() * 0.2
      return Math.exp(-Math.abs(i - j) * 0.4) * (0.8 + Math.random() * 0.2)
    })
  )

  const gridW = N * (cellSize + gap)
  const totalW = gridW * 2 + 60

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Prior vs Series Association</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Anomaly at position: {anomalyPos}
        <input type="range" min={0} max={9} step={1} value={anomalyPos} onChange={e => setAnomalyPos(parseInt(e.target.value))} className="w-28 accent-violet-500" />
      </label>
      <div className="overflow-x-auto">
        <svg width={totalW} height={gridW + 30} className="mx-auto block">
          <text x={gridW / 2} y={12} textAnchor="middle" className="text-[10px] fill-violet-600 font-semibold">Prior Association</text>
          {priorW.map((row, i) => row.map((v, j) => (
            <rect key={`p-${i}-${j}`} x={j * (cellSize + gap)} y={i * (cellSize + gap) + 18} width={cellSize} height={cellSize} rx={2}
              fill="#8b5cf6" opacity={v * 0.8} />
          )))}
          <text x={gridW + 30 + gridW / 2} y={12} textAnchor="middle" className="text-[10px] fill-orange-600 font-semibold">Series Association</text>
          {seriesW.map((row, i) => row.map((v, j) => (
            <rect key={`s-${i}-${j}`} x={gridW + 60 + j * (cellSize + gap)} y={i * (cellSize + gap) + 18} width={cellSize} height={cellSize} rx={2}
              fill={i === anomalyPos || j === anomalyPos ? '#ef4444' : '#f97316'} opacity={v * 0.8} />
          )))}
        </svg>
      </div>
      <p className="mt-2 text-center text-xs text-gray-500 dark:text-gray-400">
        Anomalous points show high discrepancy between prior (learned) and series (observed) associations
      </p>
    </div>
  )
}

export default function TransformerAnomalyDetection() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The Anomaly Transformer introduces a novel association discrepancy framework
        that leverages the difference between prior and series associations in attention
        maps to detect anomalies without requiring labeled anomaly data.
      </p>

      <DefinitionBlock title="Association Discrepancy">
        <p>For each time point, compute two association distributions:</p>
        <BlockMath math="\text{Prior: } P_t \sim \mathcal{N}(t, \sigma^2) \quad \text{(learned Gaussian kernel)}" />
        <BlockMath math="\text{Series: } S_t = \text{Softmax}(Q_t K^\top / \sqrt{d})" />
        <p className="mt-2">The anomaly score uses the KL-divergence between them:</p>
        <BlockMath math="\text{AssDis}(t) = \text{KL}(P_t \| S_t) + \text{KL}(S_t \| P_t)" />
      </DefinitionBlock>

      <AssociationDiscrepancyViz />

      <TheoremBlock title="Minimax Association Learning" id="anomaly-transformer-loss">
        <p>The Anomaly Transformer training objective uses a minimax strategy:</p>
        <BlockMath math="\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2 - \lambda \cdot \text{AssDis}(\text{stop\_grad}(P), S) + \lambda \cdot \text{AssDis}(P, \text{stop\_grad}(S))" />
        <p>
          The prior association is encouraged to approach the series association (minimizing
          discrepancy), while the series association is pushed away (maximizing discrepancy).
          This amplifies the difference for anomalous points.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Why Attention Reveals Anomalies">
        <p>
          Normal points form strong associations with their temporal neighbors — attention
          concentrates on nearby, similar patterns. Anomalous points cannot find similar
          patterns, so their attention becomes diffuse (high entropy), creating a measurable
          discrepancy from the expected prior association.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Anomaly Transformer Scoring"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class AnomalyAttentionLayer(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        # Learnable prior: sigma parameter for each head
        self.sigma = nn.Parameter(torch.ones(n_heads) * 1.0)

    def forward(self, x):  # x: (B, L, D)
        B, L, D = x.shape
        qkv = self.W_qkv(x).reshape(B, L, 3, self.n_heads, self.d_k)
        Q, K, V = qkv.unbind(dim=2)  # each: (B, L, H, d_k)
        Q, K, V = [t.transpose(1, 2) for t in (Q, K, V)]

        # Series association (standard attention)
        series_assoc = F.softmax(Q @ K.transpose(-2, -1) / self.d_k**0.5, dim=-1)

        # Prior association (Gaussian kernel)
        positions = torch.arange(L, device=x.device).float()
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1))**2
        prior_assoc = F.softmax(-dist / (2 * self.sigma.view(1, -1, 1, 1)**2 + 1e-8), dim=-1)
        prior_assoc = prior_assoc.expand(B, -1, -1, -1)

        # Association discrepancy per time point
        kl_ps = (prior_assoc * (prior_assoc.log() - series_assoc.log() + 1e-8)).sum(-1)
        kl_sp = (series_assoc * (series_assoc.log() - prior_assoc.log() + 1e-8)).sum(-1)
        discrepancy = (kl_ps + kl_sp).mean(dim=1)  # average over heads: (B, L)

        return (series_assoc @ V).transpose(1, 2).reshape(B, L, D), discrepancy

layer = AnomalyAttentionLayer()
x = torch.randn(4, 32, 64)
out, disc = layer(x)
print(f"Output: {out.shape}, Discrepancy: {disc.shape}")
print(f"Top anomaly scores: {disc[0].topk(3).values.tolist()}")`}
      />

      <NoteBlock type="note" title="Beyond Anomaly Transformer">
        <p>
          Other Transformer-based approaches include TranAD (adversarial training with
          attention), and GDN (graph deviation network for multivariate data). The key
          insight shared across methods: attention patterns contain rich information about
          temporal relationships that anomalies disrupt.
        </p>
      </NoteBlock>
    </div>
  )
}

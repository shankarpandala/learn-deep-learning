import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function GatingVisualizer() {
  const [numExperts, setNumExperts] = useState(8)
  const [topK, setTopK] = useState(2)
  const [seed, setSeed] = useState(42)

  const rng = (s) => { let x = Math.sin(s) * 10000; return x - Math.floor(x) }
  const rawScores = Array.from({ length: numExperts }, (_, i) => rng(seed + i * 7))
  const total = rawScores.reduce((a, b) => a + Math.exp(b * 3), 0)
  const probs = rawScores.map(s => Math.exp(s * 3) / total)
  const sorted = probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p)
  const selectedIdx = new Set(sorted.slice(0, topK).map(s => s.i))
  const selectedTotal = sorted.slice(0, topK).reduce((a, s) => a + s.p, 0)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Expert Gating Network</h3>
      <div className="flex items-center gap-4 mb-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Experts: {numExperts}
          <input type="range" min={4} max={64} step={4} value={numExperts} onChange={e => setNumExperts(parseInt(e.target.value))} className="w-24 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Top-K: {topK}
          <input type="range" min={1} max={Math.min(8, numExperts)} step={1} value={topK} onChange={e => setTopK(parseInt(e.target.value))} className="w-24 accent-violet-500" />
        </label>
        <button onClick={() => setSeed(seed + 1)} className="px-2 py-1 rounded text-xs bg-violet-100 text-violet-700 dark:bg-violet-900/30 dark:text-violet-300">New Token</button>
      </div>
      <div className="flex gap-1 items-end h-24">
        {probs.map((p, i) => (
          <div key={i} className="flex-1 flex flex-col items-center">
            <div className={`w-full rounded-t transition-all ${selectedIdx.has(i) ? 'bg-violet-500' : 'bg-gray-300 dark:bg-gray-600'}`} style={{ height: `${p * 300}px` }} />
            <span className="text-[8px] text-gray-500 mt-1">E{i}</span>
          </div>
        ))}
      </div>
      <p className="mt-2 text-xs text-gray-500 text-center">Top-{topK} experts activated ({(selectedTotal * 100).toFixed(1)}% of gating weight). Active FLOPs: {topK}/{numExperts} = {(topK / numExperts * 100).toFixed(0)}%</p>
    </div>
  )
}

export default function MoEArchitecture() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Mixture of Experts (MoE) scales model parameters without proportionally scaling compute
        by routing each token through only a subset of "expert" sub-networks. This enables
        trillion-parameter models with practical training and inference costs.
      </p>

      <DefinitionBlock title="Sparse MoE Layer">
        <p>An MoE layer replaces the standard FFN with a set of <InlineMath math="N" /> expert networks and a gating function:</p>
        <BlockMath math="y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x), \quad G(x) = \text{TopK}(\text{softmax}(W_g x))" />
        <p className="mt-2">where <InlineMath math="E_i" /> are expert FFNs and <InlineMath math="G(x)" /> is the gating network that selects the top-K experts. Only the selected experts compute their output, making the layer sparse.</p>
      </DefinitionBlock>

      <GatingVisualizer />

      <ExampleBlock title="MoE Efficiency Gains">
        <p>With <InlineMath math="N = 64" /> experts and <InlineMath math="K = 2" />:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>Total parameters: 64x the FFN size</li>
          <li>Active parameters per token: 2x the FFN size (only 3.1% of experts)</li>
          <li>FLOPs per token: ~2x a single expert (comparable to a dense model 64x smaller)</li>
          <li>Example: Mixtral 8x7B has 47B total params but uses ~13B per token</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Simple Top-K MoE Layer"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    """Mixture of Experts layer with top-k routing."""
    def __init__(self, dim=512, num_experts=8, top_k=2, expert_dim=2048):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, expert_dim), nn.ReLU(), nn.Linear(expert_dim, dim))
            for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: [batch, seq_len, dim]
        B, S, D = x.shape
        gate_logits = self.gate(x)                    # [B, S, num_experts]
        weights, indices = gate_logits.topk(self.top_k, dim=-1)  # [B, S, top_k]
        weights = F.softmax(weights, dim=-1)

        # Compute weighted sum of top-k expert outputs
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            for e_idx in range(len(self.experts)):
                mask = (indices[:, :, k] == e_idx)     # [B, S]
                if mask.any():
                    expert_input = x[mask]              # [num_tokens, D]
                    expert_out = self.experts[e_idx](expert_input)
                    output[mask] += weights[:, :, k][mask].unsqueeze(-1) * expert_out

        return output

moe = MoELayer(dim=512, num_experts=8, top_k=2)
x = torch.randn(2, 64, 512)
out = moe(x)
print(f"Input: {x.shape} -> Output: {out.shape}")
print(f"Total params: {sum(p.numel() for p in moe.parameters()):,}")`}
      />

      <NoteBlock type="note" title="Load Balancing Loss">
        <p>
          Without regularization, the gating network can collapse — routing all tokens to a few
          experts while others are unused. An auxiliary load balancing loss encourages uniform
          expert utilization: <InlineMath math="\mathcal{L}_{\text{aux}} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i" /> where
          <InlineMath math="f_i" /> is the fraction of tokens routed to expert <InlineMath math="i" /> and
          <InlineMath math="P_i" /> is the average gating probability for expert <InlineMath math="i" />.
        </p>
      </NoteBlock>
    </div>
  )
}

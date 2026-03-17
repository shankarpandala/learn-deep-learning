import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function MixtralComparison() {
  const [model, setModel] = useState('mixtral')
  const models = {
    mixtral: { name: 'Mixtral 8x7B', totalParams: '46.7B', activeParams: '12.9B', experts: 8, topK: 2, performance: 'Matches LLaMA-2 70B' },
    llama70: { name: 'LLaMA-2 70B', totalParams: '70B', activeParams: '70B', experts: 1, topK: 1, performance: 'Dense baseline' },
    gpt35: { name: 'GPT-3.5 (est.)', totalParams: '~175B (MoE?)', activeParams: '~20-30B', experts: '~16', topK: '~2', performance: 'Reference commercial model' },
  }
  const m = models[model]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">MoE Model Comparison</h3>
      <div className="flex gap-2 mb-3 flex-wrap">
        {Object.entries(models).map(([key, val]) => (
          <button key={key} onClick={() => setModel(key)}
            className={`px-3 py-1 rounded-lg text-sm transition ${model === key ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-2 text-sm">
        {Object.entries({ 'Total Params': m.totalParams, 'Active Params/Token': m.activeParams, 'Experts': m.experts, 'Top-K': m.topK, 'Performance': m.performance }).map(([k, v]) => (
          <div key={k} className="p-2 rounded bg-violet-50 dark:bg-violet-900/20">
            <span className="text-xs text-gray-500">{k}</span>
            <p className="font-medium text-gray-700 dark:text-gray-300">{v}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function MixtralModernMoE() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Mixtral 8x7B by Mistral AI demonstrated that open-source MoE models can match or exceed
        much larger dense models. It uses 8 expert FFN blocks with top-2 routing, achieving
        LLaMA-2 70B quality at a fraction of the inference cost.
      </p>

      <DefinitionBlock title="Mixtral Architecture">
        <p>Mixtral replaces each Transformer FFN with 8 expert FFNs, using top-2 routing:</p>
        <BlockMath math="\text{FFN}_{\text{MoE}}(x) = \sum_{i \in \text{Top2}(G(x))} g_i(x) \cdot \text{FFN}_i(x)" />
        <p className="mt-2">All other components (attention, normalization) are shared across experts. Total parameters: 46.7B. Active parameters per token: 12.9B (two 6.45B experts). This gives 70B-class quality with 13B-class inference cost.</p>
      </DefinitionBlock>

      <MixtralComparison />

      <ExampleBlock title="Expert Specialization in Mixtral">
        <p>Analysis of Mixtral's routing reveals soft specialization patterns:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>Experts show domain preferences but are not strictly specialized</li>
          <li>Routing is primarily syntax-driven (e.g., by token position, not semantics)</li>
          <li>Different layers show different specialization patterns</li>
          <li>No single expert can be removed without degrading all domains</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Mixtral-Style MoE Forward Pass"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class MixtralMoEBlock(nn.Module):
    """Mixtral-style MoE with 8 experts and top-2 routing."""
    def __init__(self, dim=4096, ffn_dim=14336, num_experts=8):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(dim, num_experts, bias=False)

        # Each expert is a standard SwiGLU FFN
        self.w1 = nn.ModuleList([nn.Linear(dim, ffn_dim, bias=False) for _ in range(num_experts)])
        self.w2 = nn.ModuleList([nn.Linear(ffn_dim, dim, bias=False) for _ in range(num_experts)])
        self.w3 = nn.ModuleList([nn.Linear(dim, ffn_dim, bias=False) for _ in range(num_experts)])

    def expert_fn(self, x, idx):
        """SwiGLU expert: w2(SiLU(w1(x)) * w3(x))"""
        return self.w2[idx](F.silu(self.w1[idx](x)) * self.w3[idx](x))

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(-1, D)

        # Gate and select top-2
        logits = self.gate(x_flat)
        weights, indices = logits.topk(2, dim=-1)
        weights = F.softmax(weights, dim=-1)

        # Weighted combination of top-2 experts
        output = torch.zeros_like(x_flat)
        for k in range(2):
            for e in range(self.num_experts):
                mask = (indices[:, k] == e)
                if mask.any():
                    expert_out = self.expert_fn(x_flat[mask], e)
                    output[mask] += weights[mask, k:k+1] * expert_out

        return output.view(B, S, D)

# Mixtral dimensions (scaled down for demo)
moe = MixtralMoEBlock(dim=256, ffn_dim=512, num_experts=8)
x = torch.randn(1, 32, 256)
out = moe(x)
total_p = sum(p.numel() for p in moe.parameters())
print(f"Output: {out.shape}, Total params: {total_p:,}")`}
      />

      <NoteBlock type="note" title="MoE Deployment Challenges">
        <p>
          MoE models require all expert weights in memory even though only a few are active per token.
          Mixtral 8x7B needs ~90GB in FP16 (all 47B params loaded), limiting it to multi-GPU setups.
          Expert offloading (keeping idle experts on CPU/disk) and expert merging are active research
          areas for making MoE models more practical for deployment.
        </p>
      </NoteBlock>
    </div>
  )
}

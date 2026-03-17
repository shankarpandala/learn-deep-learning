import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function CapacityExplorer() {
  const [numExperts, setNumExperts] = useState(128)
  const [capacityFactor, setCapacityFactor] = useState(1.25)
  const [batchTokens, setBatchTokens] = useState(4096)
  const tokensPerExpert = Math.ceil((batchTokens / numExperts) * capacityFactor)
  const totalCapacity = tokensPerExpert * numExperts
  const overhead = ((totalCapacity / batchTokens - 1) * 100).toFixed(1)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Expert Capacity Calculator</h3>
      <div className="flex items-center gap-4 mb-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Experts: {numExperts}
          <input type="range" min={8} max={256} step={8} value={numExperts} onChange={e => setNumExperts(parseInt(e.target.value))} className="w-24 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Capacity factor: {capacityFactor.toFixed(2)}
          <input type="range" min={1.0} max={2.0} step={0.05} value={capacityFactor} onChange={e => setCapacityFactor(parseFloat(e.target.value))} className="w-24 accent-violet-500" />
        </label>
      </div>
      <div className="grid grid-cols-3 gap-3 text-sm text-center">
        <div className="p-2 rounded bg-violet-50 dark:bg-violet-900/20">
          <p className="text-violet-700 dark:text-violet-300 font-medium">Tokens/Expert</p>
          <p className="font-bold">{tokensPerExpert}</p>
        </div>
        <div className="p-2 rounded bg-violet-50 dark:bg-violet-900/20">
          <p className="text-violet-700 dark:text-violet-300 font-medium">Batch Tokens</p>
          <p className="font-bold">{batchTokens}</p>
        </div>
        <div className="p-2 rounded bg-violet-50 dark:bg-violet-900/20">
          <p className="text-violet-700 dark:text-violet-300 font-medium">Buffer Overhead</p>
          <p className="font-bold">{overhead}%</p>
        </div>
      </div>
      <p className="mt-2 text-xs text-gray-500 text-center">Capacity factor &gt; 1 provides buffer for imbalanced routing. Tokens exceeding capacity are dropped.</p>
    </div>
  )
}

export default function SwitchTransformerGShard() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Switch Transformer simplifies MoE routing to top-1 expert selection, achieving better
        scaling with reduced communication cost. GShard demonstrated trillion-parameter MoE
        models across thousands of devices with expert parallelism.
      </p>

      <DefinitionBlock title="Switch Routing (Top-1)">
        <p>Switch Transformer routes each token to exactly one expert, simplifying the MoE formulation:</p>
        <BlockMath math="y = G(x)_{i^*} \cdot E_{i^*}(x), \quad i^* = \arg\max_i (W_g x)_i" />
        <p className="mt-2">The gating weight <InlineMath math="G(x)_{i^*}" /> acts as a confidence score. The expert capacity <InlineMath math="C" /> limits tokens per expert per batch:</p>
        <BlockMath math="C = \text{CF} \times \frac{\text{tokens\_in\_batch}}{N_{\text{experts}}}" />
        <p className="mt-1">where CF is the capacity factor (typically 1.0-1.5). Tokens routed to a full expert are dropped.</p>
      </DefinitionBlock>

      <CapacityExplorer />

      <ExampleBlock title="Switch Transformer Scaling Results">
        <p>Switch Transformer demonstrates superior scaling over dense models:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>1.6T parameters with 128 experts, matching T5-XXL quality with 7x fewer training steps</li>
          <li>Same FLOPs as T5-Base but 7x more parameters &rarr; significant quality improvement</li>
          <li>Expert parallelism: each expert on a separate device, all-to-all communication between them</li>
          <li>Key finding: top-1 routing works as well as top-2 with simpler implementation</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Switch Router with Capacity and Load Balancing"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class SwitchRouter(nn.Module):
    """Top-1 expert routing with capacity and load balancing."""
    def __init__(self, dim, num_experts, capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x):
        # x: [B*S, D]
        logits = self.gate(x)                           # [B*S, E]
        probs = F.softmax(logits, dim=-1)

        # Top-1 selection
        gate_values, expert_idx = probs.max(dim=-1)     # [B*S]
        num_tokens = x.shape[0]
        capacity = int(self.capacity_factor * num_tokens / self.num_experts)

        # Build dispatch mask with capacity constraint
        dispatch = torch.zeros(num_tokens, self.num_experts, device=x.device)
        expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)

        for i in range(num_tokens):
            e = expert_idx[i].item()
            if expert_counts[e] < capacity:
                dispatch[i, e] = gate_values[i]
                expert_counts[e] += 1
            # else: token is dropped (overflow)

        # Load balancing loss
        f = expert_counts.float() / num_tokens  # fraction routed
        P = probs.mean(dim=0)                     # average probability
        aux_loss = self.num_experts * (f * P).sum()

        return dispatch, aux_loss, expert_counts

router = SwitchRouter(dim=512, num_experts=8)
tokens = torch.randn(64, 512)
dispatch, loss, counts = router(tokens)
print(f"Expert load: {counts.tolist()}")
print(f"Auxiliary loss: {loss.item():.4f}")
print(f"Dropped tokens: {64 - dispatch.sum().item():.0f}")`}
      />

      <NoteBlock type="note" title="Expert Parallelism in Distributed Training">
        <p>
          GShard places each expert on a separate accelerator. An all-to-all communication step
          dispatches tokens to their assigned experts across devices, then gathers results back.
          This is communication-efficient because each token goes to exactly one expert (top-1),
          minimizing cross-device traffic. With 2048 experts across 2048 TPUs, GShard trained
          a 600B parameter model.
        </p>
      </NoteBlock>
    </div>
  )
}

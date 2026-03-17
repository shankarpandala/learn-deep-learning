import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ComputeAllocationViz() {
  const [logCompute, setLogCompute] = useState(23)
  const C = Math.pow(10, logCompute)
  const chinchillaParams = Math.pow(C / 6 / 20, 0.5) * Math.sqrt(20)
  const chinchillaTokens = C / (6 * chinchillaParams)
  const llamaParams = chinchillaParams * 0.5
  const llamaTokens = C / (6 * llamaParams)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Compute Budget Allocation Strategies</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Compute budget: 10^{logCompute} FLOPs
        <input type="range" min={20} max={26} step={0.5} value={logCompute} onChange={e => setLogCompute(parseFloat(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20">
          <p className="font-medium text-violet-700 dark:text-violet-300">Chinchilla-Optimal</p>
          <p>N: {(chinchillaParams / 1e9).toFixed(1)}B params</p>
          <p>D: {(chinchillaTokens / 1e9).toFixed(0)}B tokens</p>
          <p className="text-xs text-gray-500">Tokens/param ratio: ~20</p>
        </div>
        <div className="p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20">
          <p className="font-medium text-violet-700 dark:text-violet-300">Inference-Optimal (LLaMA-style)</p>
          <p>N: {(llamaParams / 1e9).toFixed(1)}B params</p>
          <p>D: {(llamaTokens / 1e9).toFixed(0)}B tokens</p>
          <p className="text-xs text-gray-500">Tokens/param ratio: ~{(llamaTokens / llamaParams).toFixed(0)}</p>
        </div>
      </div>
      <p className="mt-2 text-xs text-gray-500 text-center">Same compute budget, different allocation — smaller model trained on more data is cheaper at inference.</p>
    </div>
  )
}

export default function ComputeOptimalTraining() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Compute-optimal training balances model size and training tokens for a fixed compute
        budget. The Chinchilla result showed most LLMs were significantly undertrained, but
        modern practice deliberately deviates for inference efficiency.
      </p>

      <DefinitionBlock title="Compute-Optimal vs Inference-Optimal">
        <p><strong>Compute-optimal</strong> minimizes loss for a fixed training FLOP budget <InlineMath math="C" />:</p>
        <BlockMath math="\min_{N,D: 6ND = C} L(N, D) \implies D^* \approx 20N^*" />
        <p className="mt-2"><strong>Inference-optimal</strong> minimizes total cost (training + inference) over the model lifetime:</p>
        <BlockMath math="\min_{N,D} C_{\text{train}}(N, D) + T \cdot C_{\text{inference}}(N) \quad \text{s.t. } L(N, D) \leq L_{\text{target}}" />
        <p className="mt-1">where <InlineMath math="T" /> is the expected number of inference tokens. For high <InlineMath math="T" />, smaller models trained on more data are preferred.</p>
      </DefinitionBlock>

      <ComputeAllocationViz />

      <ExampleBlock title="Why LLaMA-3 Over-Trains">
        <p>LLaMA-3 70B trains on 15T tokens (~215 tokens/parameter vs Chinchilla's 20):</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>Training cost: ~10x Chinchilla-optimal for this model size</li>
          <li>But inference cost: 1x the 70B model (vs a 400B Chinchilla-optimal model)</li>
          <li>After ~1M inference requests, the total cost is lower than the larger model</li>
          <li>The log-loss reduction continues smoothly even at 200+ tokens/parameter</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Compute-Optimal vs Inference-Optimal Analysis"
        code={`import math

def total_cost(N, D, inference_tokens, cost_per_flop=1e-18):
    """Total cost = training + inference over model lifetime."""
    train_flops = 6 * N * D
    # Inference: ~2N FLOPs per token (forward pass only)
    infer_flops = 2 * N * inference_tokens
    total_flops = train_flops + infer_flops
    return total_flops * cost_per_flop

def chinchilla_loss(N, D):
    """Approximate Chinchilla loss model."""
    return 1.69 + 406.4 / N**0.34 + 410.7 / D**0.28

# Compare strategies for target loss of 1.85
target_loss = 1.85

# Strategy 1: Chinchilla-optimal (N = D/20)
# Find N such that chinchilla_loss(N, 20*N) = target
N_chin = 70e9
D_chin = 20 * N_chin

# Strategy 2: Over-trained (same loss, smaller model)
N_small = 30e9
D_small = 200 * N_small  # 200 tokens/param

# Compare total costs for different inference workloads
print(f"{'Inference Tokens':>20} {'Chinchilla Cost':>18} {'Over-trained Cost':>18} {'Winner':>10}")
print("-" * 70)
for log_infer in [12, 13, 14, 15]:
    infer_tok = 10**log_infer
    cost_chin = total_cost(N_chin, D_chin, infer_tok)
    cost_small = total_cost(N_small, D_small, infer_tok)
    winner = "Chinchilla" if cost_chin < cost_small else "Over-train"
    print(f"  10^{log_infer} tokens    ${cost_chin:.2f}          ${cost_small:.2f}           {winner}")
print("\\nConclusion: Over-training wins when inference demand is high")`}
      />

      <NoteBlock type="note" title="Beyond Power Laws: Data Quality">
        <p>
          Scaling laws assume infinite unique data. In practice, data quality, diversity, and
          deduplication matter as much as quantity. Training on curated, high-quality data can
          achieve the same loss as 10x more unfiltered data. The "data wall" — running out of
          high-quality internet text — is a growing concern for continued scaling.
        </p>
      </NoteBlock>
    </div>
  )
}

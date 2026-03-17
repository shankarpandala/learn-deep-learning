import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function EmergenceChart() {
  const [metric, setMetric] = useState('exact_match')
  const W = 400, H = 200, pad = 40
  const modelSizes = [0.1, 1, 10, 100, 1000]
  const logSizes = modelSizes.map(s => Math.log10(s))

  const metrics = {
    exact_match: { name: 'Exact Match (sharp emergence)', values: [0.0, 0.01, 0.02, 0.15, 0.85] },
    log_likelihood: { name: 'Log-Likelihood (smooth scaling)', values: [0.1, 0.25, 0.42, 0.6, 0.78] },
  }
  const m = metrics[metric]

  const xScale = (v) => pad + (v - logSizes[0]) / (logSizes[logSizes.length - 1] - logSizes[0]) * (W - 2 * pad)
  const yScale = (v) => H - pad - v * (H - 2 * pad)

  const path = m.values.map((v, i) => `${i === 0 ? 'M' : 'L'}${xScale(logSizes[i])},${yScale(v)}`).join(' ')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Emergence vs Smooth Scaling</h3>
      <div className="flex gap-2 mb-3">
        {Object.entries(metrics).map(([key, val]) => (
          <button key={key} onClick={() => setMetric(key)}
            className={`px-3 py-1 rounded-lg text-xs transition ${metric === key ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#d1d5db" strokeWidth={0.5} />
        <path d={path} fill="none" stroke="#8b5cf6" strokeWidth={2.5} />
        {m.values.map((v, i) => <circle key={i} cx={xScale(logSizes[i])} cy={yScale(v)} r={4} fill="#8b5cf6" />)}
        {modelSizes.map((s, i) => <text key={i} x={xScale(logSizes[i])} y={H - 10} textAnchor="middle" className="text-[9px] fill-gray-500">{s >= 1 ? s + 'B' : s * 1000 + 'M'}</text>)}
        <text x={W / 2} y={H - 1} textAnchor="middle" className="text-[10px] fill-gray-500">Model Size</text>
        <text x={12} y={H / 2} textAnchor="middle" transform={`rotate(-90,12,${H / 2})`} className="text-[10px] fill-gray-500">Score</text>
      </svg>
    </div>
  )
}

export default function EmergentAbilities() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        As language models scale, they exhibit qualitative changes in capability: in-context learning,
        chain-of-thought reasoning, and instruction following appear to emerge at specific scale
        thresholds. Whether these are truly "emergent" or artifacts of metric choice is debated.
      </p>

      <DefinitionBlock title="In-Context Learning (ICL)">
        <p>The ability to learn new tasks from examples provided in the prompt, without gradient updates:</p>
        <BlockMath math="p(y|x, \{(x_1, y_1), \ldots, (x_k, y_k)\})" />
        <p className="mt-2">The model conditions on <InlineMath math="k" /> demonstration examples to predict <InlineMath math="y" /> for a new input <InlineMath math="x" />. This behavior is not explicitly trained — it emerges from next-token prediction on diverse text.</p>
      </DefinitionBlock>

      <EmergenceChart />

      <WarningBlock title="Are Emergent Abilities a Mirage?">
        <p>
          Schaeffer et al. (2023) argue that "emergence" is an artifact of nonlinear metrics like
          exact match. When evaluated with continuous metrics (log-likelihood, Brier score), performance
          scales smoothly. The apparent phase transition comes from thresholding continuous improvement,
          not from a qualitative change in the model.
        </p>
      </WarningBlock>

      <PythonCode
        title="In-Context Learning vs Fine-Tuning"
        code={`import torch
import torch.nn.functional as F

def simulate_icl_performance(model_size_b, num_shots, task_difficulty=0.5):
    """Simulate how ICL performance scales with model size and shots.

    Key finding: ICL performance improves log-linearly with model size
    and log-linearly with number of examples (up to context window).
    """
    # Approximate performance model (from empirical observations)
    import math
    base = 0.5 * math.log10(model_size_b + 1) / 3  # size contribution
    shot_bonus = 0.1 * math.log2(num_shots + 1)      # shot contribution
    perf = min(base + shot_bonus - task_difficulty * 0.3, 1.0)
    return max(perf, 0.0)

# Compare ICL at different scales
for size in [1, 7, 70, 405]:
    scores = [simulate_icl_performance(size, k) for k in [0, 1, 4, 16]]
    print(f"{size:>4}B params | 0-shot: {scores[0]:.2f} | 1-shot: {scores[1]:.2f} "
          f"| 4-shot: {scores[2]:.2f} | 16-shot: {scores[3]:.2f}")`}
      />

      <ExampleBlock title="Chain-of-Thought Reasoning">
        <p>Chain-of-thought (CoT) prompting asks models to show their reasoning steps. It reliably improves performance on multi-step tasks but only works above ~100B parameters:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>GSM8K (math): 17.9% (standard) vs 57.1% (CoT) with PaLM 540B</li>
          <li>No improvement below ~10B parameters — models produce plausible but wrong reasoning</li>
          <li>"Let's think step by step" (zero-shot CoT) works surprisingly well</li>
        </ul>
      </ExampleBlock>

      <NoteBlock type="note" title="Instruction Tuning and RLHF">
        <p>
          Raw language models are poor at following instructions. Instruction tuning (fine-tuning on
          instruction-response pairs) and RLHF (learning from human preferences) dramatically improve
          usability. A 7B instruction-tuned model can outperform a 175B base model on user-facing tasks.
        </p>
      </NoteBlock>
    </div>
  )
}

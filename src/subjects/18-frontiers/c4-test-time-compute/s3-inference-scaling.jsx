import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function InferenceScalingViz() {
  const [logInference, setLogInference] = useState(2)
  const [modelSize, setModelSize] = useState(7)
  const inferenceMultiplier = Math.pow(10, logInference)
  const basePerf = 0.3 + 0.12 * Math.log10(modelSize)
  const scaledPerf = Math.min(0.95, basePerf + 0.08 * Math.log10(inferenceMultiplier))

  const equivalent = modelSize * Math.pow(inferenceMultiplier, 0.5)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Inference Compute Scaling</h3>
      <div className="flex items-center gap-4 mb-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Model: {modelSize}B params
          <input type="range" min={1} max={70} step={1} value={modelSize} onChange={e => setModelSize(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Inference compute: {inferenceMultiplier}x
          <input type="range" min={0} max={4} step={0.5} value={logInference} onChange={e => setLogInference(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <div className="grid grid-cols-3 gap-3 text-sm text-center">
        <div className="p-2 rounded bg-gray-100 dark:bg-gray-800">
          <p className="text-gray-500 font-medium">Base Performance</p>
          <p className="font-bold">{(basePerf * 100).toFixed(1)}%</p>
        </div>
        <div className="p-2 rounded bg-violet-50 dark:bg-violet-900/20">
          <p className="text-violet-700 dark:text-violet-300 font-medium">With {inferenceMultiplier}x Compute</p>
          <p className="font-bold">{(scaledPerf * 100).toFixed(1)}%</p>
        </div>
        <div className="p-2 rounded bg-violet-50 dark:bg-violet-900/20">
          <p className="text-violet-700 dark:text-violet-300 font-medium">Equivalent Dense Model</p>
          <p className="font-bold">~{equivalent.toFixed(0)}B</p>
        </div>
      </div>
    </div>
  )
}

export default function InferenceScalingLaws() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Inference scaling laws describe how model performance improves with additional test-time
        compute. This creates a new dimension for scaling: instead of (or in addition to) training
        larger models, spend more compute at inference to achieve better results.
      </p>

      <DefinitionBlock title="Inference Scaling Law">
        <p>Performance on reasoning tasks follows a power law in inference compute:</p>
        <BlockMath math="P(C_{\text{infer}}) = P_0 + k \cdot C_{\text{infer}}^{\gamma}" />
        <p className="mt-2">where <InlineMath math="C_{\text{infer}}" /> includes tokens generated for reasoning, number of samples, and search compute. Empirically, <InlineMath math="\gamma \approx 0.2\text{-}0.5" /> depending on the task and method. This means 10x more inference compute yields 60-300% relative improvement.</p>
      </DefinitionBlock>

      <InferenceScalingViz />

      <ExampleBlock title="Compute-Equivalent Scaling">
        <p>A small model with extensive search can match a much larger model:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>7B model with 256 samples + PRM matches 70B model with 1 sample on MATH</li>
          <li>Cost comparison: 7B x 256 = 1792B FLOPs vs 70B x 1 = 70B FLOPs (7B wins on accuracy per FLOP)</li>
          <li>Wait — 7B x 256 > 70B? Yes, but the cost is purely sequential generation, highly parallelizable</li>
          <li>Latency vs throughput: parallel BoN has same latency as single sample</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Comparing Training vs Inference Scaling"
        code={`import math

def training_scaled_performance(params_b, task_baseline=0.3):
    """Performance from training scaling (Chinchilla-like)."""
    return task_baseline + 0.12 * math.log10(params_b)

def inference_scaled_performance(params_b, inference_multiplier, task_baseline=0.3):
    """Performance from inference compute scaling."""
    base = training_scaled_performance(params_b, task_baseline)
    bonus = 0.08 * math.log10(max(inference_multiplier, 1))
    return min(0.95, base + bonus)

def total_cost(params_b, inference_multiplier, num_queries=1000):
    """Total cost in relative FLOPs per query."""
    return params_b * inference_multiplier * num_queries

# Find: what's more cost-effective for 80% accuracy?
target = 0.80
print(f"Target accuracy: {target*100:.0f}%")
print(f"{'Strategy':>35} | {'Accuracy':>8} | {'Cost/query':>10}")
print("-" * 60)

strategies = [
    ("70B, 1x inference", 70, 1),
    ("7B, 64x inference (BoN-64)", 7, 64),
    ("7B, 256x inference (BoN-256)", 7, 256),
    ("405B, 1x inference", 405, 1),
    ("70B, 16x inference (BoN-16)", 70, 16),
]

for name, params, mult in strategies:
    perf = inference_scaled_performance(params, mult)
    cost = params * mult
    print(f"{name:>35} | {perf*100:>7.1f}% | {cost:>8.0f}B")

print("\\nKey insight: small model + search can be more cost-effective than large model")`}
      />

      <NoteBlock type="note" title="The Two Scaling Axes">
        <p>
          Deep learning now has two independent scaling axes: <strong>training compute</strong>
          (bigger models, more data) and <strong>inference compute</strong> (longer reasoning,
          more samples, search). The optimal allocation between these axes depends on the use
          case — high-volume, simple tasks favor training scaling, while hard, rare queries
          favor inference scaling. Models like o1 represent a shift toward the inference axis.
        </p>
      </NoteBlock>
    </div>
  )
}

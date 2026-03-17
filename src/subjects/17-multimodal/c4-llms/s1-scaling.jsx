import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

function ScalingCalculator() {
  const [logParams, setLogParams] = useState(10)
  const [logTokens, setLogTokens] = useState(12)
  const params = Math.pow(10, logParams)
  const tokens = Math.pow(10, logTokens)
  const flops = 6 * params * tokens
  const chinchillaOptimalTokens = 20 * params
  const isOverTrained = tokens > chinchillaOptimalTokens * 2
  const isUnderTrained = tokens < chinchillaOptimalTokens * 0.5

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">LLM Training Compute Calculator</h3>
      <div className="flex items-center gap-4 mb-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Parameters: 10^{logParams} ({(params / 1e9).toFixed(1)}B)
          <input type="range" min={8} max={12} step={0.5} value={logParams} onChange={e => setLogParams(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Tokens: 10^{logTokens} ({(tokens / 1e12).toFixed(1)}T)
          <input type="range" min={10} max={14} step={0.5} value={logTokens} onChange={e => setLogTokens(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="p-2 rounded bg-violet-50 dark:bg-violet-900/20 text-center">
          <p className="text-violet-700 dark:text-violet-300 font-medium">Training FLOPs</p>
          <p className="font-bold">{(flops).toExponential(1)}</p>
        </div>
        <div className={`p-2 rounded text-center ${isOverTrained ? 'bg-yellow-50 dark:bg-yellow-900/20' : isUnderTrained ? 'bg-red-50 dark:bg-red-900/20' : 'bg-green-50 dark:bg-green-900/20'}`}>
          <p className="font-medium text-gray-700 dark:text-gray-300">Chinchilla Optimal Tokens</p>
          <p className="font-bold">{(chinchillaOptimalTokens).toExponential(1)}</p>
          <p className="text-xs">{isOverTrained ? 'Over-trained (inference-optimal)' : isUnderTrained ? 'Under-trained' : 'Near optimal'}</p>
        </div>
      </div>
    </div>
  )
}

export default function ScalingTrainingLLMs() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Training large language models requires carefully balancing model size, data volume, and
        compute budget. Scaling laws provide principled guidance for these tradeoffs, while
        practical recipes address the engineering challenges of distributed training.
      </p>

      <TheoremBlock title="Chinchilla Scaling Law" id="chinchilla-scaling">
        <p>For a given compute budget <InlineMath math="C" />, the optimal model size <InlineMath math="N^*" /> and training tokens <InlineMath math="D^*" /> scale equally:</p>
        <BlockMath math="N^* \propto C^{0.5}, \quad D^* \propto C^{0.5}" />
        <p className="mt-2">Rule of thumb: train on <InlineMath math="\sim 20" /> tokens per parameter. A 70B model should see ~1.4T tokens. The training FLOPs are approximately:</p>
        <BlockMath math="C \approx 6ND" />
      </TheoremBlock>

      <ScalingCalculator />

      <ExampleBlock title="Notable LLM Training Configurations">
        <ul className="list-disc list-inside space-y-1">
          <li>GPT-3 (175B): 300B tokens, ~3.6e23 FLOPs (under-trained by Chinchilla standards)</li>
          <li>Chinchilla (70B): 1.4T tokens, ~5.8e23 FLOPs (compute-optimal)</li>
          <li>LLaMA-2 70B: 2T tokens, ~8.4e23 FLOPs (over-trained for better inference)</li>
          <li>LLaMA-3 70B: 15T tokens — heavily over-trained for inference efficiency</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Estimating LLM Training Cost"
        code={`import math

def estimate_training(params_b, tokens_t, gpu_tflops=312, gpu_util=0.4):
    """Estimate LLM training requirements.

    Args:
        params_b: parameters in billions
        tokens_t: training tokens in trillions
        gpu_tflops: peak GPU TFLOPS (H100 = 989 BF16, A100 = 312)
        gpu_util: model FLOPs utilization (typically 0.3-0.5)
    """
    params = params_b * 1e9
    tokens = tokens_t * 1e12
    flops = 6 * params * tokens

    # GPU-hours
    effective_tflops = gpu_tflops * gpu_util * 1e12
    gpu_seconds = flops / effective_tflops
    gpu_hours = gpu_seconds / 3600

    # Chinchilla optimal tokens
    optimal_tokens = 20 * params

    print(f"Model: {params_b}B params, {tokens_t}T tokens")
    print(f"Training FLOPs: {flops:.2e}")
    print(f"GPU-hours (single GPU): {gpu_hours:,.0f}")
    print(f"With 1024 GPUs: {gpu_hours/1024:,.0f} hours = {gpu_hours/1024/24:,.0f} days")
    print(f"Chinchilla optimal tokens: {optimal_tokens/1e12:.1f}T")
    print(f"Token ratio: {tokens/optimal_tokens:.1f}x optimal")

estimate_training(70, 2.0, gpu_tflops=312)  # LLaMA-2 70B on A100
print()
estimate_training(70, 15.0, gpu_tflops=989)  # LLaMA-3 70B on H100`}
      />

      <NoteBlock type="note" title="Beyond Compute-Optimal Training">
        <p>
          Modern LLMs are deliberately over-trained relative to Chinchilla optimality. Since
          inference cost scales with model size (not training tokens), it is cheaper to train
          a smaller model on more data than to train a larger compute-optimal model. LLaMA-3
          trained a 70B model on 15T tokens — 10x more than Chinchilla would recommend.
        </p>
      </NoteBlock>
    </div>
  )
}

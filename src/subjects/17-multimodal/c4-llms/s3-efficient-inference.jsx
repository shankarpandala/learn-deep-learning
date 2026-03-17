import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function QuantizationExplorer() {
  const [bits, setBits] = useState(16)
  const modelParams = 70
  const memoryGB = modelParams * bits / 8
  const fp16Memory = modelParams * 2
  const savings = ((1 - memoryGB / fp16Memory) * 100).toFixed(1)
  const qualityLoss = bits >= 8 ? 'Negligible' : bits >= 4 ? 'Minor (~1% accuracy drop)' : 'Significant'

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">LLM Quantization Memory Calculator</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Quantization bits: {bits}
        <input type="range" min={2} max={16} step={1} value={bits} onChange={e => setBits(parseInt(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <div className="grid grid-cols-3 gap-3 text-sm text-center">
        <div className="p-2 rounded bg-violet-50 dark:bg-violet-900/20">
          <p className="text-violet-700 dark:text-violet-300 font-medium">70B Model Memory</p>
          <p className="text-lg font-bold">{memoryGB.toFixed(1)} GB</p>
        </div>
        <div className="p-2 rounded bg-violet-50 dark:bg-violet-900/20">
          <p className="text-violet-700 dark:text-violet-300 font-medium">Savings vs FP16</p>
          <p className="text-lg font-bold">{savings}%</p>
        </div>
        <div className="p-2 rounded bg-violet-50 dark:bg-violet-900/20">
          <p className="text-violet-700 dark:text-violet-300 font-medium">Quality Impact</p>
          <p className="text-sm font-bold">{qualityLoss}</p>
        </div>
      </div>
      <div className="mt-2 h-4 bg-gray-100 dark:bg-gray-800 rounded overflow-hidden">
        <div className="h-full bg-violet-500 transition-all duration-200" style={{ width: `${(memoryGB / fp16Memory) * 100}%` }} />
      </div>
      <p className="text-xs text-gray-500 mt-1 text-center">{memoryGB.toFixed(1)} GB / {fp16Memory} GB (FP16 baseline)</p>
    </div>
  )
}

export default function EfficientInference() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        LLM inference is memory-bandwidth bound during autoregressive decoding. Techniques like
        quantization, KV-cache optimization, and speculative decoding dramatically reduce cost
        while preserving quality.
      </p>

      <DefinitionBlock title="KV-Cache and Memory Bandwidth Bottleneck">
        <p>During autoregressive generation, each token requires reading all model weights and the KV-cache:</p>
        <BlockMath math="\text{KV cache size} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times \text{seq\_len} \times \text{bytes}" />
        <p className="mt-2">For LLaMA-2 70B with 4K context: KV cache = <InlineMath math="2 \times 80 \times 64 \times 128 \times 4096 \times 2 \approx 10.7" /> GB in FP16. The arithmetic intensity is just ~1 FLOP/byte, making inference memory-bound.</p>
      </DefinitionBlock>

      <QuantizationExplorer />

      <ExampleBlock title="Speculative Decoding">
        <p>Use a small draft model to generate <InlineMath math="K" /> candidate tokens, then verify all at once with the large model:</p>
        <BlockMath math="\text{Speedup} \approx \frac{K}{1 + (1-\alpha)K} \quad \text{where } \alpha = P(\text{draft accepted})" />
        <p className="mt-2">With acceptance rate <InlineMath math="\alpha = 0.8" /> and <InlineMath math="K = 5" /> draft tokens: ~2.5x speedup with <strong>zero quality loss</strong> (mathematically equivalent to sampling from the large model).</p>
      </ExampleBlock>

      <PythonCode
        title="Post-Training Quantization (Simplified)"
        code={`import torch

def absmax_quantize(tensor, bits=8):
    """Simple absmax quantization to int8/int4."""
    qmax = 2**(bits - 1) - 1
    scale = tensor.abs().max() / qmax
    quantized = (tensor / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return quantized, scale

def dequantize(quantized, scale):
    return quantized.float() * scale

# Simulate quantizing a weight matrix
W = torch.randn(4096, 4096)  # typical LLM layer
W_q, scale = absmax_quantize(W, bits=8)
W_deq = dequantize(W_q, scale)

# Measure quantization error
error = (W - W_deq).abs().mean()
print(f"Original dtype: {W.dtype}, size: {W.numel() * 4 / 1e6:.1f} MB")
print(f"Quantized dtype: {W_q.dtype}, size: {W_q.numel() * 1 / 1e6:.1f} MB")
print(f"Mean absolute error: {error:.6f}")
print(f"Relative error: {(error / W.abs().mean() * 100):.2f}%")
print(f"Memory savings: {(1 - 1/4) * 100:.0f}%")`}
      />

      <NoteBlock type="note" title="Grouped Query Attention (GQA)">
        <p>
          GQA reduces KV-cache size by sharing key-value heads across multiple query heads.
          LLaMA-2 70B uses 8 KV-heads shared across 64 query heads, reducing KV-cache by 8x.
          Multi-Query Attention (MQA) takes this further with a single KV-head, though GQA
          offers a better quality-efficiency tradeoff.
        </p>
      </NoteBlock>

      <ExampleBlock title="GPTQ and AWQ: Advanced Quantization">
        <p>Modern quantization methods go beyond simple absmax:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li><strong>GPTQ:</strong> Layer-wise quantization minimizing reconstruction error using Hessian information</li>
          <li><strong>AWQ:</strong> Activation-aware quantization that protects salient weight channels</li>
          <li><strong>GGUF:</strong> File format for CPU-friendly quantized inference (llama.cpp)</li>
          <li>4-bit GPTQ on LLaMA-2 70B: only ~1% accuracy drop, fits on a single 48GB GPU</li>
          <li><strong>FP8:</strong> Native 8-bit floating point supported on H100 GPUs for near-lossless inference</li>
        </ul>
        <p className="mt-2">
          The trend toward lower precision continues: 2-bit and 1.58-bit (ternary) quantization
          are active research areas with promising early results.
        </p>
      </ExampleBlock>
    </div>
  )
}

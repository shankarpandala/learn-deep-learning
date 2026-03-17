import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function AttentionWeightViz() {
  const [temperature, setTemperature] = useState(1.0)
  const labels = ['I', 'love', 'deep', 'learning']
  const rawScores = [0.8, 2.5, 1.2, 3.1]

  function softmax(scores, temp) {
    const scaled = scores.map(s => s / temp)
    const maxS = Math.max(...scaled)
    const exps = scaled.map(s => Math.exp(s - maxS))
    const sum = exps.reduce((a, b) => a + b, 0)
    return exps.map(e => e / sum)
  }

  const weights = softmax(rawScores, temperature)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Attention Weight Visualization</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-4">
        Temperature: {temperature.toFixed(2)}
        <input type="range" min={0.1} max={3.0} step={0.05} value={temperature} onChange={e => setTemperature(parseFloat(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <div className="flex gap-3 justify-center">
        {labels.map((lbl, i) => (
          <div key={lbl} className="flex flex-col items-center gap-1">
            <div className="w-16 rounded" style={{ height: `${Math.max(4, weights[i] * 120)}px`, backgroundColor: `rgba(139, 92, 246, ${0.3 + weights[i] * 0.7})` }} />
            <span className="text-xs font-mono text-gray-700 dark:text-gray-300">{lbl}</span>
            <span className="text-xs text-violet-600 dark:text-violet-400">{weights[i].toFixed(3)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function QKV() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The attention mechanism allows a model to dynamically focus on different parts of the input
        when producing each output element. The Query-Key-Value (QKV) framework provides an elegant
        formulation for computing these attention weights.
      </p>

      <DefinitionBlock title="Queries, Keys, and Values">
        <p>Given input embeddings <InlineMath math="X \in \mathbb{R}^{n \times d}" />, we project into three spaces:</p>
        <BlockMath math="Q = XW^Q, \quad K = XW^K, \quad V = XW^V" />
        <p className="mt-2">
          <strong>Query</strong> — what am I looking for? <strong>Key</strong> — what do I contain?
          <strong> Value</strong> — what information do I provide?
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="Scaled Dot-Product Attention">
        <BlockMath math="\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V" />
        <p className="mt-2">
          The scaling factor <InlineMath math="\sqrt{d_k}" /> prevents the dot products from growing large
          in magnitude, which would push the softmax into regions with extremely small gradients.
        </p>
      </DefinitionBlock>

      <AttentionWeightViz />

      <ExampleBlock title="Why Scale by sqrt(d_k)?">
        <p>
          If <InlineMath math="q, k \in \mathbb{R}^{d_k}" /> have components drawn i.i.d. from <InlineMath math="\mathcal{N}(0,1)" />, then:
        </p>
        <BlockMath math="\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k" />
        <p>Dividing by <InlineMath math="\sqrt{d_k}" /> normalizes the variance back to 1.</p>
      </ExampleBlock>

      <PythonCode
        title="Scaled Dot-Product Attention in PyTorch"
        code={`import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Compute scaled dot-product attention.

    Args:
        Q: (batch, seq_q, d_k)
        K: (batch, seq_k, d_k)
        V: (batch, seq_k, d_v)
        mask: optional (batch, seq_q, seq_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights

# Example: batch=1, seq_len=4, d_k=d_v=8
Q = torch.randn(1, 4, 8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)
output, attn_weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")       # (1, 4, 8)
print(f"Attention weights:\\n{attn_weights}")`}
      />

      <NoteBlock type="note" title="Attention as Soft Dictionary Lookup">
        <p>
          Think of attention as a differentiable dictionary. The query looks up the most relevant
          keys, and the returned value is a weighted combination of all values. Unlike a hard
          lookup, every entry contributes — just with different weights determined by the
          query-key similarity.
        </p>
      </NoteBlock>
    </div>
  )
}

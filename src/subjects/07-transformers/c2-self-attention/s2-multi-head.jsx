import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

function MultiHeadDiagram() {
  const [numHeads, setNumHeads] = useState(4)
  const dModel = 256
  const dHead = dModel / numHeads
  const headColors = ['#8b5cf6', '#a78bfa', '#c4b5fd', '#7c3aed', '#6d28d9', '#5b21b6', '#4c1d95', '#ddd6fe']

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Multi-Head Attention Structure</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Number of heads: {numHeads}
        <input type="range" min={1} max={8} step={1} value={numHeads} onChange={e => setNumHeads(parseInt(e.target.value))} className="w-32 accent-violet-500" />
      </label>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
        <InlineMath math={`d_{\\text{model}} = ${dModel}`} />, each head: <InlineMath math={`d_k = d_v = ${dHead}`} />
      </p>
      <div className="flex gap-1 justify-center flex-wrap">
        {Array.from({ length: numHeads }, (_, i) => (
          <div key={i} className="flex flex-col items-center rounded-lg p-2 border border-gray-200 dark:border-gray-700" style={{ backgroundColor: headColors[i] + '20' }}>
            <div className="text-xs font-bold mb-1" style={{ color: headColors[i] }}>Head {i + 1}</div>
            <div className="text-xs text-gray-500 dark:text-gray-400">Q K V</div>
            <div className="text-xs text-gray-500 dark:text-gray-400">{dHead}d</div>
          </div>
        ))}
      </div>
      <div className="text-center mt-3 text-sm text-gray-600 dark:text-gray-400">
        Concat all heads &rarr; Linear projection &rarr; <InlineMath math={`\\mathbb{R}^{${dModel}}`} />
      </div>
    </div>
  )
}

export default function MultiHeadAttention() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Multi-head attention runs several attention functions in parallel, allowing the model
        to jointly attend to information from different representation subspaces at different
        positions. A single attention head tends to average over multiple patterns — multiple
        heads allow specialization.
      </p>

      <DefinitionBlock title="Multi-Head Attention">
        <BlockMath math="\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O" />
        <BlockMath math="\text{where head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)" />
        <p className="mt-2">
          With <InlineMath math="h" /> heads and model dimension <InlineMath math="d_{\text{model}}" />,
          each head operates on <InlineMath math="d_k = d_v = d_{\text{model}} / h" />.
        </p>
      </DefinitionBlock>

      <MultiHeadDiagram />

      <TheoremBlock title="Parameter Count" id="mha-params">
        <p>Multi-head attention with <InlineMath math="h" /> heads has the same parameter count as single-head:</p>
        <BlockMath math="3 \cdot d_{\text{model}} \cdot d_k \cdot h + d_{\text{model}}^2 = 4 \cdot d_{\text{model}}^2" />
        <p className="mt-2">
          Since <InlineMath math="d_k = d_{\text{model}} / h" />, the total computation is the same as
          single-head attention with full dimensionality, but with richer representations.
        </p>
      </TheoremBlock>

      <ExampleBlock title="What Different Heads Learn">
        <p>
          Research shows heads specialize: some track syntactic relations (subject-verb), others
          coreference (pronoun-antecedent), and others positional patterns (attending to
          adjacent tokens). This diversity is key to the transformer's representational power.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Multi-Head Attention Implementation"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        qkv = self.W_qkv(x).reshape(B, N, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, h, N, d_k)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, V)  # (B, h, N, d_k)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.W_o(out)

mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)
print(f"Output: {mha(x).shape}")  # (2, 10, 512)
print(f"Params: {sum(p.numel() for p in mha.parameters()):,}")`}
      />

      <NoteBlock type="note" title="PyTorch Built-in">
        <p>
          In practice, use <code>torch.nn.MultiheadAttention</code> which implements fused
          multi-head attention with optimized kernels. The manual implementation above is for
          pedagogical clarity.
        </p>
      </NoteBlock>
    </div>
  )
}

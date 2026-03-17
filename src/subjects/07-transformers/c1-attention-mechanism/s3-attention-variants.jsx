import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

function ComparisonToggle() {
  const [variant, setVariant] = useState('multiplicative')

  const info = {
    additive: {
      name: 'Additive (Bahdanau)',
      formula: 'score(s_i, h_j) = v^\\top \\tanh(W_1 s_i + W_2 h_j)',
      pros: ['More expressive with learnable parameters', 'Works well with different Q/K dimensions'],
      cons: ['Slower — requires feed-forward pass per pair', 'More parameters to train'],
    },
    multiplicative: {
      name: 'Multiplicative (Luong)',
      formula: 'score(s_i, h_j) = s_i^\\top W h_j',
      pros: ['Efficient — single matrix multiply', 'Easily batched on GPUs'],
      cons: ['Assumes Q and K have compatible dimensions', 'Can suffer from large dot products without scaling'],
    },
  }

  const cur = info[variant]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Attention Variant Comparison</h3>
      <div className="flex gap-3 mb-4">
        {Object.keys(info).map(k => (
          <button key={k} onClick={() => setVariant(k)} className={`px-3 py-1.5 rounded-lg text-sm font-medium transition ${variant === k ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {info[k].name}
          </button>
        ))}
      </div>
      <BlockMath math={cur.formula} />
      <div className="grid grid-cols-2 gap-4 mt-3 text-sm">
        <div>
          <p className="font-semibold text-green-600 dark:text-green-400 mb-1">Advantages</p>
          <ul className="list-disc ml-4 text-gray-600 dark:text-gray-400">
            {cur.pros.map((p, i) => <li key={i}>{p}</li>)}
          </ul>
        </div>
        <div>
          <p className="font-semibold text-red-500 dark:text-red-400 mb-1">Disadvantages</p>
          <ul className="list-disc ml-4 text-gray-600 dark:text-gray-400">
            {cur.cons.map((c, i) => <li key={i}>{c}</li>)}
          </ul>
        </div>
      </div>
    </div>
  )
}

export default function AttentionVariants() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Before the modern transformer, two major attention variants emerged from the sequence-to-sequence
        literature: additive attention (Bahdanau et al., 2015) and multiplicative attention (Luong et al., 2015).
        Understanding their differences illuminates key design choices in attention mechanisms.
      </p>

      <DefinitionBlock title="Additive Attention (Bahdanau)">
        <BlockMath math="e_{ij} = v^\top \tanh(W_1 s_i + W_2 h_j)" />
        <p className="mt-2">
          Uses a learned feed-forward network with weight matrices <InlineMath math="W_1, W_2" /> and
          vector <InlineMath math="v" /> to compute alignment scores between decoder state
          <InlineMath math="s_i" /> and encoder hidden state <InlineMath math="h_j" />.
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="Multiplicative Attention (Luong)">
        <BlockMath math="e_{ij} = s_i^\top W h_j \quad \text{(general)} \quad \text{or} \quad e_{ij} = s_i^\top h_j \quad \text{(dot)}" />
        <p className="mt-2">
          The dot-product variant (without <InlineMath math="W" />) becomes scaled dot-product
          attention when divided by <InlineMath math="\sqrt{d_k}" /> — the basis of the Transformer.
        </p>
      </DefinitionBlock>

      <ComparisonToggle />

      <TheoremBlock title="Computational Complexity" id="attention-complexity">
        <p>For sequence length <InlineMath math="n" /> and dimension <InlineMath math="d" />:</p>
        <BlockMath math="\text{Additive: } O(n^2 \cdot d) \quad \text{Multiplicative: } O(n^2 \cdot d)" />
        <p className="mt-2">
          Both are <InlineMath math="O(n^2)" /> in sequence length, but multiplicative attention has
          a much smaller constant factor due to optimized matrix multiplication on modern hardware.
        </p>
      </TheoremBlock>

      <PythonCode
        title="Additive vs Multiplicative Attention"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W1 = nn.Linear(dim, dim, bias=False)
        self.W2 = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, query, keys, values):
        # query: (B, 1, D), keys: (B, N, D)
        scores = self.v(torch.tanh(self.W1(query) + self.W2(keys)))
        weights = F.softmax(scores.squeeze(-1), dim=-1)
        return torch.bmm(weights.unsqueeze(1), values)

class MultiplicativeAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5

    def forward(self, query, keys, values):
        scores = torch.bmm(query, keys.transpose(1, 2)) / self.scale
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, values)

d = 64
additive = AdditiveAttention(d)
multiplicative = MultiplicativeAttention(d)
q = torch.randn(2, 1, d)
k = v = torch.randn(2, 10, d)
print("Additive output:", additive(q, k, v).shape)
print("Multiplicative output:", multiplicative(q, k, v).shape)`}
      />

      <NoteBlock type="note" title="Why Transformers Use Multiplicative Attention">
        <p>
          The Transformer adopts scaled dot-product (multiplicative) attention because it can be
          computed entirely with matrix multiplications, which are highly optimized on GPUs. The
          scaling factor addresses the gradient issues of large dot products, making it both
          fast and numerically stable.
        </p>
      </NoteBlock>

      <ExampleBlock title="Historical Timeline">
        <p>
          <strong>2015:</strong> Bahdanau introduces additive attention for machine translation.{' '}
          <strong>2015:</strong> Luong proposes multiplicative variants.{' '}
          <strong>2017:</strong> Vaswani et al. use scaled dot-product attention in the Transformer,
          dispensing with recurrence entirely.
        </p>
      </ExampleBlock>
    </div>
  )
}

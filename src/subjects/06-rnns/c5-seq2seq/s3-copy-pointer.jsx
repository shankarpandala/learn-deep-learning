import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

function CopyDemo() {
  const [pGen, setPGen] = useState(0.3)
  const srcTokens = ['John', 'went', 'to', 'New', 'York']
  const vocabProbs = { 'he': 0.3, 'went': 0.2, 'to': 0.15, 'the': 0.1, 'a': 0.05 }
  const copyProbs = { 'John': 0.5, 'went': 0.1, 'to': 0.1, 'New': 0.2, 'York': 0.1 }

  const finalProbs = {}
  for (const [k, v] of Object.entries(vocabProbs)) {
    finalProbs[k] = (finalProbs[k] || 0) + pGen * v
  }
  for (const [k, v] of Object.entries(copyProbs)) {
    finalProbs[k] = (finalProbs[k] || 0) + (1 - pGen) * v
  }

  const sorted = Object.entries(finalProbs).sort((a, b) => b[1] - a[1])

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Copy Mechanism in Action</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        p_gen: {pGen.toFixed(2)} (generate)  |  1-p_gen: {(1 - pGen).toFixed(2)} (copy)
        <input type="range" min={0} max={1} step={0.05} value={pGen} onChange={e => setPGen(parseFloat(e.target.value))} className="w-32 accent-violet-500" />
      </label>
      <div className="flex gap-2 mb-3 flex-wrap">
        <span className="text-xs text-gray-500">Source:</span>
        {srcTokens.map((t, i) => (
          <span key={i} className="px-2 py-0.5 rounded bg-violet-100 dark:bg-violet-900/40 text-violet-700 dark:text-violet-300 text-xs font-mono">{t}</span>
        ))}
      </div>
      <div className="space-y-1">
        {sorted.slice(0, 6).map(([token, prob]) => (
          <div key={token} className="flex items-center gap-2">
            <span className="w-14 text-xs font-mono text-gray-600 dark:text-gray-400 text-right">{token}</span>
            <div className="flex-1 bg-gray-100 dark:bg-gray-800 rounded h-4 overflow-hidden">
              <div className="h-full bg-violet-500 rounded" style={{ width: `${prob * 100}%` }} />
            </div>
            <span className="text-xs text-gray-500 w-10 text-right">{prob.toFixed(3)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function CopyPointer() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Copy mechanisms and pointer networks extend seq2seq models to handle rare words,
        named entities, and structured outputs by allowing the decoder to copy tokens
        directly from the source sequence.
      </p>

      <DefinitionBlock title="Pointer Network">
        <p>
          A pointer network (Vinyals et al., 2015) uses attention as a pointer to select
          elements from the input sequence:
        </p>
        <BlockMath math="P(y_t = x_i) = \alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}" />
        <p className="mt-2">
          Unlike standard seq2seq which outputs from a fixed vocabulary, pointer networks can
          output any element from the variable-length input, making them ideal for combinatorial
          optimization problems (e.g., sorting, convex hull, TSP).
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="Copy Mechanism (Pointer-Generator)">
        <p>
          The pointer-generator network (See et al., 2017) combines generation and copying
          using a soft switch <InlineMath math="p_{\text{gen}}" />:
        </p>
        <BlockMath math="p_{\text{gen}} = \sigma(w_c^T c_t + w_s^T s_t + w_x^T x_t + b)" />
        <BlockMath math="P(w) = p_{\text{gen}} \, P_{\text{vocab}}(w) + (1 - p_{\text{gen}}) \sum_{i: x_i = w} \alpha_{t,i}" />
        <p className="mt-2">
          This allows the model to generate from the vocabulary or copy from the source
          via the attention distribution.
        </p>
      </DefinitionBlock>

      <CopyDemo />

      <ExampleBlock title="Use Cases">
        <p>Copy mechanisms are critical for:</p>
        <ul className="list-disc ml-6 mt-1 space-y-1">
          <li><strong>Summarization</strong>: copying factual details (names, numbers) from the article</li>
          <li><strong>Dialogue</strong>: repeating entities mentioned by the user</li>
          <li><strong>Code generation</strong>: copying variable names from the context</li>
          <li><strong>Data-to-text</strong>: faithfully reproducing values from structured data</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Pointer-Generator Network"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.vocab_proj = nn.Linear(hidden_dim, vocab_size)
        self.p_gen_linear = nn.Linear(hidden_dim * 2 + hidden_dim, 1)

    def forward(self, dec_state, enc_outputs, src_ids):
        # dec_state: (B, H), enc_outputs: (B, S, H), src_ids: (B, S)
        B, S, H = enc_outputs.shape

        # Attention
        dec_exp = dec_state.unsqueeze(1).expand(-1, S, -1)
        scores = self.attn(torch.cat([dec_exp, enc_outputs], -1)).squeeze(-1)
        attn_weights = F.softmax(scores, dim=-1)  # (B, S)

        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)

        # p_gen switch
        p_gen = torch.sigmoid(self.p_gen_linear(
            torch.cat([context, dec_state, dec_state], -1)  # simplified
        ))  # (B, 1)

        # Vocab distribution
        vocab_dist = F.softmax(self.vocab_proj(dec_state), dim=-1) * p_gen

        # Copy distribution
        copy_dist = torch.zeros(B, self.vocab_size, device=dec_state.device)
        copy_dist.scatter_add_(1, src_ids, attn_weights * (1 - p_gen))

        return vocab_dist + copy_dist, attn_weights

model = PointerGenerator(vocab_size=10000, hidden_dim=256)
dec_h = torch.randn(4, 256)
enc_out = torch.randn(4, 20, 256)
src = torch.randint(0, 10000, (4, 20))
probs, attn = model(dec_h, enc_out, src)
print(f"Output dist: {probs.shape}")  # (4, 10000)
print(f"Sum: {probs.sum(-1)}")  # ~1.0`}
      />

      <TheoremBlock title="Coverage Mechanism" id="coverage-mechanism">
        <p>
          To prevent repetition in generation, a coverage vector tracks cumulative attention:
        </p>
        <BlockMath math="\text{cov}_t = \sum_{t'=0}^{t-1} \alpha_{t'}" />
        <p>A coverage loss penalizes re-attending to already-covered positions:</p>
        <BlockMath math="\mathcal{L}_{\text{cov}} = \sum_i \min(\alpha_{t,i}, \text{cov}_{t,i})" />
      </TheoremBlock>

      <NoteBlock type="note" title="Legacy and Modern Influence">
        <p>
          While Transformers have largely replaced RNN-based seq2seq, the copy mechanism concept
          lives on in retrieval-augmented generation and tool-use paradigms in large language
          models. The idea of selectively copying from a source remains fundamental in
          modern NLP architectures.
        </p>
      </NoteBlock>
    </div>
  )
}

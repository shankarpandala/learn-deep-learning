import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function CrossAttentionDiagram() {
  const [decoderIdx, setDecoderIdx] = useState(0)
  const encoder = ['Le', 'chat', 'est', 'noir']
  const decoder = ['The', 'cat', 'is']
  const weights = [
    [0.55, 0.15, 0.20, 0.10],
    [0.10, 0.60, 0.15, 0.15],
    [0.15, 0.10, 0.55, 0.20],
  ]

  const row = weights[decoderIdx]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Cross-Attention: Decoder attends to Encoder</h3>
      <div className="flex gap-2 mb-4 mt-3">
        {decoder.map((t, i) => (
          <button key={i} onClick={() => setDecoderIdx(i)} className={`px-3 py-1 rounded-lg text-sm font-medium transition ${decoderIdx === i ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {t}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-6">
        <div>
          <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2">Decoder (Query)</p>
          <div className="flex gap-2">
            {decoder.map((t, i) => (
              <span key={i} className={`px-2 py-1 rounded text-sm font-mono ${i === decoderIdx ? 'bg-violet-100 text-violet-700 dark:bg-violet-900 dark:text-violet-300 font-bold' : 'text-gray-500 dark:text-gray-400'}`}>{t}</span>
            ))}
          </div>
        </div>
        <div>
          <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2">Encoder (Key/Value)</p>
          <div className="flex gap-2">
            {encoder.map((t, i) => (
              <div key={i} className="flex flex-col items-center gap-1">
                <div className="w-12 rounded" style={{ height: `${Math.max(4, row[i] * 80)}px`, backgroundColor: `rgba(139, 92, 246, ${0.2 + row[i] * 0.8})` }} />
                <span className="text-xs font-mono text-gray-700 dark:text-gray-300">{t}</span>
                <span className="text-xs text-violet-600 dark:text-violet-400">{row[i].toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default function CrossAttention() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Cross-attention bridges two different sequences — typically the encoder output and the decoder
        input. The decoder generates queries from its own representation, while the keys and values come
        from the encoder, allowing each decoder position to attend to the full source sequence.
      </p>

      <DefinitionBlock title="Cross-Attention">
        <p>Given encoder output <InlineMath math="H^{enc}" /> and decoder hidden states <InlineMath math="H^{dec}" />:</p>
        <BlockMath math="Q = H^{dec} W^Q, \quad K = H^{enc} W^K, \quad V = H^{enc} W^V" />
        <BlockMath math="\text{CrossAttn} = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V" />
      </DefinitionBlock>

      <CrossAttentionDiagram />

      <ExampleBlock title="Translation with Cross-Attention">
        <p>
          When translating &quot;Le chat est noir&quot; to &quot;The cat is black&quot;, cross-attention helps each
          decoder token align with its source counterpart. &quot;The&quot; primarily attends to &quot;Le&quot;,
          &quot;cat&quot; attends to &quot;chat&quot;, and so on — learning soft alignments without explicit
          alignment supervision.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Cross-Attention Module"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, decoder_hidden, encoder_output):
        B, Nq, D = decoder_hidden.shape
        Nk = encoder_output.shape[1]
        h, dk = self.num_heads, self.d_k

        Q = self.W_q(decoder_hidden).view(B, Nq, h, dk).transpose(1, 2)
        K = self.W_k(encoder_output).view(B, Nk, h, dk).transpose(1, 2)
        V = self.W_v(encoder_output).view(B, Nk, h, dk).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (dk ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, Nq, D)
        return self.W_o(out)

# Encoder output: 10 source tokens; decoder: 6 target tokens
enc_out = torch.randn(2, 10, 256)
dec_hidden = torch.randn(2, 6, 256)
cross_attn = CrossAttention(d_model=256, num_heads=8)
output = cross_attn(dec_hidden, enc_out)
print(f"Output: {output.shape}")  # (2, 6, 256)`}
      />

      <WarningBlock title="Encoder KV Caching">
        <p>
          Since encoder outputs do not change during decoding, the K and V projections from the
          encoder can be computed once and cached. Recomputing them at each decoder step is a
          common source of unnecessary overhead in naive implementations.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Beyond Seq2Seq">
        <p>
          Cross-attention appears in many architectures beyond translation: vision transformers
          use it to combine image patches with text queries (CLIP), diffusion models use it
          to condition on text prompts, and retrieval-augmented models use it to attend over
          retrieved documents.
        </p>
      </NoteBlock>
    </div>
  )
}

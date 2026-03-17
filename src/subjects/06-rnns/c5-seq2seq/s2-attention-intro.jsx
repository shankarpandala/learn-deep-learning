import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

function AttentionHeatmap() {
  const [attnType, setAttnType] = useState('bahdanau')
  const srcTokens = ['le', 'chat', 'noir', 'dort']
  const tgtTokens = ['the', 'black', 'cat', 'sleeps']

  const bahdanauWeights = [
    [0.85, 0.05, 0.05, 0.05],
    [0.05, 0.10, 0.80, 0.05],
    [0.05, 0.75, 0.10, 0.10],
    [0.05, 0.05, 0.05, 0.85],
  ]
  const luongWeights = [
    [0.80, 0.10, 0.05, 0.05],
    [0.05, 0.08, 0.82, 0.05],
    [0.08, 0.72, 0.12, 0.08],
    [0.03, 0.05, 0.07, 0.85],
  ]
  const weights = attnType === 'bahdanau' ? bahdanauWeights : luongWeights

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Attention Alignment</h3>
      <div className="flex gap-2 mb-3">
        <button onClick={() => setAttnType('bahdanau')}
          className={`px-3 py-1 rounded-lg text-sm ${attnType === 'bahdanau' ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
          Bahdanau (additive)
        </button>
        <button onClick={() => setAttnType('luong')}
          className={`px-3 py-1 rounded-lg text-sm ${attnType === 'luong' ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
          Luong (multiplicative)
        </button>
      </div>
      <div className="flex justify-center">
        <div className="inline-grid gap-0.5" style={{ gridTemplateColumns: `60px repeat(${srcTokens.length}, 48px)` }}>
          <div />
          {srcTokens.map((t, i) => <div key={i} className="text-center text-xs text-gray-500 font-mono">{t}</div>)}
          {tgtTokens.map((t, row) => (
            <>
              <div key={`l${row}`} className="text-right pr-2 text-xs text-gray-500 font-mono leading-8">{t}</div>
              {weights[row].map((w, col) => (
                <div key={`${row}-${col}`} className="w-12 h-8 rounded flex items-center justify-center text-xs font-mono"
                  style={{ backgroundColor: `rgba(139, 92, 246, ${w})`, color: w > 0.5 ? 'white' : '#6b7280' }}>
                  {w.toFixed(2)}
                </div>
              ))}
            </>
          ))}
        </div>
      </div>
      <p className="text-xs text-center mt-2 text-gray-500">Source (columns) vs Target (rows). Darker = higher attention weight.</p>
    </div>
  )
}

export default function AttentionIntro() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Attention mechanisms allow the decoder to dynamically focus on different parts of the
        encoder output at each decoding step, overcoming the information bottleneck of
        fixed-size context vectors.
      </p>

      <DefinitionBlock title="Bahdanau Attention (Additive)">
        <p>The alignment score uses a learned feedforward network:</p>
        <BlockMath math="e_{t,i} = v^T \tanh(W_s\, s_{t-1} + W_h\, h_i)" />
        <BlockMath math="\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}" />
        <BlockMath math="c_t = \sum_i \alpha_{t,i}\, h_i" />
        <p className="mt-2">
          where <InlineMath math="s_{t-1}" /> is the decoder state and <InlineMath math="h_i" /> are encoder outputs.
          The context <InlineMath math="c_t" /> changes at every decoder step.
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="Luong Attention (Multiplicative)">
        <p>Luong attention computes alignment with a simpler dot product or bilinear form:</p>
        <BlockMath math="e_{t,i} = s_t^T W_a\, h_i \quad \text{(general)}" />
        <BlockMath math="e_{t,i} = s_t^T h_i \quad \text{(dot)}" />
        <p className="mt-2">
          Luong attention uses the current decoder state <InlineMath math="s_t" /> (not <InlineMath math="s_{t-1}" />),
          and applies attention after the decoder RNN step rather than before.
        </p>
      </DefinitionBlock>

      <AttentionHeatmap />

      <TheoremBlock title="Attention as Soft Alignment" id="soft-alignment">
        <p>
          The attention weights <InlineMath math="\alpha_{t,i}" /> form a probability distribution over
          source positions. This acts as a differentiable, soft version of word alignment used
          in statistical MT. The model learns these alignments end-to-end without explicit
          alignment supervision.
        </p>
      </TheoremBlock>

      <PythonCode
        title="Bahdanau Attention Module"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs):
        # decoder_state: (B, 1, H), encoder_outputs: (B, S, H)
        scores = self.v(torch.tanh(
            self.W_s(decoder_state) + self.W_h(encoder_outputs)
        ))  # (B, S, 1)
        weights = F.softmax(scores.squeeze(-1), dim=-1)  # (B, S)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)  # (B, 1, H)
        return context, weights

attn = BahdanauAttention(256)
dec_state = torch.randn(4, 1, 256)
enc_out = torch.randn(4, 20, 256)
ctx, weights = attn(dec_state, enc_out)
print(f"Context: {ctx.shape}")    # (4, 1, 256)
print(f"Weights: {weights.shape}")  # (4, 20)
print(f"Weights sum: {weights.sum(-1)}")  # all 1.0`}
      />

      <NoteBlock type="note" title="Impact of Attention">
        <p>
          Attention improved BLEU scores on WMT translation by 2-5 points and enabled training
          on longer sentences. It also provides interpretability through attention weight
          visualization. The concept of attention evolved into the <strong>self-attention</strong> mechanism
          at the core of Transformers, which apply attention within a single sequence rather
          than across encoder-decoder pairs.
        </p>
      </NoteBlock>
    </div>
  )
}

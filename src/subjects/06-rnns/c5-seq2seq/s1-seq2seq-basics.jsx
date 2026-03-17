import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function TeacherForcingDemo() {
  const [tfRatio, setTfRatio] = useState(0.5)
  const steps = 6
  const tokens = ['<s>', 'the', 'cat', 'sat', 'down', '.']
  const predicted = ['<s>', 'a', 'cat', 'sit', 'down', ',']

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Teacher Forcing Ratio</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        TF ratio: {tfRatio.toFixed(2)}
        <input type="range" min={0} max={1} step={0.05} value={tfRatio} onChange={e => setTfRatio(parseFloat(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <div className="flex gap-1 justify-center">
        {Array.from({ length: steps }, (_, i) => {
          const useTF = (i / steps) < tfRatio
          return (
            <div key={i} className="flex flex-col items-center gap-1">
              <div className={`px-2 py-1 rounded text-xs font-mono ${useTF ? 'bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-300' : 'bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300'}`}>
                {useTF ? tokens[i] : predicted[i]}
              </div>
              <span className="text-xs text-gray-400">{useTF ? 'truth' : 'pred'}</span>
            </div>
          )
        })}
      </div>
      <p className="text-xs text-center mt-2 text-gray-500">
        Violet = ground truth input (teacher forcing), Orange = model's own prediction
      </p>
    </div>
  )
}

function BeamSearchViz() {
  const [beamWidth, setBeamWidth] = useState(3)
  const beams = [
    { tokens: ['the', 'cat'], score: -0.8 },
    { tokens: ['a', 'cat'], score: -1.2 },
    { tokens: ['the', 'dog'], score: -1.5 },
    { tokens: ['a', 'dog'], score: -2.1 },
    { tokens: ['my', 'cat'], score: -2.4 },
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Beam Search (width={beamWidth})</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Beam width:
        <input type="range" min={1} max={5} step={1} value={beamWidth} onChange={e => setBeamWidth(parseInt(e.target.value))} className="w-32 accent-violet-500" />
      </label>
      <div className="space-y-1">
        {beams.slice(0, beamWidth).map((b, i) => (
          <div key={i} className="flex items-center gap-3">
            <span className="font-mono text-sm text-violet-600 dark:text-violet-400 w-28">{b.tokens.join(' ')}</span>
            <div className="flex-1 bg-gray-100 dark:bg-gray-800 rounded h-4 overflow-hidden">
              <div className="h-full bg-violet-500 rounded" style={{ width: `${Math.exp(b.score) * 100}%` }} />
            </div>
            <span className="text-xs text-gray-500 w-12 text-right">{b.score.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function Seq2SeqBasics() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Sequence-to-sequence models combine an encoder and decoder to handle tasks where both
        input and output are variable-length sequences. Key training and inference techniques
        include teacher forcing and beam search.
      </p>

      <DefinitionBlock title="Teacher Forcing">
        <p>
          During training, the decoder receives the <strong>ground-truth</strong> previous token
          as input rather than its own prediction:
        </p>
        <BlockMath math="\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t^* | y_1^*, \ldots, y_{t-1}^*, c)" />
        <p className="mt-2">
          This stabilizes training but creates a mismatch between training (seeing perfect inputs)
          and inference (seeing its own potentially erroneous predictions), known as <strong>exposure bias</strong>.
        </p>
      </DefinitionBlock>

      <TeacherForcingDemo />

      <DefinitionBlock title="Beam Search">
        <p>
          At inference, beam search maintains the top-<InlineMath math="k" /> most likely partial
          sequences at each step:
        </p>
        <BlockMath math="\text{score}(y_{1:t}) = \sum_{i=1}^{t} \log P(y_i | y_{<i}, c)" />
        <p className="mt-2">
          Length normalization prevents bias toward shorter sequences:
          <InlineMath math="\text{score}_{\text{norm}} = \frac{1}{T^\alpha} \sum_{t} \log P(y_t)" /> with <InlineMath math="\alpha \approx 0.6" />.
        </p>
      </DefinitionBlock>

      <BeamSearchViz />

      <PythonCode
        title="Seq2Seq with Teacher Forcing"
        code={`import torch
import torch.nn as nn
import random

class Seq2Seq(nn.Module):
    def __init__(self, enc_vocab, dec_vocab, embed_dim, hidden_dim):
        super().__init__()
        self.enc_embed = nn.Embedding(enc_vocab, embed_dim)
        self.dec_embed = nn.Embedding(dec_vocab, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, dec_vocab)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        _, (h, c) = self.encoder(self.enc_embed(src))
        B, T = tgt.shape
        outputs = torch.zeros(B, T, self.fc.out_features, device=src.device)
        dec_input = tgt[:, 0:1]  # <sos> token

        for t in range(T):
            out, (h, c) = self.decoder(self.dec_embed(dec_input), (h, c))
            outputs[:, t:t+1] = self.fc(out)
            if t < T - 1:
                use_tf = random.random() < teacher_forcing_ratio
                dec_input = tgt[:, t+1:t+2] if use_tf else outputs[:, t].argmax(-1, keepdim=True)
        return outputs

model = Seq2Seq(5000, 6000, 128, 256)
src = torch.randint(0, 5000, (4, 15))
tgt = torch.randint(0, 6000, (4, 12))
out = model(src, tgt, teacher_forcing_ratio=0.5)
print(f"Output: {out.shape}")  # (4, 12, 6000)`}
      />

      <WarningBlock title="Exposure Bias">
        <p>
          Scheduled sampling (Bengio et al., 2015) gradually decreases the teacher forcing ratio
          during training, easing the transition from training to inference. However, it can
          slow convergence. Alternative approaches include sequence-level training with REINFORCE
          or minimum risk training.
        </p>
      </WarningBlock>
    </div>
  )
}

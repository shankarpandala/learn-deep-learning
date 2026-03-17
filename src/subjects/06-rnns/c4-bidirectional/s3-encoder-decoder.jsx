import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function EncoderDecoderDiagram() {
  const [showContext, setShowContext] = useState(true)
  const W = 460, H = 160
  const encSteps = ['le', 'chat', 'est', 'noir']
  const decSteps = ['the', 'cat', 'is', 'black']

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Encoder-Decoder Architecture</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        <input type="checkbox" checked={showContext} onChange={e => setShowContext(e.target.checked)} className="accent-violet-500" />
        Show context vector
      </label>
      <svg width={W} height={H} className="mx-auto block">
        {encSteps.map((w, i) => {
          const cx = 30 + i * 55
          return (
            <g key={`e${i}`}>
              <rect x={cx - 20} y={50} width={40} height={28} rx={4} fill="#8b5cf6" opacity={0.8} />
              <text x={cx} y={68} textAnchor="middle" fill="white" fontSize={8} fontWeight="bold">enc</text>
              <text x={cx} y={100} textAnchor="middle" fill="#6b7280" fontSize={9}>{w}</text>
              {i < 3 && <line x1={cx + 20} y1={64} x2={cx + 35} y2={64} stroke="#a78bfa" strokeWidth={1.5} />}
            </g>
          )
        })}
        {showContext && (
          <>
            <circle cx={230} cy={64} r={14} fill="#7c3aed" opacity={0.9} />
            <text x={230} y={68} textAnchor="middle" fill="white" fontSize={8} fontWeight="bold">c</text>
            <line x1={195} y1={64} x2={216} y2={64} stroke="#a78bfa" strokeWidth={2} />
          </>
        )}
        {decSteps.map((w, i) => {
          const cx = 270 + i * 50
          return (
            <g key={`d${i}`}>
              <rect x={cx - 20} y={50} width={40} height={28} rx={4} fill="#f97316" opacity={0.8} />
              <text x={cx} y={68} textAnchor="middle" fill="white" fontSize={8} fontWeight="bold">dec</text>
              <text x={cx} y={28} textAnchor="middle" fill="#6b7280" fontSize={9}>{w}</text>
              {i < 3 && <line x1={cx + 20} y1={64} x2={cx + 30} y2={64} stroke="#fdba74" strokeWidth={1.5} />}
              {showContext && <line x1={230} y1={78} x2={cx} y2={50} stroke="#c4b5fd" strokeWidth={0.7} strokeDasharray="3,2" opacity={0.5} />}
            </g>
          )
        })}
        <text x={100} y={140} textAnchor="middle" fill="#8b5cf6" fontSize={10}>Encoder</text>
        <text x={345} y={140} textAnchor="middle" fill="#f97316" fontSize={10}>Decoder</text>
      </svg>
    </div>
  )
}

export default function EncoderDecoder() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The encoder-decoder framework maps variable-length input sequences to variable-length
        output sequences through a fixed-size context vector, forming the basis of
        sequence-to-sequence (seq2seq) models.
      </p>

      <DefinitionBlock title="Encoder-Decoder Framework">
        <p>The <strong>encoder</strong> reads the input sequence and produces a context vector:</p>
        <BlockMath math="h_t^{\text{enc}} = f_{\text{enc}}(x_t, h_{t-1}^{\text{enc}}), \quad c = h_T^{\text{enc}}" />
        <p className="mt-2">The <strong>decoder</strong> generates the output sequence conditioned on <InlineMath math="c" />:</p>
        <BlockMath math="h_t^{\text{dec}} = f_{\text{dec}}(y_{t-1}, h_{t-1}^{\text{dec}}, c)" />
        <BlockMath math="P(y_t | y_{<t}, x) = \text{softmax}(W_o h_t^{\text{dec}})" />
      </DefinitionBlock>

      <EncoderDecoderDiagram />

      <WarningBlock title="Information Bottleneck">
        <p>
          The entire source sentence is compressed into a single fixed-size vector <InlineMath math="c" />.
          For long sequences, this bottleneck causes information loss and degraded performance.
          This limitation directly motivated the invention of <strong>attention mechanisms</strong>,
          which allow the decoder to selectively access all encoder hidden states.
        </p>
      </WarningBlock>

      <ExampleBlock title="Machine Translation Pipeline">
        <p>For translating "le chat est noir" to "the cat is black":</p>
        <ol className="list-decimal ml-6 mt-1 space-y-1">
          <li>Encoder processes each French token, updating hidden state</li>
          <li>Final encoder state <InlineMath math="c" /> summarizes the French sentence</li>
          <li>Decoder receives <InlineMath math="c" /> as initial state and a start token</li>
          <li>At each step, decoder predicts next English token and feeds it back as input</li>
        </ol>
      </ExampleBlock>

      <PythonCode
        title="Encoder-Decoder in PyTorch"
        code={`import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        embedded = self.embed(src)
        outputs, (h, c) = self.lstm(embedded)
        return outputs, h, c

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt, h, c):
        embedded = self.embed(tgt)
        output, (h, c) = self.lstm(embedded, (h, c))
        prediction = self.fc(output)
        return prediction, h, c

enc = Encoder(5000, 128, 256)
dec = Decoder(6000, 128, 256)

src = torch.randint(0, 5000, (4, 15))
tgt = torch.randint(0, 6000, (4, 12))

enc_out, h, c = enc(src)
output, _, _ = dec(tgt, h, c)
print(f"Decoder output: {output.shape}")  # (4, 12, 6000)`}
      />

      <NoteBlock type="note" title="Historical Significance">
        <p>
          The encoder-decoder framework (Cho et al., 2014; Sutskever et al., 2014) was a
          breakthrough for neural machine translation, replacing phrase-based statistical systems.
          Combined with attention (Bahdanau et al., 2015), it became the dominant NMT approach
          until Transformers arrived. The architectural pattern remains foundational in modern
          sequence models.
        </p>
      </NoteBlock>
    </div>
  )
}

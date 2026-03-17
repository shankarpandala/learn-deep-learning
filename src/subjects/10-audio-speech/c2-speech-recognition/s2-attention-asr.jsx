import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ASRArchitectureComparison() {
  const [arch, setArch] = useState('las')

  const architectures = {
    las: { name: 'Listen, Attend, Spell', encoder: 'Pyramidal BiLSTM', decoder: 'LSTM + attention', alignment: 'Soft attention', strengths: 'Strong language modeling', weaknesses: 'Slow autoregressive decoding' },
    rnnt: { name: 'RNN-Transducer', encoder: 'LSTM / Conformer', decoder: 'Prediction network (LSTM)', alignment: 'RNN-T loss (CTC-like)', strengths: 'Streaming capable', weaknesses: 'Complex training' },
    transformer: { name: 'Transformer ASR', encoder: 'Conformer / Transformer', decoder: 'Transformer decoder', alignment: 'Cross-attention', strengths: 'Best offline accuracy', weaknesses: 'High memory, non-streaming' },
  }

  const a = architectures[arch]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">ASR Architecture Comparison</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.entries(architectures).map(([key, val]) => (
          <button key={key} onClick={() => setArch(key)}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${arch === key ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3 text-sm">
        {[['Encoder', a.encoder], ['Decoder', a.decoder], ['Alignment', a.alignment], ['Strengths', a.strengths], ['Weaknesses', a.weaknesses]].map(([label, val]) => (
          <div key={label} className={`rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3 ${label === 'Weaknesses' ? 'col-span-2 sm:col-span-1' : ''}`}>
            <p className="text-xs text-violet-600 dark:text-violet-400 font-semibold">{label}</p>
            <p className="text-gray-700 dark:text-gray-300">{val}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function AttentionBasedASR() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Attention-based models replace CTC's conditional independence assumption with an
        autoregressive decoder that attends to encoder outputs, enabling the model to
        learn implicit language modeling jointly with acoustic modeling.
      </p>

      <DefinitionBlock title="Listen, Attend and Spell (LAS)">
        <p>LAS consists of three components:</p>
        <BlockMath math="h = \text{Encoder}(X), \quad c_i = \text{Attention}(s_{i-1}, h), \quad y_i = \text{Decoder}(s_{i-1}, y_{i-1}, c_i)" />
        <p className="mt-2">
          The encoder (listener) processes audio features, attention computes a context
          vector <InlineMath math="c_i" />, and the decoder (speller) generates tokens autoregressively.
        </p>
      </DefinitionBlock>

      <ASRArchitectureComparison />

      <ExampleBlock title="Conformer: Convolution-Augmented Transformer">
        <p>
          The Conformer block combines self-attention with depthwise convolutions:
        </p>
        <BlockMath math="y = x + \tfrac{1}{2}\text{FFN}(x) + \text{MHSA}(x) + \text{Conv}(x) + \tfrac{1}{2}\text{FFN}(x)" />
        <p className="mt-1">
          This captures both global context (via attention) and local patterns (via convolution),
          achieving state-of-the-art results on LibriSpeech with a 1.9% WER.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Conformer-based ASR with torchaudio"
        code={`import torch
import torchaudio

# Conformer encoder (available in torchaudio)
conformer = torchaudio.models.Conformer(
    input_dim=80,         # mel features
    num_heads=4,
    ffn_dim=256,
    num_layers=8,
    depthwise_conv_kernel_size=31,
)

# Simulated log-mel input: [batch, time, features]
features = torch.randn(2, 200, 80)
lengths = torch.tensor([200, 180])

# Encode
encoded, out_lengths = conformer(features, lengths)
print(f"Encoder output: {encoded.shape}")  # [2, 200, 80]

# Simple CTC head on top of conformer
ctc_head = torch.nn.Linear(80, 29)  # vocab size
logits = ctc_head(encoded)
log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)  # [T, B, C]
print(f"CTC logits: {log_probs.shape}")`}
      />

      <WarningBlock title="Attention Failures in Long Audio">
        <p>
          Pure attention-based ASR can fail on very long utterances because the attention mechanism
          may not learn a monotonic left-to-right alignment. Solutions include monotonic attention
          constraints, CTC-attention joint training, or chunked processing.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="CTC-Attention Hybrid">
        <p>
          Modern ASR systems often combine CTC and attention losses:
          <InlineMath math="\mathcal{L} = \lambda \mathcal{L}_\text{CTC} + (1-\lambda)\mathcal{L}_\text{attention}" />.
          The CTC loss enforces monotonic alignment as a regularizer, while the attention decoder
          provides superior language modeling. This is the default approach in ESPnet and other toolkits.
        </p>
      </NoteBlock>
    </div>
  )
}

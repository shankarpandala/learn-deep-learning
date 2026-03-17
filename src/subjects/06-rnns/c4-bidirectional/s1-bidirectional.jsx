import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function BiRNNDiagram() {
  const [step, setStep] = useState(3)
  const W = 440, H = 200

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Bidirectional RNN</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Time steps: {step + 1}
        <input type="range" min={2} max={5} step={1} value={step} onChange={e => setStep(parseInt(e.target.value))} className="w-32 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <defs>
          <marker id="biArr" markerWidth={7} markerHeight={5} refX={7} refY={2.5} orient="auto">
            <path d="M0,0 L7,2.5 L0,5 Z" fill="#8b5cf6" />
          </marker>
          <marker id="biArrB" markerWidth={7} markerHeight={5} refX={7} refY={2.5} orient="auto">
            <path d="M0,0 L7,2.5 L0,5 Z" fill="#f97316" />
          </marker>
        </defs>
        {Array.from({ length: step + 1 }, (_, i) => {
          const cx = 50 + i * (340 / step)
          return (
            <g key={i}>
              <rect x={cx - 20} y={50} width={40} height={28} rx={4} fill="#8b5cf6" opacity={0.8} />
              <text x={cx} y={69} textAnchor="middle" fill="white" fontSize={9} fontWeight="bold">fwd</text>
              <rect x={cx - 20} y={100} width={40} height={28} rx={4} fill="#f97316" opacity={0.8} />
              <text x={cx} y={119} textAnchor="middle" fill="white" fontSize={9} fontWeight="bold">bwd</text>
              <text x={cx} y={178} textAnchor="middle" fill="#6b7280" fontSize={10}>x_{i}</text>
              <line x1={cx} y1={168} x2={cx} y2={128} stroke="#9ca3af" strokeWidth={1} />
              <line x1={cx} y1={168} x2={cx} y2={78} stroke="#9ca3af" strokeWidth={1} />
              <text x={cx} y={30} textAnchor="middle" fill="#6b7280" fontSize={10}>y_{i}</text>
              <line x1={cx} y1={50} x2={cx} y2={36} stroke="#9ca3af" strokeWidth={1} />
              {i < step && <line x1={cx + 20} y1={64} x2={cx + (340 / step) - 20} y2={64} stroke="#8b5cf6" strokeWidth={1.5} markerEnd="url(#biArr)" />}
              {i > 0 && <line x1={cx - 20} y1={114} x2={cx - (340 / step) + 20} y2={114} stroke="#f97316" strokeWidth={1.5} markerEnd="url(#biArrB)" />}
            </g>
          )
        })}
        <text x={W - 10} y={69} textAnchor="end" fill="#8b5cf6" fontSize={9}>forward</text>
        <text x={W - 10} y={119} textAnchor="end" fill="#f97316" fontSize={9}>backward</text>
      </svg>
    </div>
  )
}

export default function BidirectionalRNN() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Bidirectional RNNs process sequences in both forward and backward directions,
        allowing each output to depend on both past and future context. This is especially
        powerful for tasks where the full sequence is available at inference time.
      </p>

      <DefinitionBlock title="Bidirectional RNN">
        <p>A bidirectional RNN runs two separate hidden state sequences:</p>
        <BlockMath math="\overrightarrow{h_t} = f(W_{\rightarrow}[\overrightarrow{h_{t-1}}, x_t])" />
        <BlockMath math="\overleftarrow{h_t} = f(W_{\leftarrow}[\overleftarrow{h_{t+1}}, x_t])" />
        <p className="mt-2">The output at each step concatenates both directions:</p>
        <BlockMath math="h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}] \in \mathbb{R}^{2d}" />
      </DefinitionBlock>

      <BiRNNDiagram />

      <ExampleBlock title="Named Entity Recognition">
        <p>
          In the sentence "Paris is the capital of France", recognizing "Paris" as a location
          benefits from seeing "capital of France" to the right. A forward-only RNN at position 0
          has no future context. A BiRNN at position 0 sees both the word itself and a backward
          summary of the full sentence.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Bidirectional LSTM in PyTorch"
        code={`import torch
import torch.nn as nn

bilstm = nn.LSTM(
    input_size=64,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
    bidirectional=True  # key argument
)

x = torch.randn(8, 30, 64)
output, (h_n, c_n) = bilstm(x)

print(f"Output: {output.shape}")   # (8, 30, 256) = 2 * hidden_size
print(f"Hidden: {h_n.shape}")      # (4, 8, 128) = 2*num_layers x batch x hidden

# Extract final forward and backward hidden states
h_forward = h_n[-2]   # last layer, forward
h_backward = h_n[-1]  # last layer, backward
h_combined = torch.cat([h_forward, h_backward], dim=-1)
print(f"Combined: {h_combined.shape}")  # (8, 256)

# For classification, use the combined representation
classifier = nn.Linear(256, 10)
logits = classifier(h_combined)
print(f"Logits: {logits.shape}")  # (8, 10)`}
      />

      <WarningBlock title="Cannot Use for Autoregressive Generation">
        <p>
          Bidirectional RNNs require the full input sequence at inference time. They are
          <strong> not suitable for autoregressive generation</strong> (e.g., language modeling,
          text generation), where tokens are produced one at a time and future context is
          unavailable. Use unidirectional RNNs for generation tasks.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Applications">
        <p>
          BiRNNs are widely used in NER, POS tagging, sentiment analysis, machine translation
          encoders, and speech recognition. ELMo (Peters et al., 2018) uses a deep bidirectional
          LSTM to produce contextual word embeddings that became foundational for transfer learning
          in NLP.
        </p>
      </NoteBlock>
    </div>
  )
}

import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function RNNDiagram() {
  const [step, setStep] = useState(0)
  const maxSteps = 4
  const W = 460, H = 180

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Unrolled RNN ({step + 1} time steps)</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Steps
          <input type="range" min={0} max={maxSteps - 1} step={1} value={step} onChange={e => setStep(parseInt(e.target.value))} className="w-32 accent-violet-500" />
          {step + 1}
        </label>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        {Array.from({ length: step + 1 }, (_, i) => {
          const cx = 60 + i * 100, cyH = 60, cyX = 150, cyY = 20
          return (
            <g key={i}>
              <rect x={cx - 25} y={cyH - 20} width={50} height={40} rx={6} fill="#8b5cf6" opacity={0.85} />
              <text x={cx} y={cyH + 5} textAnchor="middle" fill="white" fontSize={12} fontWeight="bold">h_{i}</text>
              <text x={cx} y={cyX} textAnchor="middle" fill="#6b7280" fontSize={11}>x_{i}</text>
              <line x1={cx} y1={cyX - 10} x2={cx} y2={cyH + 20} stroke="#8b5cf6" strokeWidth={1.5} markerEnd="url(#arr)" />
              <text x={cx} y={cyY} textAnchor="middle" fill="#6b7280" fontSize={11}>y_{i}</text>
              <line x1={cx} y1={cyH - 20} x2={cx} y2={cyY + 6} stroke="#8b5cf6" strokeWidth={1.5} markerEnd="url(#arr)" />
              {i < step && <line x1={cx + 25} y1={cyH} x2={cx + 75} y2={cyH} stroke="#a78bfa" strokeWidth={2} markerEnd="url(#arr)" />}
            </g>
          )
        })}
        <defs>
          <marker id="arr" markerWidth={8} markerHeight={6} refX={8} refY={3} orient="auto">
            <path d="M0,0 L8,3 L0,6 Z" fill="#8b5cf6" />
          </marker>
        </defs>
      </svg>
    </div>
  )
}

export default function RNNBasics() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Recurrent Neural Networks process sequential data by maintaining a hidden state that captures
        information from previous time steps, making them naturally suited for tasks involving
        temporal or ordered data such as text, speech, and time series.
      </p>

      <DefinitionBlock title="Recurrent Neural Network">
        <p>An RNN computes a hidden state <InlineMath math="h_t" /> at each time step <InlineMath math="t" /> via:</p>
        <BlockMath math="h_t = \tanh(W_{hh}\, h_{t-1} + W_{xh}\, x_t + b_h)" />
        <BlockMath math="y_t = W_{hy}\, h_t + b_y" />
        <p className="mt-2">
          The same weight matrices <InlineMath math="W_{hh}, W_{xh}, W_{hy}" /> are shared across all time steps,
          giving RNNs a fixed parameter count regardless of sequence length.
        </p>
      </DefinitionBlock>

      <RNNDiagram />

      <ExampleBlock title="Hidden State Dimensions">
        <p>
          For input dimension <InlineMath math="d = 50" /> and hidden size <InlineMath math="h = 128" />:
        </p>
        <BlockMath math="W_{xh} \in \mathbb{R}^{128 \times 50},\quad W_{hh} \in \mathbb{R}^{128 \times 128},\quad W_{hy} \in \mathbb{R}^{|V| \times 128}" />
        <p>Total recurrent parameters: <InlineMath math="128 \times 50 + 128 \times 128 = 22{,}784" />.</p>
      </ExampleBlock>

      <PythonCode
        title="Vanilla RNN in PyTorch"
        code={`import torch
import torch.nn as nn

# Single-layer RNN: input_size=50, hidden_size=128
rnn = nn.RNN(input_size=50, hidden_size=128, batch_first=True)

# Batch of 8 sequences, each length 20, feature dim 50
x = torch.randn(8, 20, 50)
h0 = torch.zeros(1, 8, 128)  # initial hidden state

output, h_n = rnn(x, h0)
print(f"Output shape: {output.shape}")   # (8, 20, 128)
print(f"Final hidden: {h_n.shape}")      # (1, 8, 128)

# Manual single step
W_xh = rnn.weight_ih_l0  # (4*128, 50) for RNN it's (128, 50)
W_hh = rnn.weight_hh_l0  # (128, 128)
print(f"Params: {sum(p.numel() for p in rnn.parameters()):,}")`}
      />

      <NoteBlock type="note" title="Weight Sharing is Key">
        <p>
          Unlike feedforward networks that have separate parameters per layer, an RNN
          <strong> reuses the same weights</strong> at every time step. This weight sharing acts as
          an inductive bias for temporal invariance and keeps the model compact, but it also makes
          training via backpropagation through time challenging due to vanishing or exploding gradients.
        </p>
      </NoteBlock>

      <DefinitionBlock title="Hidden State as Memory">
        <p>
          The hidden state <InlineMath math="h_t" /> is a compressed representation of the entire
          input history <InlineMath math="(x_1, x_2, \ldots, x_t)" />. In practice, vanilla RNNs
          struggle to remember information beyond roughly 10-20 time steps due to the vanishing
          gradient problem, motivating architectures like LSTM and GRU.
        </p>
      </DefinitionBlock>

      <ExampleBlock title="Common RNN Patterns">
        <p>
          RNNs can be configured in several input-output patterns:
          <strong> One-to-many</strong> (image captioning),
          <strong> many-to-one</strong> (sentiment classification),
          <strong> many-to-many</strong> (machine translation, language modeling).
          The same core recurrence equation applies in all cases; only the input feeding
          and output tapping differ.
        </p>
      </ExampleBlock>
    </div>
  )
}

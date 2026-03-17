import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function sigmoid(x) { return 1 / (1 + Math.exp(-x)) }

function GRUGateDemo() {
  const [updateVal, setUpdateVal] = useState(0.6)
  const [resetVal, setResetVal] = useState(0.8)
  const [prevH, setPrevH] = useState(1.0)
  const candidateRaw = 0.5

  const candidateH = Math.tanh(candidateRaw + resetVal * prevH)
  const newH = (1 - updateVal) * prevH + updateVal * candidateH

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">GRU Gate Explorer</h3>
      <div className="grid grid-cols-3 gap-3 mb-4">
        <label className="flex flex-col gap-1 text-sm text-gray-600 dark:text-gray-400">
          Update gate (z): {updateVal.toFixed(2)}
          <input type="range" min={0} max={1} step={0.01} value={updateVal} onChange={e => setUpdateVal(parseFloat(e.target.value))} className="accent-violet-500" />
        </label>
        <label className="flex flex-col gap-1 text-sm text-gray-600 dark:text-gray-400">
          Reset gate (r): {resetVal.toFixed(2)}
          <input type="range" min={0} max={1} step={0.01} value={resetVal} onChange={e => setResetVal(parseFloat(e.target.value))} className="accent-violet-500" />
        </label>
        <label className="flex flex-col gap-1 text-sm text-gray-600 dark:text-gray-400">
          Previous h: {prevH.toFixed(2)}
          <input type="range" min={-2} max={2} step={0.01} value={prevH} onChange={e => setPrevH(parseFloat(e.target.value))} className="accent-violet-500" />
        </label>
      </div>
      <div className="flex gap-6 justify-center text-sm">
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/30 px-4 py-2 text-center">
          <div className="text-violet-700 dark:text-violet-300 font-semibold">Candidate</div>
          <div className="text-lg font-mono">{candidateH.toFixed(4)}</div>
        </div>
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/30 px-4 py-2 text-center">
          <div className="text-violet-700 dark:text-violet-300 font-semibold">New h_t</div>
          <div className="text-lg font-mono">{newH.toFixed(4)}</div>
        </div>
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/30 px-4 py-2 text-center">
          <div className="text-violet-700 dark:text-violet-300 font-semibold">% from prev</div>
          <div className="text-lg font-mono">{((1 - updateVal) * 100).toFixed(0)}%</div>
        </div>
      </div>
    </div>
  )
}

export default function GRUArchitecture() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The Gated Recurrent Unit (GRU), introduced by Cho et al. (2014), simplifies the LSTM
        architecture by merging the cell and hidden states into a single state vector and using
        only two gates instead of three.
      </p>

      <DefinitionBlock title="GRU Equations">
        <p>Given input <InlineMath math="x_t" /> and previous hidden state <InlineMath math="h_{t-1}" />:</p>
        <BlockMath math="z_t = \sigma(W_z [h_{t-1}, x_t] + b_z) \quad \text{(update gate)}" />
        <BlockMath math="r_t = \sigma(W_r [h_{t-1}, x_t] + b_r) \quad \text{(reset gate)}" />
        <BlockMath math="\tilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t] + b_h) \quad \text{(candidate)}" />
        <BlockMath math="h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t" />
      </DefinitionBlock>

      <GRUGateDemo />

      <ExampleBlock title="Gate Intuition">
        <p>
          <strong>Update gate</strong> <InlineMath math="z_t" />: Controls how much new information to mix in.
          When <InlineMath math="z_t \to 0" />, the hidden state is copied unchanged (like an LSTM with <InlineMath math="f_t = 1" />).
          When <InlineMath math="z_t \to 1" />, the state is fully replaced by the candidate.
        </p>
        <p className="mt-2">
          <strong>Reset gate</strong> <InlineMath math="r_t" />: Controls how much past state to expose when computing
          the candidate. When <InlineMath math="r_t \to 0" />, the candidate ignores the past, allowing the GRU
          to drop irrelevant history.
        </p>
      </ExampleBlock>

      <PythonCode
        title="GRU Implementation from Scratch"
        code={`import torch
import torch.nn as nn

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Update gate
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        # Reset gate
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        # Candidate
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h_prev):
        combined = torch.cat([h_prev, x], dim=-1)
        z = torch.sigmoid(self.W_z(combined))
        r = torch.sigmoid(self.W_r(combined))
        combined_r = torch.cat([r * h_prev, x], dim=-1)
        h_candidate = torch.tanh(self.W_h(combined_r))
        h_new = (1 - z) * h_prev + z * h_candidate
        return h_new

# Test
cell = GRUCell(64, 128)
h = torch.zeros(8, 128)
for t in range(20):
    x_t = torch.randn(8, 64)
    h = cell(x_t, h)
print(f"Final h: {h.shape}")  # (8, 128)`}
      />

      <NoteBlock type="note" title="GRU Simplicity Advantage">
        <p>
          With no separate cell state and fewer gates, the GRU is faster to compute per step
          and has ~25% fewer parameters than an LSTM. This makes GRUs particularly attractive
          for smaller datasets or when inference speed is critical. The GRU achieves comparable
          performance to the LSTM on many benchmarks.
        </p>
      </NoteBlock>
    </div>
  )
}

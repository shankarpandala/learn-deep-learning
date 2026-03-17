import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function sigmoid(x) { return 1 / (1 + Math.exp(-x)) }

function GateDemo() {
  const [forgetVal, setForgetVal] = useState(0.8)
  const [inputVal, setInputVal] = useState(0.5)
  const [candidate, setCandidate] = useState(0.6)
  const [prevCell, setPrevCell] = useState(1.0)

  const newCell = forgetVal * prevCell + inputVal * candidate
  const outputGate = 0.7
  const hidden = outputGate * Math.tanh(newCell)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Interactive LSTM Gate Demo</h3>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <label className="flex flex-col gap-1 text-sm text-gray-600 dark:text-gray-400">
          Forget gate (f): {forgetVal.toFixed(2)}
          <input type="range" min={0} max={1} step={0.01} value={forgetVal} onChange={e => setForgetVal(parseFloat(e.target.value))} className="accent-violet-500" />
        </label>
        <label className="flex flex-col gap-1 text-sm text-gray-600 dark:text-gray-400">
          Input gate (i): {inputVal.toFixed(2)}
          <input type="range" min={0} max={1} step={0.01} value={inputVal} onChange={e => setInputVal(parseFloat(e.target.value))} className="accent-violet-500" />
        </label>
        <label className="flex flex-col gap-1 text-sm text-gray-600 dark:text-gray-400">
          Candidate (g): {candidate.toFixed(2)}
          <input type="range" min={-1} max={1} step={0.01} value={candidate} onChange={e => setCandidate(parseFloat(e.target.value))} className="accent-violet-500" />
        </label>
        <label className="flex flex-col gap-1 text-sm text-gray-600 dark:text-gray-400">
          Previous cell (c_prev): {prevCell.toFixed(2)}
          <input type="range" min={-2} max={2} step={0.01} value={prevCell} onChange={e => setPrevCell(parseFloat(e.target.value))} className="accent-violet-500" />
        </label>
      </div>
      <div className="flex gap-6 justify-center text-sm">
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/30 px-4 py-2 text-center">
          <div className="text-violet-700 dark:text-violet-300 font-semibold">New Cell</div>
          <div className="text-lg font-mono">{newCell.toFixed(4)}</div>
        </div>
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/30 px-4 py-2 text-center">
          <div className="text-violet-700 dark:text-violet-300 font-semibold">Hidden</div>
          <div className="text-lg font-mono">{hidden.toFixed(4)}</div>
        </div>
      </div>
    </div>
  )
}

export default function LSTMGates() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Long Short-Term Memory networks address the vanishing gradient problem by introducing
        a gated cell state that can maintain information over long time spans. The gating
        mechanism learns when to store, update, and output information.
      </p>

      <DefinitionBlock title="LSTM Equations">
        <p>Given input <InlineMath math="x_t" /> and previous states <InlineMath math="h_{t-1}, c_{t-1}" />:</p>
        <BlockMath math="f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}" />
        <BlockMath math="i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(input gate)}" />
        <BlockMath math="\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c) \quad \text{(candidate)}" />
        <BlockMath math="c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(cell update)}" />
        <BlockMath math="o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(output gate)}" />
        <BlockMath math="h_t = o_t \odot \tanh(c_t)" />
      </DefinitionBlock>

      <GateDemo />

      <ExampleBlock title="Cell State as a Highway">
        <p>
          The cell state update <InlineMath math="c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t" /> acts
          like a highway: when <InlineMath math="f_t \approx 1" /> and <InlineMath math="i_t \approx 0" />,
          the cell state passes through unchanged, allowing gradients to flow across many time steps
          without decay.
        </p>
      </ExampleBlock>

      <PythonCode
        title="LSTM in PyTorch"
        code={`import torch
import torch.nn as nn

lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)

x = torch.randn(8, 30, 64)  # batch=8, seq=30, features=64
h0 = torch.zeros(2, 8, 128)  # 2 layers
c0 = torch.zeros(2, 8, 128)

output, (h_n, c_n) = lstm(x, (h0, c0))
print(f"Output: {output.shape}")       # (8, 30, 128)
print(f"Hidden: {h_n.shape}")          # (2, 8, 128)
print(f"Cell:   {c_n.shape}")          # (2, 8, 128)
print(f"Params: {sum(p.numel() for p in lstm.parameters()):,}")
# LSTM has 4x parameters of vanilla RNN (4 gate weight matrices)`}
      />

      <NoteBlock type="note" title="Why 4x Parameters?">
        <p>
          An LSTM with hidden size <InlineMath math="h" /> and input size <InlineMath math="d" /> has
          four gate matrices, each of size <InlineMath math="(h+d) \times h" />, giving total
          recurrent parameters <InlineMath math="4 \cdot h \cdot (h + d) + 4h" /> (including biases).
          This is exactly 4 times a vanilla RNN of the same hidden size.
        </p>
      </NoteBlock>

      <ExampleBlock title="Gate Values in Practice">
        <p>
          During training on language modeling tasks, forget gates typically learn values
          close to 1 (remembering most information), while input and output gates show
          more variation. The forget gate bias is commonly initialized to 1.0 (Gers et al., 2000)
          to encourage information flow early in training.
        </p>
        <BlockMath math="f_t \approx 0.9,\quad i_t \in [0.1, 0.8],\quad o_t \in [0.3, 0.9]" />
      </ExampleBlock>
    </div>
  )
}

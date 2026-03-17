import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function EfficiencyChart() {
  const [seqLen, setSeqLen] = useState(100)
  const models = [
    { name: 'LSTM', base: 1.0, parallel: false },
    { name: 'GRU', base: 0.75, parallel: false },
    { name: 'SRU', base: 0.4, parallel: true },
    { name: 'QRNN', base: 0.35, parallel: true },
  ]
  const maxTime = seqLen * 1.0

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Relative Throughput</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Sequence length: {seqLen}
        <input type="range" min={10} max={500} step={10} value={seqLen} onChange={e => setSeqLen(parseInt(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <div className="space-y-2">
        {models.map(m => {
          const time = m.parallel ? m.base * Math.log2(seqLen + 1) : m.base * seqLen
          const width = Math.min((time / maxTime) * 100, 100)
          return (
            <div key={m.name} className="flex items-center gap-3">
              <span className="w-14 text-sm text-gray-600 dark:text-gray-400 font-mono">{m.name}</span>
              <div className="flex-1 bg-gray-100 dark:bg-gray-800 rounded h-5 overflow-hidden">
                <div className="h-full bg-violet-500 rounded" style={{ width: `${width}%` }} />
              </div>
              <span className="text-xs text-gray-500 w-16 text-right">{m.parallel ? 'parallel' : 'sequential'}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default function MinimalRNNs() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Standard RNNs, LSTMs, and GRUs are inherently sequential, limiting GPU utilization.
        Efficient recurrent variants like SRU and QRNN restructure the computation to
        maximize parallelism while retaining sequential modeling capability.
      </p>

      <DefinitionBlock title="Simple Recurrent Unit (SRU)">
        <p>The SRU (Lei et al., 2018) separates gating from hidden state computation:</p>
        <BlockMath math="\tilde{x}_t = W x_t" />
        <BlockMath math="f_t = \sigma(W_f x_t + b_f)" />
        <BlockMath math="c_t = f_t \odot c_{t-1} + (1 - f_t) \odot \tilde{x}_t" />
        <BlockMath math="h_t = r_t \odot \tanh(c_t) + (1 - r_t) \odot x_t" />
        <p className="mt-2">
          The key insight: <InlineMath math="W x_t" /> and <InlineMath math="W_f x_t" /> depend only on the
          input and can be computed in parallel across all time steps. Only the lightweight
          element-wise recurrence is sequential.
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="Quasi-Recurrent Neural Network (QRNN)">
        <p>The QRNN (Bradbury et al., 2017) applies convolutions across time, then a minimal recurrence:</p>
        <BlockMath math="Z = \tanh(W_z * X), \quad F = \sigma(W_f * X)" />
        <BlockMath math="c_t = f_t \odot c_{t-1} + (1 - f_t) \odot z_t" />
        <p className="mt-2">
          The convolution <InlineMath math="*" /> captures local temporal patterns in parallel,
          while the element-wise gated pooling propagates information sequentially. This
          is 2-17x faster than LSTMs in practice.
        </p>
      </DefinitionBlock>

      <EfficiencyChart />

      <ExampleBlock title="Parallelism Comparison">
        <p>For a sequence of length <InlineMath math="T" /> with hidden size <InlineMath math="h" />:</p>
        <ul className="list-disc ml-6 mt-1 space-y-1">
          <li><strong>LSTM/GRU</strong>: <InlineMath math="O(T)" /> sequential matrix multiplications of <InlineMath math="O(h^2)" /></li>
          <li><strong>SRU</strong>: One batched matmul <InlineMath math="O(Th^2)" /> parallel, then <InlineMath math="O(T)" /> element-wise ops</li>
          <li><strong>QRNN</strong>: One convolution <InlineMath math="O(Thk)" /> parallel, then <InlineMath math="O(T)" /> element-wise ops</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="SRU-like Efficient Recurrence"
        code={`import torch
import torch.nn as nn

class SimpleSRU(nn.Module):
    """Simplified SRU showing the parallel/sequential split."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W = nn.Linear(input_size, hidden_size)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.W_r = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        # Parallel across time: matrix multiplications
        x_tilde = self.W(x)            # (B, T, H)
        f = torch.sigmoid(self.W_f(x)) # (B, T, H)
        r = torch.sigmoid(self.W_r(x)) # (B, T, H)

        # Sequential: lightweight element-wise recurrence
        B, T, H = x_tilde.shape
        c = torch.zeros(B, H, device=x.device)
        outputs = []
        for t in range(T):
            c = f[:, t] * c + (1 - f[:, t]) * x_tilde[:, t]
            h = r[:, t] * torch.tanh(c) + (1 - r[:, t]) * x[:, t, :H]
            outputs.append(h)
        return torch.stack(outputs, dim=1)

model = SimpleSRU(64, 128)
x = torch.randn(8, 100, 64)
out = model(x)
print(f"Output: {out.shape}")  # (8, 100, 128)`}
      />

      <NoteBlock type="note" title="When to Use Efficient RNN Variants">
        <p>
          SRU and QRNN shine when you need RNN-like sequential modeling but cannot afford
          the latency of standard LSTMs. They are especially useful for long sequences on GPUs.
          However, for most modern NLP tasks, Transformers have supplanted these architectures.
          Efficient RNNs remain relevant in <strong>streaming</strong>, <strong>edge deployment</strong>,
          and <strong>low-latency</strong> applications.
        </p>
      </NoteBlock>
    </div>
  )
}

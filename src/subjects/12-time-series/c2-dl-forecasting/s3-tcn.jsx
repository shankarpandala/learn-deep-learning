import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function DilatedConvViz() {
  const [layer, setLayer] = useState(0)
  const dilations = [1, 2, 4, 8]
  const d = dilations[layer]
  const W = 400, H = 160, N = 16
  const cellW = W / N, rowH = 30

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Causal Dilated Convolution</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Layer: {layer} (dilation = {d})
          <input type="range" min={0} max={3} step={1} value={layer} onChange={e => setLayer(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <span className="text-xs text-gray-500">Receptive field: {1 + 2 * (Math.pow(2, layer + 1) - 1)} steps</span>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        {Array.from({ length: N }, (_, i) => (
          <g key={i}>
            <rect x={i * cellW + 1} y={H - rowH} width={cellW - 2} height={rowH - 2} rx={3} fill="#e5e7eb" />
            <text x={i * cellW + cellW / 2} y={H - 8} textAnchor="middle" className="text-[9px] fill-gray-500">t-{N - 1 - i}</text>
            <rect x={i * cellW + 1} y={H - 2 * rowH - 4} width={cellW - 2} height={rowH - 2} rx={3}
              fill={i >= N - 1 - 2 * d && i <= N - 1 && (N - 1 - i) % d === 0 ? '#8b5cf6' : '#f9fafb'}
              stroke={i >= N - 1 - 2 * d && i <= N - 1 && (N - 1 - i) % d === 0 ? '#8b5cf6' : '#e5e7eb'} strokeWidth={1} />
          </g>
        ))}
        {[0, d, 2 * d].filter(k => N - 1 - k >= 0).map(k => {
          const i = N - 1 - k
          return <line key={k} x1={i * cellW + cellW / 2} y1={H - rowH} x2={(N - 1) * cellW + cellW / 2} y2={H - 2 * rowH - 4 + rowH} stroke="#8b5cf6" strokeWidth={1.5} opacity={0.6} />
        })}
        <text x={W - 10} y={H - rowH - 10} textAnchor="end" className="text-[10px] fill-violet-600 font-semibold">output</text>
        <text x={W - 10} y={H - 6} textAnchor="end" className="text-[10px] fill-gray-500">input</text>
      </svg>
      <p className="mt-2 text-center text-xs text-gray-500 dark:text-gray-400">
        Kernel size 3 with dilation {d}: reads positions t, t-{d}, t-{2 * d} (causal — no future information)
      </p>
    </div>
  )
}

export default function TemporalCNNs() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Temporal Convolutional Networks (TCNs) use causal dilated convolutions to model
        sequences. They achieve large receptive fields with fewer layers than standard CNNs
        and can be parallelized more efficiently than RNNs during training.
      </p>

      <DefinitionBlock title="Causal Dilated Convolution">
        <p>A 1D convolution with dilation <InlineMath math="d" /> and kernel size <InlineMath math="k" /> applied causally:</p>
        <BlockMath math="(x *_d f)(t) = \sum_{i=0}^{k-1} f(i) \cdot x_{t - d \cdot i}" />
        <p className="mt-2">Causal constraint: the output at time <InlineMath math="t" /> depends only on <InlineMath math="x_t, x_{t-d}, x_{t-2d}, \ldots" /> (no future leakage).</p>
      </DefinitionBlock>

      <DilatedConvViz />

      <TheoremBlock title="Receptive Field Growth" id="tcn-receptive-field">
        <p>With <InlineMath math="L" /> layers, kernel size <InlineMath math="k" />, and exponential dilation <InlineMath math="d_\ell = 2^\ell" />:</p>
        <BlockMath math="R = 1 + (k-1) \sum_{\ell=0}^{L-1} 2^\ell = 1 + (k-1)(2^L - 1)" />
        <p>The receptive field grows <strong>exponentially</strong> with depth, while parameter count grows only linearly.</p>
      </TheoremBlock>

      <ExampleBlock title="TCN vs LSTM">
        <p>
          With <InlineMath math="k=3" /> and <InlineMath math="L=8" /> layers, the TCN has a receptive field
          of <InlineMath math="1 + 2 \times 255 = 511" /> time steps — comparable to an LSTM processing
          511 steps, but fully parallelizable during training (no sequential dependency).
        </p>
      </ExampleBlock>

      <PythonCode
        title="TCN Residual Block in PyTorch"
        code={`import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation  # causal padding
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)
        self.pad = pad

    def forward(self, x):  # x: (B, C, T)
        out = self.drop(torch.relu(self.norm1(self.conv1(x)[:, :, :x.size(2)])))
        out = self.drop(torch.relu(self.norm2(self.conv2(out)[:, :, :x.size(2)])))
        return torch.relu(out + x)  # residual connection

class TCN(nn.Module):
    def __init__(self, in_ch=1, hidden=64, layers=6, horizon=6):
        super().__init__()
        self.input_proj = nn.Conv1d(in_ch, hidden, 1)
        self.blocks = nn.ModuleList([
            TCNBlock(hidden, dilation=2**i) for i in range(layers)
        ])
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):  # x: (B, T, 1)
        h = self.input_proj(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        return self.head(h[:, :, -1])  # use last timestep

model = TCN()
x = torch.randn(8, 48, 1)
print(f"Forecast shape: {model(x).shape}")  # (8, 6)`}
      />

      <NoteBlock type="note" title="When to Choose TCN over RNN">
        <p>
          TCNs offer faster training via parallelism and stable gradients (no vanishing gradient
          problem). They are preferred when the effective context length is known and fixed.
          RNNs remain competitive when input lengths vary greatly or when online (streaming)
          inference with minimal memory is required.
        </p>
      </NoteBlock>
    </div>
  )
}

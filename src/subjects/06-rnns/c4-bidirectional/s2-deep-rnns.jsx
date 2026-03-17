import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

function StackedRNNDiagram() {
  const [layers, setLayers] = useState(3)
  const W = 400, H = 220
  const steps = 4

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Stacked RNN ({layers} layers)</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Layers: {layers}
        <input type="range" min={1} max={4} step={1} value={layers} onChange={e => setLayers(parseInt(e.target.value))} className="w-32 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        {Array.from({ length: layers }, (_, l) => {
          const y = H - 40 - l * 45
          const opacity = 0.5 + l * (0.5 / (layers - 1 || 1))
          return Array.from({ length: steps }, (_, t) => {
            const cx = 50 + t * 90
            return (
              <g key={`${l}-${t}`}>
                <rect x={cx - 18} y={y - 14} width={36} height={28} rx={4} fill="#8b5cf6" opacity={opacity} />
                <text x={cx} y={y + 4} textAnchor="middle" fill="white" fontSize={8}>L{l}</text>
                {t < steps - 1 && <line x1={cx + 18} y1={y} x2={cx + 72} y2={y} stroke="#a78bfa" strokeWidth={1} opacity={0.6} />}
                {l > 0 && <line x1={cx} y1={y + 14} x2={cx} y2={y + 31} stroke="#c4b5fd" strokeWidth={1} opacity={0.6} />}
              </g>
            )
          })
        })}
        {Array.from({ length: steps }, (_, t) => (
          <text key={t} x={50 + t * 90} y={H - 8} textAnchor="middle" fill="#6b7280" fontSize={10}>t={t}</text>
        ))}
        <text x={W / 2} y={14} textAnchor="middle" fill="#6b7280" fontSize={10}>output layer</text>
      </svg>
    </div>
  )
}

export default function DeepRNNs() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Stacking multiple RNN layers creates a deep recurrent architecture where each
        layer processes the output sequence of the layer below, enabling the network
        to learn hierarchical temporal representations.
      </p>

      <DefinitionBlock title="Stacked (Multi-Layer) RNN">
        <p>For a stack of <InlineMath math="L" /> layers, the hidden state at layer <InlineMath math="l" /> and time <InlineMath math="t" /> is:</p>
        <BlockMath math="h_t^{(l)} = f(W^{(l)} [h_{t-1}^{(l)}, h_t^{(l-1)}] + b^{(l)})" />
        <p className="mt-2">
          where <InlineMath math="h_t^{(0)} = x_t" /> is the input. Each layer has its own set of
          parameters, and the output of the top layer <InlineMath math="h_t^{(L)}" /> is used for predictions.
        </p>
      </DefinitionBlock>

      <StackedRNNDiagram />

      <TheoremBlock title="Residual Connections for Deep RNNs" id="residual-rnn">
        <p>For deep stacks (3+ layers), residual connections prevent gradient degradation:</p>
        <BlockMath math="h_t^{(l)} = f(W^{(l)} [h_{t-1}^{(l)}, h_t^{(l-1)}]) + h_t^{(l-1)}" />
        <p>
          This ensures that gradients can flow directly from the output to early layers,
          analogous to residual connections in deep feedforward networks. The identity shortcut
          makes it easy for the network to learn an identity mapping at each layer.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Typical Depth Guidelines">
        <p>
          <strong>2 layers</strong>: Standard choice for most tasks, significant improvement over 1 layer.
          <strong> 3-4 layers</strong>: Used in machine translation and speech recognition.
          <strong> 8+ layers</strong>: Rare for RNNs; usually requires residual connections, layer
          normalization, and careful initialization. Google's NMT system used 8 LSTM layers with residuals.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Deep LSTM with Residual Connections"
        code={`import torch
import torch.nn as nn

class ResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=4, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.layers.append(nn.LSTM(in_dim, hidden_size, batch_first=True))
            self.norms.append(nn.LayerNorm(hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(input_size, hidden_size)  # for first residual

    def forward(self, x):
        for i, (lstm, norm) in enumerate(zip(self.layers, self.norms)):
            out, _ = lstm(x)
            out = norm(out)
            out = self.dropout(out)
            if i == 0:
                x = out + self.proj(x)  # project input to hidden dim
            else:
                x = out + x  # residual connection
        return x

model = ResidualLSTM(64, 256, num_layers=4)
x = torch.randn(8, 50, 64)
out = model(x)
print(f"Output: {out.shape}")  # (8, 50, 256)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")`}
      />

      <NoteBlock type="note" title="Layer Normalization in RNNs">
        <p>
          Unlike batch normalization, <strong>layer normalization</strong> normalizes across the
          feature dimension within each time step, making it natural for variable-length sequences.
          Applied after each recurrent layer, it stabilizes training of deep RNNs and allows
          higher learning rates. It has become the default normalization for recurrent architectures.
        </p>
      </NoteBlock>
    </div>
  )
}

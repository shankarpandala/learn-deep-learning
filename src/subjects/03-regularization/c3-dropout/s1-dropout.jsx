import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function DropoutVisualizer() {
  const [dropRate, setDropRate] = useState(0.5)
  const [seed, setSeed] = useState(0)

  const layers = [4, 6, 6, 3]
  const W = 360, H = 220
  const layerX = layers.map((_, i) => 50 + i * ((W - 100) / (layers.length - 1)))

  const rng = (i, j) => Math.sin(seed * 1000 + i * 137 + j * 73) * 0.5 + 0.5
  const isDropped = (l, n) => l > 0 && l < layers.length - 1 && rng(l, n) < dropRate

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Dropout Visualization</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          p = {dropRate.toFixed(1)}
          <input type="range" min={0} max={0.9} step={0.1} value={dropRate} onChange={e => setDropRate(parseFloat(e.target.value))} className="w-32 accent-violet-500" />
        </label>
        <button onClick={() => setSeed(s => s + 1)} className="rounded bg-violet-500 px-3 py-1 text-xs text-white hover:bg-violet-600">Resample</button>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        {layers.map((n, l) => {
          const nodeY = Array.from({ length: n }, (_, i) => (H / 2) - ((n - 1) * 25) / 2 + i * 25)
          return nodeY.map((y, i) => {
            const dropped = isDropped(l, i)
            if (l < layers.length - 1) {
              const nextN = layers[l + 1]
              const nextY = Array.from({ length: nextN }, (_, j) => (H / 2) - ((nextN - 1) * 25) / 2 + j * 25)
              return nextY.map((ny, j) => {
                const nextDropped = isDropped(l + 1, j)
                if (dropped || nextDropped) return null
                return <line key={`e${l}${i}${j}`} x1={layerX[l]} y1={y} x2={layerX[l + 1]} y2={ny} stroke="#d1d5db" strokeWidth={0.5} />
              })
            }
            return null
          })
        })}
        {layers.map((n, l) => {
          const nodeY = Array.from({ length: n }, (_, i) => (H / 2) - ((n - 1) * 25) / 2 + i * 25)
          return nodeY.map((y, i) => {
            const dropped = isDropped(l, i)
            return (
              <circle key={`n${l}${i}`} cx={layerX[l]} cy={y} r={8}
                fill={dropped ? '#e5e7eb' : '#8b5cf6'} stroke={dropped ? '#9ca3af' : '#7c3aed'}
                strokeWidth={1.5} opacity={dropped ? 0.4 : 1} />
            )
          })
        })}
      </svg>
      <p className="text-xs text-center text-gray-500 mt-2">
        Active neurons: {layers.slice(1, -1).reduce((a, n, l) => a + Array.from({ length: n }, (_, i) => !isDropped(l + 1, i)).filter(Boolean).length, 0)} / {layers.slice(1, -1).reduce((a, n) => a + n, 0)} hidden
      </p>
    </div>
  )
}

export default function Dropout() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Dropout is one of the most effective and widely used regularization techniques.
        During training, it randomly deactivates neurons, preventing co-adaptation and
        acting as an implicit ensemble of subnetworks.
      </p>

      <DefinitionBlock title="Dropout">
        <p>During training, each hidden unit is independently set to zero with probability <InlineMath math="p" />:</p>
        <BlockMath math="\tilde{h}_i = \begin{cases} 0 & \text{with probability } p \\ \frac{h_i}{1-p} & \text{with probability } 1-p \end{cases}" />
        <p className="mt-2">The scaling by <InlineMath math="1/(1-p)" /> is called <strong>inverted dropout</strong> and ensures expected values match at test time.</p>
      </DefinitionBlock>

      <DropoutVisualizer />

      <TheoremBlock title="Ensemble Interpretation" id="dropout-ensemble">
        <p>
          A network with <InlineMath math="n" /> droppable units implicitly trains <InlineMath math="2^n" /> subnetworks
          that share weights. At test time, using all units with scaled weights approximates
          the geometric mean of all subnetwork predictions:
        </p>
        <BlockMath math="f_{\text{test}}(x) \approx \left(\prod_{m=1}^{2^n} f_m(x)^{1/2^n}\right)" />
      </TheoremBlock>

      <ExampleBlock title="Typical Dropout Rates">
        <ul className="list-disc ml-4 space-y-1">
          <li><strong>Input layer</strong>: <InlineMath math="p = 0.2" /> (drop 20% of inputs)</li>
          <li><strong>Hidden layers</strong>: <InlineMath math="p = 0.5" /> (the original default)</li>
          <li><strong>Transformers</strong>: <InlineMath math="p = 0.1" /> (attention and FFN sublayers)</li>
          <li><strong>Convolutional layers</strong>: often not used (use Spatial Dropout instead)</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Dropout in PyTorch"
        code={`import torch
import torch.nn as nn

class RegularizedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=drop_p),  # inverted dropout by default
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=drop_p),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

model = RegularizedMLP(784, 256, 10)
x = torch.randn(32, 784)

model.train()   # dropout active
out_train = model(x)

model.eval()    # dropout disabled, weights scaled
out_eval = model(x)

print(f"Train output variance: {out_train.var():.4f}")
print(f"Eval output variance: {out_eval.var():.4f}")`}
      />

      <NoteBlock type="note" title="Dropout and Batch Normalization">
        <p>
          Combining dropout with batch normalization requires care. The variance shift from
          dropout at train time vs test time can conflict with batch norm statistics. In
          practice, many modern architectures use batch norm <em>without</em> dropout, or
          apply dropout only after the final batch norm layer.
        </p>
      </NoteBlock>
    </div>
  )
}

import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function relu(x) { return Math.max(0, x) }
function leakyRelu(x, a) { return x >= 0 ? x : a * x }
function elu(x, a) { return x >= 0 ? x : a * (Math.exp(x) - 1) }

function ReLUPlot() {
  const [alpha, setAlpha] = useState(0.1)
  const W = 400, H = 250, ox = W / 2, oy = H * 0.65, sx = 30, sy = 30

  const range = Array.from({ length: 161 }, (_, i) => -4 + i * 0.05)
  const toSVG = (x, y) => `${ox + x * sx},${oy - y * sy}`

  const reluPath = range.map((x, i) => `${i === 0 ? 'M' : 'L'}${toSVG(x, relu(x))}`).join(' ')
  const leakyPath = range.map((x, i) => `${i === 0 ? 'M' : 'L'}${toSVG(x, leakyRelu(x, alpha))}`).join(' ')
  const eluPath = range.map((x, i) => `${i === 0 ? 'M' : 'L'}${toSVG(x, elu(x, alpha * 10))}`).join(' ')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">ReLU Family Comparison</h3>
      <label className="flex items-center gap-2 mb-3 text-sm text-gray-600 dark:text-gray-400">
        α = {alpha.toFixed(2)}
        <input type="range" min={0.01} max={0.5} step={0.01} value={alpha} onChange={e => setAlpha(parseFloat(e.target.value))} className="w-32 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={0} y1={oy} x2={W} y2={oy} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={ox} y1={0} x2={ox} y2={H} stroke="#d1d5db" strokeWidth={0.5} />
        <path d={reluPath} fill="none" stroke="#8b5cf6" strokeWidth={2.5} />
        <path d={leakyPath} fill="none" stroke="#f97316" strokeWidth={2} />
        <path d={eluPath} fill="none" stroke="#10b981" strokeWidth={2} />
      </svg>
      <div className="mt-2 flex justify-center gap-4 text-xs">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-violet-500" /> ReLU</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-orange-500" /> Leaky ReLU</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-emerald-500" /> ELU</span>
      </div>
    </div>
  )
}

export default function ReLUFamily() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The Rectified Linear Unit (ReLU) revolutionized deep learning by solving the vanishing
        gradient problem. Its simplicity and effectiveness made it the default activation for hidden layers.
      </p>

      <DefinitionBlock title="ReLU (Rectified Linear Unit)">
        <BlockMath math="\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}" />
        <p className="mt-2">Derivative: <InlineMath math="\text{ReLU}'(x) = \mathbb{1}[x > 0]" /> (1 for positive, 0 for negative inputs)</p>
      </DefinitionBlock>

      <DefinitionBlock title="Leaky ReLU & PReLU">
        <BlockMath math="\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}" />
        <p className="mt-2">
          <strong>Leaky ReLU</strong>: <InlineMath math="\alpha" /> is a fixed small constant (typically 0.01).
          <strong> PReLU</strong>: <InlineMath math="\alpha" /> is a learnable parameter.
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="ELU (Exponential Linear Unit)">
        <BlockMath math="\text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}" />
        <p className="mt-2">Smooth for negative values, pushing mean activations closer to zero.</p>
      </DefinitionBlock>

      <ReLUPlot />

      <WarningBlock title="Dying ReLU Problem">
        <p>
          If a neuron's weights lead to negative inputs for all training examples,
          its gradient is always zero and it stops learning entirely. This is called
          a <strong>dead neuron</strong>. Leaky ReLU and ELU avoid this by maintaining
          a small gradient for negative inputs.
        </p>
      </WarningBlock>

      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left border-collapse">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <th className="py-2 pr-4 font-semibold text-gray-900 dark:text-gray-100">Activation</th>
              <th className="py-2 pr-4 font-semibold text-gray-900 dark:text-gray-100">Range</th>
              <th className="py-2 pr-4 font-semibold text-gray-900 dark:text-gray-100">Dead Neurons?</th>
              <th className="py-2 font-semibold text-gray-900 dark:text-gray-100">Compute Cost</th>
            </tr>
          </thead>
          <tbody className="text-gray-700 dark:text-gray-300">
            <tr className="border-b border-gray-100 dark:border-gray-800"><td className="py-2 pr-4">ReLU</td><td className="py-2 pr-4">[0, ∞)</td><td className="py-2 pr-4">Yes</td><td className="py-2">Very low</td></tr>
            <tr className="border-b border-gray-100 dark:border-gray-800"><td className="py-2 pr-4">Leaky ReLU</td><td className="py-2 pr-4">(-∞, ∞)</td><td className="py-2 pr-4">No</td><td className="py-2">Very low</td></tr>
            <tr className="border-b border-gray-100 dark:border-gray-800"><td className="py-2 pr-4">PReLU</td><td className="py-2 pr-4">(-∞, ∞)</td><td className="py-2 pr-4">No</td><td className="py-2">Low</td></tr>
            <tr><td className="py-2 pr-4">ELU</td><td className="py-2 pr-4">(-α, ∞)</td><td className="py-2 pr-4">No</td><td className="py-2">Medium (exp)</td></tr>
          </tbody>
        </table>
      </div>

      <PythonCode
        title="ReLU Variants in PyTorch"
        code={`import torch
import torch.nn as nn

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

relu = nn.ReLU()
leaky = nn.LeakyReLU(0.1)
prelu = nn.PReLU(init=0.1)  # learnable alpha
elu = nn.ELU(alpha=1.0)

print(f"ReLU:      {relu(x).tolist()}")
print(f"LeakyReLU: {leaky(x).tolist()}")
print(f"PReLU:     {prelu(x).tolist()}")
print(f"ELU:       {[f'{v:.4f}' for v in elu(x).tolist()]}")`}
      />

      <NoteBlock variant="tip" title="When to Use What">
        <p>
          <strong>ReLU</strong> is the default choice for most architectures.
          Use <strong>Leaky ReLU</strong> if you observe many dead neurons.
          <strong>ELU</strong> can improve convergence in deep networks but costs more.
          Modern architectures like Transformers prefer <strong>GELU</strong> (next section).
        </p>
      </NoteBlock>
    </div>
  )
}

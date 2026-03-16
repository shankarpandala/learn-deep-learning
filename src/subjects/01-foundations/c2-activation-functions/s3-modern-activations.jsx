import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function sigmoid(x) { return 1 / (1 + Math.exp(-x)) }
function relu(x) { return Math.max(0, x) }
function gelu(x) { return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x))) }
function swish(x, beta) { return x * sigmoid(beta * x) }

function ModernPlot() {
  const [beta, setBeta] = useState(1.0)
  const W = 420, H = 260, ox = W / 2, oy = H * 0.6, sx = 35, sy = 35

  const range = Array.from({ length: 161 }, (_, i) => -4 + i * 0.05)
  const toSVG = (x, y) => `${ox + x * sx},${oy - y * sy}`

  const geluPath = range.map((x, i) => `${i === 0 ? 'M' : 'L'}${toSVG(x, gelu(x))}`).join(' ')
  const swishPath = range.map((x, i) => `${i === 0 ? 'M' : 'L'}${toSVG(x, swish(x, beta))}`).join(' ')
  const reluPath = range.map((x, i) => `${i === 0 ? 'M' : 'L'}${toSVG(x, relu(x))}`).join(' ')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">GELU vs Swish vs ReLU</h3>
      <label className="flex items-center gap-2 mb-3 text-sm text-gray-600 dark:text-gray-400">
        Swish β = {beta.toFixed(1)}
        <input type="range" min={0.1} max={5} step={0.1} value={beta} onChange={e => setBeta(parseFloat(e.target.value))} className="w-32 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={0} y1={oy} x2={W} y2={oy} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={ox} y1={0} x2={ox} y2={H} stroke="#d1d5db" strokeWidth={0.5} />
        <path d={reluPath} fill="none" stroke="#9ca3af" strokeWidth={1.5} strokeDasharray="4,4" />
        <path d={geluPath} fill="none" stroke="#8b5cf6" strokeWidth={2.5} />
        <path d={swishPath} fill="none" stroke="#f97316" strokeWidth={2.5} />
      </svg>
      <div className="mt-2 flex justify-center gap-4 text-xs">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-violet-500" /> GELU</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-orange-500" /> Swish(β={beta.toFixed(1)})</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-gray-400" style={{borderTop: '1px dashed'}} /> ReLU</span>
      </div>
    </div>
  )
}

export default function ModernActivations() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Modern architectures like Transformers and EfficientNet use smooth activation functions
        that outperform ReLU in practice. GELU and Swish/SiLU are now standard in state-of-the-art models.
      </p>

      <DefinitionBlock title="GELU (Gaussian Error Linear Unit)">
        <BlockMath math="\text{GELU}(x) = x \cdot \Phi(x)" />
        <p className="mt-2">
          where <InlineMath math="\Phi(x)" /> is the CDF of the standard normal distribution.
          Practical approximation:
        </p>
        <BlockMath math="\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)" />
      </DefinitionBlock>

      <DefinitionBlock title="Swish / SiLU">
        <BlockMath math="\text{Swish}(x) = x \cdot \sigma(\beta x) = \frac{x}{1 + e^{-\beta x}}" />
        <p className="mt-2">
          When <InlineMath math="\beta = 1" />, this is called <strong>SiLU</strong> (Sigmoid Linear Unit).
          As <InlineMath math="\beta \to \infty" />, Swish approaches ReLU.
          As <InlineMath math="\beta \to 0" />, Swish approaches the linear function <InlineMath math="x/2" />.
        </p>
      </DefinitionBlock>

      <ModernPlot />

      <NoteBlock type="historical" title="Discovery & Adoption">
        <p>
          <strong>Swish</strong> was discovered by Google Brain (2017) through automated search
          over activation function spaces. <strong>GELU</strong> was proposed by Hendrycks & Gimpel (2016).
          GELU is the default in BERT, GPT, and most Transformer architectures.
          Swish/SiLU is used in EfficientNet, Mish in YOLOv4.
        </p>
      </NoteBlock>

      <h2 className="text-xl font-bold text-gray-900 dark:text-white mt-8">Why Smooth Activations Work Better</h2>

      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Unlike ReLU, GELU and Swish are smooth everywhere and allow small negative values.
        This creates a <strong>non-monotonic</strong> region near zero that acts as a soft gate,
        allowing the network to modulate information flow more smoothly.
      </p>

      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left border-collapse">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <th className="py-2 pr-4 font-semibold text-gray-900 dark:text-gray-100">Activation</th>
              <th className="py-2 pr-4 font-semibold text-gray-900 dark:text-gray-100">Used In</th>
              <th className="py-2 font-semibold text-gray-900 dark:text-gray-100">Key Property</th>
            </tr>
          </thead>
          <tbody className="text-gray-700 dark:text-gray-300">
            <tr className="border-b border-gray-100 dark:border-gray-800"><td className="py-2 pr-4">GELU</td><td className="py-2 pr-4">BERT, GPT, ViT</td><td className="py-2">Probabilistic gating</td></tr>
            <tr className="border-b border-gray-100 dark:border-gray-800"><td className="py-2 pr-4">SiLU/Swish</td><td className="py-2 pr-4">EfficientNet, LLaMA</td><td className="py-2">Self-gated, smooth</td></tr>
            <tr><td className="py-2 pr-4">Mish</td><td className="py-2 pr-4">YOLOv4</td><td className="py-2">x·tanh(softplus(x))</td></tr>
          </tbody>
        </table>
      </div>

      <PythonCode
        title="Modern Activations in PyTorch"
        code={`import torch
import torch.nn as nn

x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

gelu = nn.GELU()
silu = nn.SiLU()  # Swish with beta=1

print("GELU:", [f"{v:.4f}" for v in gelu(x).tolist()])
print("SiLU:", [f"{v:.4f}" for v in silu(x).tolist()])

# In a Transformer FFN block:
class TransformerFFN(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()  # standard in Transformers

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))`}
      />

      <NoteBlock type="tip" title="Practical Advice">
        <p>
          For <strong>Transformer-based models</strong>, use GELU.
          For <strong>CNNs</strong>, ReLU or SiLU work well.
          For <strong>general use</strong>, SiLU is a safe, modern default that rarely underperforms ReLU.
        </p>
      </NoteBlock>
    </div>
  )
}

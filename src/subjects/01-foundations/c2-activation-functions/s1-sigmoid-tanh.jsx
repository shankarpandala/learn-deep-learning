import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function sigmoid(x) { return 1 / (1 + Math.exp(-x)) }
function tanh_fn(x) { return Math.tanh(x) }
function sigmoidDeriv(x) { const s = sigmoid(x); return s * (1 - s) }
function tanhDeriv(x) { const t = tanh_fn(x); return 1 - t * t }

function ActivationPlot() {
  const [showDeriv, setShowDeriv] = useState(false)
  const [probe, setProbe] = useState(0)
  const W = 400, H = 250, ox = W / 2, oy = H / 2, sx = 30, sy = 80

  const range = Array.from({ length: 161 }, (_, i) => -4 + i * 0.05)

  function toSVG(x, y) { return `${ox + x * sx},${oy - y * sy}` }

  const sigPath = range.map((x, i) => `${i === 0 ? 'M' : 'L'}${toSVG(x, sigmoid(x))}`).join(' ')
  const tanhPath = range.map((x, i) => `${i === 0 ? 'M' : 'L'}${toSVG(x, tanh_fn(x))}`).join(' ')
  const sigDerivPath = range.map((x, i) => `${i === 0 ? 'M' : 'L'}${toSVG(x, sigmoidDeriv(x))}`).join(' ')
  const tanhDerivPath = range.map((x, i) => `${i === 0 ? 'M' : 'L'}${toSVG(x, tanhDeriv(x))}`).join(' ')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Sigmoid & Tanh Plot</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          <input type="checkbox" checked={showDeriv} onChange={e => setShowDeriv(e.target.checked)} className="accent-violet-500" />
          Show derivatives
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          x = {probe.toFixed(1)}
          <input type="range" min={-4} max={4} step={0.1} value={probe} onChange={e => setProbe(parseFloat(e.target.value))} className="w-32 accent-violet-500" />
        </label>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={0} y1={oy} x2={W} y2={oy} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={ox} y1={0} x2={ox} y2={H} stroke="#d1d5db" strokeWidth={0.5} />
        <path d={sigPath} fill="none" stroke="#8b5cf6" strokeWidth={2.5} />
        <path d={tanhPath} fill="none" stroke="#f97316" strokeWidth={2.5} />
        {showDeriv && <path d={sigDerivPath} fill="none" stroke="#8b5cf6" strokeWidth={1.5} strokeDasharray="4,4" opacity={0.6} />}
        {showDeriv && <path d={tanhDerivPath} fill="none" stroke="#f97316" strokeWidth={1.5} strokeDasharray="4,4" opacity={0.6} />}
        <line x1={ox + probe * sx} y1={0} x2={ox + probe * sx} y2={H} stroke="#9ca3af" strokeWidth={0.8} strokeDasharray="3,3" />
        <circle cx={ox + probe * sx} cy={oy - sigmoid(probe) * sy} r={4} fill="#8b5cf6" />
        <circle cx={ox + probe * sx} cy={oy - tanh_fn(probe) * sy} r={4} fill="#f97316" />
      </svg>
      <div className="mt-2 flex justify-center gap-6 text-xs">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-violet-500" /> σ(x) = {sigmoid(probe).toFixed(4)}</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-orange-500" /> tanh(x) = {tanh_fn(probe).toFixed(4)}</span>
      </div>
    </div>
  )
}

export default function SigmoidTanh() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Sigmoid and tanh were the dominant activation functions in early neural networks.
        Understanding their properties and limitations is essential background for modern deep learning.
      </p>

      <DefinitionBlock title="Sigmoid Function">
        <BlockMath math="\sigma(x) = \frac{1}{1 + e^{-x}}" />
        <p className="mt-2">Range: <InlineMath math="(0, 1)" />. Derivative: <InlineMath math="\sigma'(x) = \sigma(x)(1 - \sigma(x))" /></p>
      </DefinitionBlock>

      <DefinitionBlock title="Hyperbolic Tangent">
        <BlockMath math="\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1" />
        <p className="mt-2">Range: <InlineMath math="(-1, 1)" />. Derivative: <InlineMath math="\tanh'(x) = 1 - \tanh^2(x)" /></p>
      </DefinitionBlock>

      <ActivationPlot />

      <TheoremBlock title="Vanishing Gradient Problem" id="vanishing-gradient">
        <p>
          For both sigmoid and tanh, the maximum derivative value is:
        </p>
        <BlockMath math="\max_x \sigma'(x) = 0.25, \quad \max_x \tanh'(x) = 1" />
        <p>
          In a deep network with <InlineMath math="L" /> layers, gradients are multiplied
          through the chain rule. With sigmoid, gradients shrink by at least a factor
          of 4 per layer: <InlineMath math="\|\nabla\| \leq 0.25^L" />, causing
          vanishing gradients in deep networks.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Gradient Through 10 Sigmoid Layers">
        <p>Maximum gradient magnitude after 10 layers:</p>
        <BlockMath math="0.25^{10} = 9.5 \times 10^{-7}" />
        <p>The gradient has effectively vanished — the early layers learn almost nothing.</p>
      </ExampleBlock>

      <PythonCode
        title="Sigmoid & Tanh in PyTorch"
        code={`import torch
import torch.nn as nn

x = torch.linspace(-4, 4, 100, requires_grad=True)

# Sigmoid
sigmoid = nn.Sigmoid()
y_sig = sigmoid(x)
y_sig.sum().backward()
print(f"σ(0) = {sigmoid(torch.tensor(0.0)):.4f}")
print(f"σ'(0) = {x.grad[50]:.4f}")  # max derivative at x=0

x.grad.zero_()

# Tanh
tanh = nn.Tanh()
y_tanh = tanh(x)
y_tanh.sum().backward()
print(f"tanh(0) = {tanh(torch.tensor(0.0)):.4f}")
print(f"tanh'(0) = {x.grad[50]:.4f}")`}
      />

      <NoteBlock type="note" title="When to Use Sigmoid vs Tanh">
        <p>
          <strong>Sigmoid</strong> is still used for output layers in binary classification
          (mapping to probabilities). <strong>Tanh</strong> is preferred over sigmoid for
          hidden layers because it is zero-centered, leading to faster convergence.
          However, both have been largely replaced by ReLU and its variants in hidden layers.
        </p>
      </NoteBlock>
    </div>
  )
}

import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ChainRuleDepthViz() {
  const [depth, setDepth] = useState(3)
  const gradFactor = 0.25
  const gradMag = Math.pow(gradFactor, depth)

  const barWidth = 320
  const maxBars = 8

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Chain Rule Gradient Magnitude vs Depth</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Layers: {depth}
        <input type="range" min={1} max={maxBars} step={1} value={depth} onChange={e => setDepth(parseInt(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <svg width={400} height={maxBars * 30 + 10} className="mx-auto block">
        {Array.from({ length: maxBars }, (_, i) => {
          const mag = Math.pow(gradFactor, i + 1)
          const w = Math.max(2, mag * barWidth)
          const active = i < depth
          return (
            <g key={i}>
              <rect x={10} y={i * 30 + 5} width={w} height={22} rx={3} fill={active ? '#8b5cf6' : '#e5e7eb'} opacity={active ? 1 : 0.4} />
              <text x={w + 16} y={i * 30 + 20} fontSize={11} fill={active ? '#5b21b6' : '#9ca3af'}>
                L={i + 1}: {mag.toExponential(1)}
              </text>
            </g>
          )
        })}
      </svg>
      <p className="text-center text-sm text-violet-600 dark:text-violet-400 mt-2">
        Gradient at layer 1 after {depth} layers: {gradMag.toExponential(2)}
      </p>
    </div>
  )
}

export default function ChainRuleReview() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The chain rule is the mathematical foundation of backpropagation. In deep networks, it
        describes how gradients flow through compositions of many functions, one per layer.
      </p>

      <TheoremBlock title="Multivariate Chain Rule" id="multivariate-chain-rule">
        <p>
          For composed functions <InlineMath math="L = f_L \circ f_{L-1} \circ \cdots \circ f_1(x)" />,
          the gradient with respect to input <InlineMath math="x" /> is:
        </p>
        <BlockMath math="\frac{\partial L}{\partial x} = \frac{\partial f_L}{\partial f_{L-1}} \cdot \frac{\partial f_{L-1}}{\partial f_{L-2}} \cdots \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial x}" />
        <p>This is a product of <InlineMath math="L" /> Jacobian matrices.</p>
      </TheoremBlock>

      <DefinitionBlock title="Layer-wise Gradient">
        <p>
          For layer <InlineMath math="l" /> with parameters <InlineMath math="W_l" /> and
          forward <InlineMath math="h_l = f_l(W_l, h_{l-1})" />, the parameter gradient is:
        </p>
        <BlockMath math="\frac{\partial L}{\partial W_l} = \frac{\partial L}{\partial h_l} \cdot \frac{\partial h_l}{\partial W_l}" />
      </DefinitionBlock>

      <ExampleBlock title="Two-Layer Network Gradient">
        <p>
          For <InlineMath math="y = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2" /> with MSE loss:
        </p>
        <BlockMath math="\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial y} \cdot W_2^\top \cdot \text{diag}(\mathbf{1}[W_1 x + b_1 > 0]) \cdot x^\top" />
        <p>
          Each factor corresponds to one layer's local Jacobian, multiplied from output back to input.
        </p>
      </ExampleBlock>

      <ChainRuleDepthViz />

      <PythonCode
        title="Chain Rule Across Layers in PyTorch"
        code={`import torch
import torch.nn as nn

# 3-layer network
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

x = torch.randn(1, 4)
y = model(x)
y.backward()

# Inspect gradient magnitudes at each layer
for i, layer in enumerate(model):
    if hasattr(layer, 'weight'):
        grad_norm = layer.weight.grad.norm().item()
        print(f"Layer {i} weight grad norm: {grad_norm:.6f}")`}
      />

      <NoteBlock type="note" title="Gradient Is a Product of Matrices">
        <p>
          The key insight is that the gradient through <InlineMath math="L" /> layers is a product
          of <InlineMath math="L" /> matrices. If most factors have spectral norm{' '}
          <InlineMath math="< 1" />, the product shrinks exponentially (vanishing). If most have
          spectral norm <InlineMath math="> 1" />, it grows exponentially (exploding).
        </p>
      </NoteBlock>
    </div>
  )
}

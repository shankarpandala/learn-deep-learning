import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ComputationGraphDiagram() {
  const [inputA, setInputA] = useState(2)
  const [inputB, setInputB] = useState(3)
  const mul = inputA * inputB
  const added = mul + 1
  const output = 1 / (1 + Math.exp(-added))

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Interactive Computation Graph</h3>
      <div className="flex items-center gap-4 mb-4">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          a = {inputA.toFixed(1)}
          <input type="range" min={-3} max={5} step={0.1} value={inputA} onChange={e => setInputA(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          b = {inputB.toFixed(1)}
          <input type="range" min={-3} max={5} step={0.1} value={inputB} onChange={e => setInputB(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <svg width={460} height={140} className="mx-auto block">
        <rect x={10} y={20} width={60} height={36} rx={6} fill="#ede9fe" stroke="#8b5cf6" strokeWidth={1.5} />
        <text x={40} y={43} textAnchor="middle" fontSize={13} fill="#5b21b6">a={inputA.toFixed(1)}</text>
        <rect x={10} y={85} width={60} height={36} rx={6} fill="#ede9fe" stroke="#8b5cf6" strokeWidth={1.5} />
        <text x={40} y={108} textAnchor="middle" fontSize={13} fill="#5b21b6">b={inputB.toFixed(1)}</text>
        <line x1={70} y1={38} x2={120} y2={60} stroke="#a78bfa" strokeWidth={1.5} />
        <line x1={70} y1={103} x2={120} y2={75} stroke="#a78bfa" strokeWidth={1.5} />
        <rect x={120} y={50} width={50} height={30} rx={6} fill="#f5f3ff" stroke="#7c3aed" strokeWidth={1.5} />
        <text x={145} y={70} textAnchor="middle" fontSize={12} fill="#6d28d9">*</text>
        <line x1={170} y1={65} x2={220} y2={65} stroke="#a78bfa" strokeWidth={1.5} />
        <text x={195} y={58} textAnchor="middle" fontSize={11} fill="#7c3aed">{mul.toFixed(1)}</text>
        <rect x={220} y={50} width={50} height={30} rx={6} fill="#f5f3ff" stroke="#7c3aed" strokeWidth={1.5} />
        <text x={245} y={70} textAnchor="middle" fontSize={12} fill="#6d28d9">+1</text>
        <line x1={270} y1={65} x2={320} y2={65} stroke="#a78bfa" strokeWidth={1.5} />
        <text x={295} y={58} textAnchor="middle" fontSize={11} fill="#7c3aed">{added.toFixed(1)}</text>
        <rect x={320} y={50} width={50} height={30} rx={6} fill="#f5f3ff" stroke="#7c3aed" strokeWidth={1.5} />
        <text x={345} y={70} textAnchor="middle" fontSize={11} fill="#6d28d9">sig</text>
        <line x1={370} y1={65} x2={420} y2={65} stroke="#a78bfa" strokeWidth={1.5} />
        <text x={435} y={70} textAnchor="middle" fontSize={12} fontWeight="bold" fill="#5b21b6">{output.toFixed(4)}</text>
      </svg>
    </div>
  )
}

export default function ForwardPass() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The forward pass computes a neural network's output by propagating inputs through a directed
        acyclic graph of operations. Each node performs a local computation and passes its result downstream.
      </p>

      <DefinitionBlock title="Computation Graph">
        <p>
          A <strong>computation graph</strong> is a directed acyclic graph where each node represents an operation
          (addition, multiplication, activation) and edges carry intermediate values. Given inputs{' '}
          <InlineMath math="x_1, \dots, x_n" />, the forward pass evaluates nodes in topological order:
        </p>
        <BlockMath math="v_i = f_i(v_{\text{parents}(i)})" />
      </DefinitionBlock>

      <ExampleBlock title="Forward Pass Through a Simple Graph">
        <p>Consider <InlineMath math="y = \sigma(a \cdot b + 1)" />. The forward pass proceeds:</p>
        <BlockMath math="v_1 = a \cdot b, \quad v_2 = v_1 + 1, \quad y = \sigma(v_2) = \frac{1}{1 + e^{-v_2}}" />
        <p>
          For <InlineMath math="a=2, b=3" />: <InlineMath math="v_1=6" />, <InlineMath math="v_2=7" />,{' '}
          <InlineMath math="y = \sigma(7) \approx 0.9991" />.
        </p>
      </ExampleBlock>

      <ComputationGraphDiagram />

      <NoteBlock type="note" title="Topological Ordering">
        <p>
          The forward pass requires evaluating nodes in <strong>topological order</strong> so that every
          node's inputs are computed before it executes. Modern frameworks like PyTorch build this graph
          dynamically (define-by-run), while TensorFlow 1.x used static graphs (define-then-run).
        </p>
      </NoteBlock>

      <PythonCode
        title="Forward Pass in PyTorch"
        code={`import torch
import torch.nn as nn

# Simple computation graph: y = sigmoid(a * b + 1)
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

# Forward pass — PyTorch builds the graph automatically
v1 = a * b          # multiply node
v2 = v1 + 1         # add node
y = torch.sigmoid(v2)  # sigmoid node

print(f"v1 = a * b = {v1.item():.1f}")
print(f"v2 = v1 + 1 = {v2.item():.1f}")
print(f"y = sigmoid(v2) = {y.item():.4f}")

# A full linear layer forward pass
layer = nn.Linear(4, 2)
x = torch.randn(3, 4)  # batch of 3, 4 features
output = layer(x)       # forward: x @ W^T + b
print(f"Output shape: {output.shape}")`}
      />

      <NoteBlock type="note" title="Intermediate Values Are Cached">
        <p>
          During the forward pass, all intermediate values (<InlineMath math="v_1, v_2, \dots" />) are
          stored on a <strong>tape</strong>. These cached values are essential for computing gradients
          in the backward pass. This is the fundamental memory-compute trade-off of backpropagation.
        </p>
      </NoteBlock>
    </div>
  )
}

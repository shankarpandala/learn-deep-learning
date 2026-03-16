import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

const GATES = {
  AND: { points: [[0,0,-1],[0,1,-1],[1,0,-1],[1,1,1]], separable: true },
  OR:  { points: [[0,0,-1],[0,1,1],[1,0,1],[1,1,1]], separable: true },
  XOR: { points: [[0,0,-1],[0,1,1],[1,0,1],[1,1,-1]], separable: false },
}

function GateViz() {
  const [gate, setGate] = useState('AND')
  const { points, separable } = GATES[gate]
  const s = 120, ox = 60, oy = 180

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Logic Gate Decision Boundaries</h3>
      <div className="flex gap-2 mb-4">
        {Object.keys(GATES).map(g => (
          <button key={g} onClick={() => setGate(g)}
            className={`rounded-lg px-4 py-1.5 text-sm font-medium transition-colors ${gate === g ? 'bg-violet-600 text-white' : 'border border-gray-300 text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:text-gray-300'}`}>
            {g}
          </button>
        ))}
      </div>
      <svg width={260} height={220} className="mx-auto block">
        <rect x={ox-10} y={oy-s-10} width={s+20} height={s+20} fill="#f9fafb" stroke="#e5e7eb" rx={4} className="dark:fill-gray-800/30 dark:stroke-gray-700" />
        {separable && gate === 'AND' && <line x1={ox+s*0.3} y1={oy+10} x2={ox+s+10} y2={oy-s*0.7} stroke="#8b5cf6" strokeWidth={2} strokeDasharray="6,3" />}
        {separable && gate === 'OR' && <line x1={ox-10} y1={oy-s*0.3} x2={ox+s*0.7} y2={oy-s-10} stroke="#8b5cf6" strokeWidth={2} strokeDasharray="6,3" />}
        {!separable && <text x={ox+s/2} y={oy+30} textAnchor="middle" fontSize={11} fill="#ef4444" fontWeight="bold">No linear boundary exists!</text>}
        {points.map(([x, y, label], i) => (
          <circle key={i} cx={ox + x * s} cy={oy - y * s} r={10}
            fill={label === 1 ? '#8b5cf6' : '#ef4444'} stroke="white" strokeWidth={2} />
        ))}
      </svg>
      <p className="mt-2 text-center text-sm text-gray-500 dark:text-gray-400">
        {separable ? '✓ Linearly separable' : '✗ NOT linearly separable'}
      </p>
    </div>
  )
}

export default function LinearSeparability() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The perceptron is powerful but has a fundamental limitation: it can only solve
        problems where the two classes can be separated by a single hyperplane.
      </p>

      <DefinitionBlock title="Linear Separability">
        <p>
          A dataset <InlineMath math="\{(\mathbf{x}_i, y_i)\}" /> is <strong>linearly separable</strong> if
          there exists a weight vector <InlineMath math="\mathbf{w}" /> and bias <InlineMath math="b" /> such that:
        </p>
        <BlockMath math="y_i(\mathbf{w}^\top \mathbf{x}_i + b) > 0 \quad \forall i" />
      </DefinitionBlock>

      <GateViz />

      <TheoremBlock title="XOR Impossibility" id="xor-impossibility">
        <p>
          No single perceptron can compute the XOR function. To see this, note that XOR
          requires <InlineMath math="(0,0) \to 0" /> and <InlineMath math="(1,1) \to 0" /> (same class)
          but <InlineMath math="(0,1) \to 1" /> and <InlineMath math="(1,0) \to 1" /> (different class).
          No single line in 2D can separate these four points into two correct groups.
        </p>
      </TheoremBlock>

      <NoteBlock variant="historical" title="The AI Winter">
        <p>
          In 1969, Minsky and Papert published <em>Perceptrons</em>, proving that
          single-layer perceptrons cannot solve XOR or any non-linearly-separable problem.
          This contributed to a decline in neural network research known as the
          <strong> first AI Winter</strong>. The solution — multi-layer networks trained
          with backpropagation — took nearly two decades to become practical.
        </p>
      </NoteBlock>

      <h2 className="text-xl font-bold text-gray-900 dark:text-white mt-8">Solving XOR with Multiple Layers</h2>

      <ExampleBlock title="XOR via Two Layers">
        <p>XOR can be decomposed as:</p>
        <BlockMath math="\text{XOR}(x_1, x_2) = \text{AND}(\text{OR}(x_1, x_2), \text{NAND}(x_1, x_2))" />
        <p className="mt-2">
          Layer 1 computes OR and NAND in parallel. Layer 2 combines them with AND.
          This is the key insight: <strong>stacking layers enables solving non-linear problems</strong>.
        </p>
      </ExampleBlock>

      <PythonCode
        title="XOR with a 2-Layer Network"
        code={`import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Hand-crafted weights for XOR
# Layer 1: OR neuron + NAND neuron
W1 = np.array([[20, 20], [-20, -20]])  # 2 neurons, 2 inputs
b1 = np.array([-10, 30])

# Layer 2: AND neuron
W2 = np.array([[20, 20]])  # 1 neuron, 2 inputs
b2 = np.array([-30])

def xor_network(x):
    h = sigmoid(W1 @ x + b1)  # hidden layer
    o = sigmoid(W2 @ h + b2)  # output layer
    return o[0]

# Test all inputs
for x1, x2 in [(0,0), (0,1), (1,0), (1,1)]:
    x = np.array([x1, x2])
    y = xor_network(x)
    print(f"XOR({x1}, {x2}) = {y:.4f} ≈ {round(y)}")`}
      />

      <NoteBlock variant="tip" title="The Power of Depth">
        <p>
          This example demonstrates the fundamental principle of deep learning:
          by composing simple functions (neurons) in layers, we can approximate
          arbitrarily complex functions. Each layer builds higher-level features
          from the previous layer's outputs.
        </p>
      </NoteBlock>
    </div>
  )
}

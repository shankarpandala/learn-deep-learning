import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

/* ------------------------------------------------------------------ */
/*  Logic gate data and visualization                                  */
/* ------------------------------------------------------------------ */
const GATES = {
  AND: {
    points: [
      { x: 0, y: 0, label: -1 },
      { x: 0, y: 1, label: -1 },
      { x: 1, y: 0, label: -1 },
      { x: 1, y: 1, label: 1 },
    ],
    separable: true,
    // w1*x + w2*y + b = 0 => decision line
    boundary: { x1: -0.2, y1: 1.5, x2: 1.5, y2: -0.2 },
    truth: ['0 AND 0 = 0', '0 AND 1 = 0', '1 AND 0 = 0', '1 AND 1 = 1'],
  },
  OR: {
    points: [
      { x: 0, y: 0, label: -1 },
      { x: 0, y: 1, label: 1 },
      { x: 1, y: 0, label: 1 },
      { x: 1, y: 1, label: 1 },
    ],
    separable: true,
    boundary: { x1: -0.2, y1: 0.5, x2: 0.5, y2: -0.2 },
    truth: ['0 OR 0 = 0', '0 OR 1 = 1', '1 OR 0 = 1', '1 OR 1 = 1'],
  },
  XOR: {
    points: [
      { x: 0, y: 0, label: -1 },
      { x: 0, y: 1, label: 1 },
      { x: 1, y: 0, label: 1 },
      { x: 1, y: 1, label: -1 },
    ],
    separable: false,
    boundary: null,
    truth: ['0 XOR 0 = 0', '0 XOR 1 = 1', '1 XOR 0 = 1', '1 XOR 1 = 0'],
  },
}

function GateVisualizer() {
  const [gate, setGate] = useState('AND')
  const current = GATES[gate]
  const scale = 140
  const ox = 80
  const oy = 200

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">
        Logic Gate Decision Boundaries
      </h3>
      <p className="mb-3 text-sm text-gray-500 dark:text-gray-400">
        Toggle between AND, OR, and XOR to see which can be separated by a single line.
      </p>

      {/* Gate selector buttons */}
      <div className="flex gap-2 mb-4">
        {Object.keys(GATES).map(g => (
          <button key={g} onClick={() => setGate(g)}
            className={`rounded-lg px-5 py-1.5 text-sm font-semibold transition-colors ${
              gate === g
                ? 'bg-indigo-600 text-white shadow'
                : 'border border-gray-300 text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-800'
            }`}>
            {g}
          </button>
        ))}
      </div>

      <div className="flex flex-col sm:flex-row items-center gap-6">
        {/* SVG plot */}
        <svg width={300} height={260} className="shrink-0 block" aria-label={`${gate} gate decision boundary`}>
          {/* Background grid */}
          <rect x={ox - 20} y={oy - scale - 20} width={scale + 40} height={scale + 40}
            fill="#f9fafb" stroke="#e5e7eb" rx="6" className="dark:fill-gray-800/30 dark:stroke-gray-700" />

          {/* Grid lines */}
          <line x1={ox} y1={oy - scale - 15} x2={ox} y2={oy + 15} stroke="#d1d5db" strokeWidth="0.5" />
          <line x1={ox + scale} y1={oy - scale - 15} x2={ox + scale} y2={oy + 15} stroke="#d1d5db" strokeWidth="0.5" />
          <line x1={ox - 15} y1={oy} x2={ox + scale + 15} y2={oy} stroke="#d1d5db" strokeWidth="0.5" />
          <line x1={ox - 15} y1={oy - scale} x2={ox + scale + 15} y2={oy - scale} stroke="#d1d5db" strokeWidth="0.5" />

          {/* Axis labels */}
          <text x={ox} y={oy + 22} textAnchor="middle" className="text-[9px] fill-gray-400">0</text>
          <text x={ox + scale} y={oy + 22} textAnchor="middle" className="text-[9px] fill-gray-400">1</text>
          <text x={ox - 16} y={oy + 4} textAnchor="middle" className="text-[9px] fill-gray-400">0</text>
          <text x={ox - 16} y={oy - scale + 4} textAnchor="middle" className="text-[9px] fill-gray-400">1</text>
          <text x={ox + scale / 2} y={oy + 38} textAnchor="middle" className="text-[10px] fill-gray-500">x1</text>
          <text x={ox - 32} y={oy - scale / 2} textAnchor="middle" className="text-[10px] fill-gray-500">x2</text>

          {/* Decision boundary (if separable) */}
          {current.boundary && (
            <line
              x1={ox + current.boundary.x1 * scale}
              y1={oy - current.boundary.y1 * scale}
              x2={ox + current.boundary.x2 * scale}
              y2={oy - current.boundary.y2 * scale}
              stroke="#8b5cf6" strokeWidth="2.5" strokeDasharray="6,3" />
          )}

          {/* Shading for "no boundary" */}
          {!current.separable && (
            <text x={ox + scale / 2} y={oy + 50} textAnchor="middle"
              className="text-[11px] font-bold" fill="#ef4444">
              No single line can separate these!
            </text>
          )}

          {/* Data points */}
          {current.points.map((pt, i) => (
            <g key={i}>
              <circle
                cx={ox + pt.x * scale} cy={oy - pt.y * scale} r="12"
                fill={pt.label === 1 ? '#6366f1' : '#ef4444'}
                stroke="white" strokeWidth="2.5" />
              <text x={ox + pt.x * scale} y={oy - pt.y * scale + 1}
                textAnchor="middle" dominantBaseline="middle"
                className="text-[10px] font-bold" fill="white">
                {pt.label === 1 ? '1' : '0'}
              </text>
            </g>
          ))}
        </svg>

        {/* Truth table */}
        <div className="text-sm">
          <p className="font-semibold text-gray-700 dark:text-gray-300 mb-2">{gate} Truth Table:</p>
          <table className="border-collapse text-xs">
            <thead>
              <tr className="border-b border-gray-300 dark:border-gray-600">
                <th className="px-3 py-1 text-gray-500">x1</th>
                <th className="px-3 py-1 text-gray-500">x2</th>
                <th className="px-3 py-1 text-gray-500">{gate}</th>
              </tr>
            </thead>
            <tbody className="text-gray-700 dark:text-gray-300 font-mono">
              {current.points.map((pt, i) => (
                <tr key={i} className="border-b border-gray-200 dark:border-gray-700">
                  <td className="px-3 py-1 text-center">{pt.x}</td>
                  <td className="px-3 py-1 text-center">{pt.y}</td>
                  <td className={`px-3 py-1 text-center font-bold ${pt.label === 1 ? 'text-indigo-600 dark:text-indigo-400' : 'text-red-500'}`}>
                    {pt.label === 1 ? 1 : 0}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 text-xs text-gray-500 dark:text-gray-400">
            {current.separable
              ? <span className="text-green-600 dark:text-green-400 font-semibold">Linearly separable -- a single perceptron suffices.</span>
              : <span className="text-red-500 font-semibold">NOT linearly separable -- a single perceptron cannot learn this.</span>
            }
          </p>
        </div>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Main Section Component                                             */
/* ------------------------------------------------------------------ */
export default function LinearSeparability() {
  return (
    <div className="space-y-6">
      {/* --- Introduction --- */}
      <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100">
        Linear Separability & Limits
      </h2>
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The perceptron is elegant, but it has a fundamental limitation: it can only learn to
        classify data that is <em>linearly separable</em>. This section explores what that
        means, why XOR breaks the perceptron, and how stacking layers overcomes this barrier.
      </p>

      {/* --- Definition --- */}
      <DefinitionBlock
        title="Linear Separability"
        label="Definition 1.3"
        definition="A dataset $\{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ with $y_i \in \{-1, +1\}$ is linearly separable if there exists a weight vector $\mathbf{w} \in \mathbb{R}^n$ and bias $b \in \mathbb{R}$ such that $y_i(\mathbf{w}^\top \mathbf{x}_i + b) > 0$ for all $i$. Geometrically, this means there exists a hyperplane that perfectly separates the two classes with a positive margin."
        notation="In 2D, the separating hyperplane is a line. In 3D, it is a plane. In $n$ dimensions, it is an $(n{-}1)$-dimensional hyperplane."
      />

      {/* --- Interactive Gate Visualization --- */}
      <GateVisualizer />

      {/* --- Impossibility Theorem --- */}
      <TheoremBlock
        title="Single-Layer Impossibility"
        label="Theorem 1.2"
        statement="A single perceptron (one neuron with a step activation) can only represent linearly separable functions. In particular, it cannot compute the XOR function."
        proof="Suppose for contradiction that a perceptron computes XOR. Then there exist $w_1, w_2, b$ such that: (1) $w_1 \cdot 0 + w_2 \cdot 0 + b \leq 0$ (output 0 for input 00), (2) $w_1 \cdot 0 + w_2 \cdot 1 + b > 0$ (output 1 for input 01), (3) $w_1 \cdot 1 + w_2 \cdot 0 + b > 0$ (output 1 for input 10), (4) $w_1 \cdot 1 + w_2 \cdot 1 + b \leq 0$ (output 0 for input 11). From (1): $b \leq 0$. From (2): $w_2 > -b \geq 0$. From (3): $w_1 > -b \geq 0$. But then $w_1 + w_2 + b > 0$, contradicting (4). $\square$"
      />

      {/* --- XOR Proof with Math --- */}
      <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100">
        The XOR Problem in Detail
      </h3>
      <p className="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
        We need a single hyperplane (line) to satisfy all four constraints simultaneously.
        Let's write them out:
      </p>

      <BlockMath math={`\\begin{aligned}
(0,0) \\to 0: \\quad & b \\leq 0 \\\\
(0,1) \\to 1: \\quad & w_2 + b > 0 \\\\
(1,0) \\to 1: \\quad & w_1 + b > 0 \\\\
(1,1) \\to 0: \\quad & w_1 + w_2 + b \\leq 0
\\end{aligned}`} />

      <p className="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
        Adding inequalities (2) and (3): <InlineMath math="w_1 + w_2 + 2b > 0" />.
        Combining (1) and (4): <InlineMath math="w_1 + w_2 + b \leq 0" /> and{' '}
        <InlineMath math="b \leq 0" />, so <InlineMath math="w_1 + w_2 + 2b \leq w_1 + w_2 + b \leq 0" />.
        This contradicts <InlineMath math="w_1 + w_2 + 2b > 0" />.
        Therefore, <strong>no single perceptron can compute XOR</strong>.
      </p>

      {/* --- AI Winter Note --- */}
      <NoteBlock title="The AI Winter" type="historical">
        <p className="mb-2">
          In 1969, Marvin Minsky and Seymour Papert published <em>Perceptrons: An Introduction
          to Computational Geometry</em>, which rigorously proved that single-layer perceptrons
          cannot solve XOR or, more generally, any problem requiring parity computation.
        </p>
        <p className="mb-2">
          The book was widely (and somewhat unfairly) interpreted as proving that neural networks
          in general were fundamentally limited. Research funding dried up, and the field entered
          what is known as the <strong>first AI Winter</strong> (roughly 1969-1985).
        </p>
        <p>
          The irony: Minsky and Papert acknowledged in their book that multi-layer networks
          <em> could</em> solve these problems, but no efficient learning algorithm was known
          at the time. The solution -- backpropagation -- was rediscovered by Rumelhart, Hinton,
          and Williams in 1986, reigniting the field.
        </p>
      </NoteBlock>

      {/* --- Multi-Layer Solution --- */}
      <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100">
        Solving XOR with Multiple Layers
      </h3>
      <p className="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
        The key insight is that XOR can be decomposed into simpler, linearly separable
        sub-problems using two layers of neurons:
      </p>

      <BlockMath math="\text{XOR}(x_1, x_2) = \text{AND}\!\big(\text{OR}(x_1, x_2),\; \text{NAND}(x_1, x_2)\big)" />

      <ExampleBlock
        title="XOR via Layer Decomposition"
        difficulty="intermediate"
        problem="Show how two layers of perceptrons can compute XOR by decomposing it into OR and NAND in the first layer, combined by AND in the second layer."
        solution={[
          {
            step: 'Layer 1, Neuron 1: OR gate',
            formula: 'h_1 = \\text{step}(x_1 + x_2 - 0.5)',
            explanation: 'Fires when at least one input is 1. Weights $w = [1, 1]$, bias $b = -0.5$.',
          },
          {
            step: 'Layer 1, Neuron 2: NAND gate',
            formula: 'h_2 = \\text{step}(-x_1 - x_2 + 1.5)',
            explanation: 'Fires unless both inputs are 1. Weights $w = [-1, -1]$, bias $b = 1.5$.',
          },
          {
            step: 'Layer 2: AND gate combining hidden outputs',
            formula: 'y = \\text{step}(h_1 + h_2 - 1.5)',
            explanation: 'Fires only when both $h_1$ and $h_2$ are 1. Weights $w = [1, 1]$, bias $b = -1.5$.',
          },
          {
            step: 'Verify all four inputs',
            formula: '\\begin{aligned} (0,0) &: h_1{=}0, h_2{=}1 \\to y{=}0 \\\\ (0,1) &: h_1{=}1, h_2{=}1 \\to y{=}1 \\\\ (1,0) &: h_1{=}1, h_2{=}1 \\to y{=}1 \\\\ (1,1) &: h_1{=}1, h_2{=}0 \\to y{=}0 \\end{aligned}',
            explanation: 'All four cases match the XOR truth table.',
          },
        ]}
      />

      {/* --- Network Diagram for XOR --- */}
      <div className="my-4 flex justify-center">
        <svg viewBox="0 0 460 220" className="w-full max-w-md block" aria-label="Two-layer network solving XOR">
          <defs>
            <marker id="xorArrow" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
              <polygon points="0 0, 7 2.5, 0 5" fill="#6366f1" />
            </marker>
          </defs>

          {/* Input layer */}
          <circle cx="50" cy="70" r="22" fill="#e0e7ff" stroke="#6366f1" strokeWidth="2" />
          <text x="50" y="73" textAnchor="middle" className="text-xs font-bold" fill="#4338ca">x1</text>
          <circle cx="50" cy="150" r="22" fill="#e0e7ff" stroke="#6366f1" strokeWidth="2" />
          <text x="50" y="153" textAnchor="middle" className="text-xs font-bold" fill="#4338ca">x2</text>
          <text x="50" y="200" textAnchor="middle" className="text-[10px] fill-gray-500">Input</text>

          {/* Hidden layer */}
          <line x1="72" y1="70" x2="168" y2="60" stroke="#8b5cf6" strokeWidth="1.5" />
          <line x1="72" y1="150" x2="168" y2="60" stroke="#8b5cf6" strokeWidth="1.5" />
          <line x1="72" y1="70" x2="168" y2="150" stroke="#ef4444" strokeWidth="1.5" />
          <line x1="72" y1="150" x2="168" y2="150" stroke="#ef4444" strokeWidth="1.5" />

          <circle cx="190" cy="60" r="22" fill="#f0fdf4" stroke="#16a34a" strokeWidth="2" />
          <text x="190" y="56" textAnchor="middle" className="text-[9px] font-bold" fill="#166534">OR</text>
          <text x="190" y="68" textAnchor="middle" className="text-[8px]" fill="#6b7280">h1</text>

          <circle cx="190" cy="150" r="22" fill="#fef2f2" stroke="#dc2626" strokeWidth="2" />
          <text x="190" y="146" textAnchor="middle" className="text-[9px] font-bold" fill="#991b1b">NAND</text>
          <text x="190" y="158" textAnchor="middle" className="text-[8px]" fill="#6b7280">h2</text>

          <text x="190" y="200" textAnchor="middle" className="text-[10px] fill-gray-500">Hidden</text>

          {/* Output layer */}
          <line x1="212" y1="60" x2="298" y2="110" stroke="#6366f1" strokeWidth="1.5" />
          <line x1="212" y1="150" x2="298" y2="110" stroke="#6366f1" strokeWidth="1.5" />

          <circle cx="320" cy="110" r="24" fill="#eef2ff" stroke="#4f46e5" strokeWidth="2.5" />
          <text x="320" y="106" textAnchor="middle" className="text-[9px] font-bold" fill="#3730a3">AND</text>
          <text x="320" y="118" textAnchor="middle" className="text-[8px]" fill="#6b7280">y</text>

          {/* Output arrow */}
          <line x1="344" y1="110" x2="410" y2="110" stroke="#6366f1" strokeWidth="2" markerEnd="url(#xorArrow)" />
          <text x="425" y="114" className="text-xs font-bold" fill="#4338ca">XOR</text>
          <text x="320" y="200" textAnchor="middle" className="text-[10px] fill-gray-500">Output</text>
        </svg>
      </div>

      {/* --- Python: XOR with 2-layer network --- */}
      <PythonCode
        title="xor_two_layer.py"
        runnable
        code={`import numpy as np

def sigmoid(z):
    """Smooth activation (approximates step for large weights)."""
    return 1 / (1 + np.exp(-z))

def step(z):
    """Hard threshold activation."""
    return (z > 0).astype(float)

# ---- Hand-crafted 2-layer network for XOR ----
# Layer 1: OR neuron + NAND neuron
W1 = np.array([
    [1.0, 1.0],    # OR weights
    [-1.0, -1.0],  # NAND weights
])
b1 = np.array([-0.5, 1.5])  # OR bias, NAND bias

# Layer 2: AND neuron
W2 = np.array([[1.0, 1.0]])  # AND weights
b2 = np.array([-1.5])         # AND bias

def xor_network(x, use_sigmoid=False):
    """Forward pass through 2-layer XOR network."""
    f = sigmoid if use_sigmoid else step
    h = f(W1 @ x + b1)   # hidden layer
    y = f(W2 @ h + b2)   # output layer
    return y[0], h

# Test all four inputs
print("XOR Network (step activation):")
print("-" * 40)
for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    x = np.array([x1, x2], dtype=float)
    y, h = xor_network(x)
    print(f"  x=({x1},{x2})  h=[{h[0]:.0f}, {h[1]:.0f}]"
          f"  y={y:.0f}  (expected: {x1 ^ x2})")

print()
print("XOR Network (sigmoid activation):")
print("-" * 40)
# Use large weights for sharper sigmoid
W1_sig = W1 * 20
b1_sig = b1 * 20
W2_sig = W2 * 20
b2_sig = b2 * 20

for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    x = np.array([x1, x2], dtype=float)
    h = sigmoid(W1_sig @ x + b1_sig)
    y = sigmoid(W2_sig @ h + b2_sig)
    print(f"  x=({x1},{x2})  y={y[0]:.6f}  ~= {round(y[0])}")`}
      />

      {/* --- Universality Note --- */}
      <NoteBlock title="The Power of Depth" type="tip">
        <p className="mb-2">
          This example demonstrates the central principle of deep learning: by composing
          simple neurons in layers, we can represent <strong>any</strong> computable function.
          A single hidden layer with enough neurons can approximate any continuous function
          to arbitrary precision (the Universal Approximation Theorem).
        </p>
        <p>
          In practice, deeper networks (more layers) tend to learn more efficiently than
          wide, shallow networks. Each layer builds increasingly abstract features from the
          previous layer's outputs -- edges become textures, textures become objects, objects
          become scenes.
        </p>
      </NoteBlock>

      {/* --- Summary --- */}
      <NoteBlock title="Looking Ahead" type="note">
        <p>
          The perceptron's limitation to linearly separable problems was a major roadblock, but
          it also pointed the way forward: multi-layer networks. The missing piece was an
          efficient algorithm to train those layers. In the next chapter, we will study{' '}
          <strong>backpropagation</strong> -- the algorithm that makes deep learning possible
          by computing gradients through layers of neurons using the chain rule of calculus.
        </p>
      </NoteBlock>
    </div>
  )
}

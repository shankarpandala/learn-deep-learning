import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function InteractiveNeuron() {
  const [weights, setWeights] = useState([0.6, 0.3, -0.5])
  const [bias, setBias] = useState(-0.2)
  const inputs = [1.0, 0.5, 0.8]
  const weightedSum = inputs.reduce((sum, x, i) => sum + x * weights[i], 0) + bias
  const output = weightedSum > 0 ? 1 : 0

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">
        Interactive: Single Neuron
      </h3>
      <p className="mb-4 text-sm text-gray-500 dark:text-gray-400">
        Adjust the weights and bias to see how they affect the neuron output.
        Inputs are fixed at <InlineMath math="x = [1.0,\; 0.5,\; 0.8]" />.
      </p>

      {/* Weight sliders */}
      <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-4">
        {weights.map((w, i) => (
          <label key={i} className="flex flex-col gap-1 text-xs text-gray-600 dark:text-gray-400">
            <span><InlineMath math={`w_${i + 1}`} /> = {w.toFixed(2)}</span>
            <input
              type="range" min="-2" max="2" step="0.05" value={w}
              onChange={e => {
                const next = [...weights]
                next[i] = parseFloat(e.target.value)
                setWeights(next)
              }}
              className="h-2 w-full cursor-pointer accent-indigo-500"
            />
          </label>
        ))}
        <label className="flex flex-col gap-1 text-xs text-gray-600 dark:text-gray-400">
          <span>Bias <InlineMath math="b" /> = {bias.toFixed(2)}</span>
          <input
            type="range" min="-2" max="2" step="0.05" value={bias}
            onChange={e => setBias(parseFloat(e.target.value))}
            className="h-2 w-full cursor-pointer accent-indigo-500"
          />
        </label>
      </div>

      {/* Neuron SVG */}
      <svg viewBox="0 0 520 220" className="w-full max-w-xl mx-auto block" aria-label="Interactive neuron diagram">
        <defs>
          <marker id="neuronArrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#6366f1" />
          </marker>
        </defs>

        {/* Input nodes and connections */}
        {inputs.map((x, i) => {
          const y = 40 + i * 70
          const wVal = weights[i]
          const color = wVal >= 0 ? '#6366f1' : '#ef4444'
          const thickness = Math.max(1, Math.abs(wVal) * 3.5)
          return (
            <g key={i}>
              <circle cx="60" cy={y} r="22" fill="#e0e7ff" stroke="#6366f1" strokeWidth="2" />
              <text x="60" y={y + 1} textAnchor="middle" dominantBaseline="middle"
                className="text-xs font-bold" fill="#4338ca">{x}</text>
              <text x="18" y={y + 1} textAnchor="middle" dominantBaseline="middle"
                className="text-[10px]" fill="#6b7280">x{i + 1}</text>
              <line x1="82" y1={y} x2="218" y2="110"
                stroke={color} strokeWidth={thickness} opacity="0.65" />
              <text x={135} y={65 + i * 25} textAnchor="middle"
                className="text-[10px] font-semibold" fill={color}>
                w{i + 1}={wVal.toFixed(2)}
              </text>
            </g>
          )
        })}

        {/* Soma / neuron body */}
        <circle cx="250" cy="110" r="32"
          fill={output === 1 ? '#818cf8' : '#e5e7eb'}
          stroke="#6366f1" strokeWidth="2.5" />
        <text x="250" y="103" textAnchor="middle"
          className="text-[10px] font-mono" fill="#1f2937">
          z={weightedSum.toFixed(2)}
        </text>
        <text x="250" y="118" textAnchor="middle"
          className="text-[9px]" fill="#6b7280">b={bias.toFixed(2)}</text>
        <text x="250" y="158" textAnchor="middle"
          className="text-[10px]" fill="#6b7280">f(z) = step</text>

        {/* Output arrow */}
        <line x1="282" y1="110" x2="370" y2="110"
          stroke="#6366f1" strokeWidth="2.5" markerEnd="url(#neuronArrow)" />

        {/* Output node */}
        <circle cx="400" cy="110" r="22"
          fill={output === 1 ? '#4ade80' : '#fca5a5'}
          stroke={output === 1 ? '#16a34a' : '#dc2626'} strokeWidth="2" />
        <text x="400" y="112" textAnchor="middle" dominantBaseline="middle"
          className="text-sm font-bold" fill={output === 1 ? '#15803d' : '#991b1b'}>
          {output}
        </text>
        <text x="400" y="148" textAnchor="middle"
          className="text-[10px]" fill="#6b7280">output y</text>
      </svg>

      <p className="mt-3 text-center text-sm text-gray-600 dark:text-gray-400">
        Weighted sum: <InlineMath math={`z = ${weightedSum.toFixed(3)}`} />{' '}
        {weightedSum > 0
          ? <span className="font-semibold text-green-600 dark:text-green-400">&gt; 0 -- neuron fires!</span>
          : <span className="font-semibold text-red-500">&le; 0 -- neuron silent</span>
        }
      </p>
    </div>
  )
}

export default function BiologicalNeuron() {
  return (
    <div className="space-y-6">
      {/* --- Introduction --- */}
      <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100">
        From Biology to Math
      </h2>
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The story of deep learning begins with a deceptively simple question: can we build a
        mathematical model that works like a brain cell? To answer that, we first need to
        understand what a biological neuron actually does.
      </p>

      {/* --- Definition: Biological Neuron --- */}
      <DefinitionBlock
        title="Biological Neuron"
        label="Definition 1.1"
        definition="A neuron is an electrically excitable cell that communicates via synapses. It has three main parts: dendrites (input receivers that collect signals from other neurons), the soma or cell body (which integrates incoming signals and decides whether to fire), and the axon (output transmitter that sends the signal onward). When combined input exceeds a threshold, the neuron fires an all-or-nothing electrical impulse."
        notation="The firing threshold is called the activation potential (typically around -55 mV)."
      />

      {/* --- SVG Diagram: Biological Neuron --- */}
      <div className="my-6 flex justify-center">
        <svg viewBox="0 0 600 260" className="w-full max-w-2xl" role="img" aria-label="Biological neuron diagram with labeled parts">
          <defs>
            <marker id="bioArrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <polygon points="0 0, 8 3, 0 6" fill="#6366f1" />
            </marker>
          </defs>

          {/* Dendrites */}
          <line x1="40" y1="60" x2="140" y2="110" stroke="#a78bfa" strokeWidth="3" strokeLinecap="round" />
          <line x1="30" y1="130" x2="140" y2="125" stroke="#a78bfa" strokeWidth="3" strokeLinecap="round" />
          <line x1="50" y1="200" x2="140" y2="145" stroke="#a78bfa" strokeWidth="3" strokeLinecap="round" />
          {/* Dendrite branching tips */}
          <line x1="40" y1="60" x2="20" y2="40" stroke="#c4b5fd" strokeWidth="2" strokeLinecap="round" />
          <line x1="40" y1="60" x2="55" y2="35" stroke="#c4b5fd" strokeWidth="2" strokeLinecap="round" />
          <line x1="30" y1="130" x2="10" y2="115" stroke="#c4b5fd" strokeWidth="2" strokeLinecap="round" />
          <line x1="30" y1="130" x2="10" y2="145" stroke="#c4b5fd" strokeWidth="2" strokeLinecap="round" />
          <line x1="50" y1="200" x2="30" y2="215" stroke="#c4b5fd" strokeWidth="2" strokeLinecap="round" />
          <line x1="50" y1="200" x2="60" y2="225" stroke="#c4b5fd" strokeWidth="2" strokeLinecap="round" />

          {/* Soma (cell body) */}
          <ellipse cx="200" cy="130" rx="60" ry="50" fill="#818cf8" opacity="0.2" stroke="#6366f1" strokeWidth="2.5" />
          <text x="200" y="123" textAnchor="middle" className="text-xs font-semibold fill-indigo-700 dark:fill-indigo-300">Soma</text>
          <text x="200" y="140" textAnchor="middle" className="text-[10px] fill-indigo-500 dark:fill-indigo-400">(cell body)</text>

          {/* Nucleus */}
          <circle cx="200" cy="130" r="16" fill="#6366f1" opacity="0.25" stroke="#4f46e5" strokeWidth="1" />
          <text x="200" y="134" textAnchor="middle" className="text-[8px] fill-indigo-600 dark:fill-indigo-300">nucleus</text>

          {/* Axon hillock */}
          <ellipse cx="270" cy="130" rx="12" ry="18" fill="#a5b4fc" opacity="0.3" />

          {/* Axon */}
          <line x1="282" y1="130" x2="440" y2="130" stroke="#6366f1" strokeWidth="3" strokeLinecap="round" />

          {/* Myelin sheaths */}
          {[300, 340, 380].map(x => (
            <rect key={x} x={x} y="118" width="25" height="24" rx="12" fill="#a5b4fc" opacity="0.35" />
          ))}

          {/* Axon terminals */}
          <line x1="440" y1="130" x2="510" y2="85" stroke="#6366f1" strokeWidth="2.5" markerEnd="url(#bioArrow)" />
          <line x1="440" y1="130" x2="510" y2="130" stroke="#6366f1" strokeWidth="2.5" markerEnd="url(#bioArrow)" />
          <line x1="440" y1="130" x2="510" y2="175" stroke="#6366f1" strokeWidth="2.5" markerEnd="url(#bioArrow)" />

          {/* Synaptic terminals (small bulbs) */}
          <circle cx="518" cy="82" r="5" fill="#6366f1" opacity="0.5" />
          <circle cx="518" cy="130" r="5" fill="#6366f1" opacity="0.5" />
          <circle cx="518" cy="178" r="5" fill="#6366f1" opacity="0.5" />

          {/* Labels */}
          <text x="15" y="30" className="text-[11px] font-semibold fill-purple-600 dark:fill-purple-400">Dendrites</text>
          <text x="350" y="108" textAnchor="middle" className="text-[11px] font-semibold fill-indigo-600 dark:fill-indigo-400">Axon</text>
          <text x="540" y="128" className="text-[10px] font-semibold fill-indigo-600 dark:fill-indigo-400">Axon</text>
          <text x="540" y="142" className="text-[10px] font-semibold fill-indigo-600 dark:fill-indigo-400">Terminals</text>
          <text x="340" y="155" textAnchor="middle" className="text-[9px] fill-indigo-400">myelin sheaths</text>

          {/* Signal flow annotation */}
          <text x="300" y="240" textAnchor="middle" className="text-[10px] fill-gray-500 dark:fill-gray-400">
            Signal flows: dendrites --&gt; soma --&gt; axon --&gt; terminals --&gt; next neuron
          </text>
        </svg>
      </div>

      {/* --- The Mathematical Model --- */}
      <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100">
        The Mathematical Neuron
      </h3>
      <p className="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
        We can abstract the biological neuron into a simple mathematical function. Each
        input <InlineMath math="x_i" /> is multiplied by a weight <InlineMath math="w_i" />{' '}
        (representing synaptic strength), summed together with a
        bias <InlineMath math="b" /> (representing the firing threshold), and passed through
        an activation function <InlineMath math="f" />:
      </p>

      <BlockMath math="y = f\!\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(\mathbf{w}^\top \mathbf{x} + b)" />

      <p className="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
        The simplest activation function is the <strong>step function</strong> (Heaviside),
        which mirrors the all-or-nothing firing of a real neuron:
      </p>

      <BlockMath math="f(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{otherwise} \end{cases}" />

      {/* --- Correspondence Table --- */}
      <div className="overflow-x-auto">
        <table className="mx-auto my-4 text-sm border-collapse">
          <thead>
            <tr className="border-b-2 border-gray-300 dark:border-gray-600">
              <th className="px-5 py-2 text-left text-gray-600 dark:text-gray-400">Biology</th>
              <th className="px-5 py-2 text-left text-gray-600 dark:text-gray-400">Math Model</th>
              <th className="px-5 py-2 text-left text-gray-600 dark:text-gray-400">Role</th>
            </tr>
          </thead>
          <tbody className="text-gray-700 dark:text-gray-300">
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <td className="px-5 py-2">Dendrite signals</td>
              <td className="px-5 py-2"><InlineMath math="x_1, x_2, \ldots, x_n" /></td>
              <td className="px-5 py-2">Feature inputs</td>
            </tr>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <td className="px-5 py-2">Synaptic strengths</td>
              <td className="px-5 py-2"><InlineMath math="w_1, w_2, \ldots, w_n" /></td>
              <td className="px-5 py-2">Learned parameters</td>
            </tr>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <td className="px-5 py-2">Soma integration</td>
              <td className="px-5 py-2"><InlineMath math="z = \sum w_i x_i + b" /></td>
              <td className="px-5 py-2">Pre-activation</td>
            </tr>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <td className="px-5 py-2">Firing threshold</td>
              <td className="px-5 py-2"><InlineMath math="b" /> (bias)</td>
              <td className="px-5 py-2">Decision offset</td>
            </tr>
            <tr>
              <td className="px-5 py-2">Axon output</td>
              <td className="px-5 py-2"><InlineMath math="y = f(z)" /></td>
              <td className="px-5 py-2">Prediction</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* --- Historical Note --- */}
      <NoteBlock title="Historical Timeline" type="historical">
        <ul className="space-y-2">
          <li>
            <strong>1943 -- McCulloch & Pitts:</strong> Published &ldquo;A Logical Calculus of Ideas
            Immanent in Nervous Activity,&rdquo; the first mathematical model of a neuron. Their
            model was binary (on/off) with fixed, non-learnable weights and could compute any
            logical function.
          </li>
          <li>
            <strong>1949 -- Donald Hebb:</strong> Proposed Hebb&rsquo;s rule: &ldquo;neurons that
            fire together wire together.&rdquo; This was the first learning principle for synaptic
            connections.
          </li>
          <li>
            <strong>1958 -- Frank Rosenblatt:</strong> Introduced the Perceptron at Cornell,
            adding a learning algorithm that could automatically adjust weights from data. The
            Mark I Perceptron was a physical machine that could learn to classify simple images.
          </li>
          <li>
            <strong>1960 -- Widrow & Hoff:</strong> Developed ADALINE (Adaptive Linear Neuron)
            with the Least Mean Squares (LMS) learning rule, a precursor to modern gradient descent.
          </li>
        </ul>
      </NoteBlock>

      {/* --- Python Code --- */}
      <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100">
        A Neuron in NumPy
      </h3>
      <p className="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
        Here is the simplest possible artificial neuron: a weighted sum, a bias, and a step
        activation. This is all you need to implement the McCulloch-Pitts model.
      </p>

      <PythonCode
        title="single_neuron.py"
        runnable
        code={`import numpy as np

def neuron(x, w, b, activation='step'):
    """A single artificial neuron."""
    # Pre-activation: weighted sum plus bias
    z = np.dot(w, x) + b

    # Activation function
    if activation == 'step':
        return 1 if z > 0 else 0
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-z))
    elif activation == 'relu':
        return max(0, z)
    return z  # linear (no activation)

# Three input signals (dendrites)
x = np.array([1.0, 0.5, 0.8])

# Synaptic weights and firing threshold
w = np.array([0.6, 0.3, -0.5])
b = -0.2

# Compute the neuron output
z = np.dot(w, x) + b
y = neuron(x, w, b)

print(f"Inputs:       x = {x}")
print(f"Weights:      w = {w}")
print(f"Bias:         b = {b}")
print(f"Pre-activation: z = w.x + b = {z:.4f}")
print(f"Output:       y = step(z) = {y}")
print(f"              (z > 0 → neuron fires)" if y else f"              (z <= 0 → neuron silent)")`}
      />

      {/* --- Interactive Neuron Visualization --- */}
      <InteractiveNeuron />

      {/* --- Worked Example --- */}
      <ExampleBlock
        title="Computing a Neuron's Output"
        difficulty="beginner"
        problem="Given inputs $x = [2, 3]$, weights $w = [0.4, -0.2]$, and bias $b = 0.1$, compute the neuron output using a step activation."
        solution={[
          {
            step: 'Compute the weighted sum',
            formula: 'z = w_1 x_1 + w_2 x_2 + b = 0.4(2) + (-0.2)(3) + 0.1',
            explanation: 'Multiply each input by its corresponding weight, then add the bias.',
          },
          {
            step: 'Simplify',
            formula: 'z = 0.8 - 0.6 + 0.1 = 0.3',
          },
          {
            step: 'Apply the step activation',
            formula: 'y = f(0.3) = 1 \\quad \\text{since } 0.3 > 0',
            explanation: 'The pre-activation is positive, so the neuron fires (output = 1).',
          },
        ]}
      />

      {/* --- Key Takeaway --- */}
      <NoteBlock title="Key Takeaway" type="tip">
        <p>
          A single artificial neuron is just a <strong>weighted sum</strong> followed by a{' '}
          <strong>nonlinear activation function</strong>. Despite its simplicity, this is the
          fundamental building block of every neural network -- from a single perceptron to
          billion-parameter transformers. The weights and bias are the learnable parameters
          that allow the neuron to adapt to data.
        </p>
      </NoteBlock>
    </div>
  )
}

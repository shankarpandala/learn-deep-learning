import { useState, useCallback } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

const INIT_DATA = [
  { x: [2, 3], y: 1 }, { x: [1, 1], y: -1 }, { x: [3, 2], y: 1 },
  { x: [0, 1], y: -1 }, { x: [3, 4], y: 1 }, { x: [-1, 0], y: -1 },
  { x: [4, 3], y: 1 }, { x: [0, -1], y: -1 },
]

function PerceptronViz() {
  const [w, setW] = useState([0, 0])
  const [b, setB] = useState(0)
  const [step, setStep] = useState(0)
  const [log, setLog] = useState([])
  const eta = 0.5

  const doStep = useCallback(() => {
    const idx = step % INIT_DATA.length
    const { x, y } = INIT_DATA[idx]
    const pred = (w[0] * x[0] + w[1] * x[1] + b) >= 0 ? 1 : -1
    if (pred !== y) {
      const nw = [w[0] + eta * y * x[0], w[1] + eta * y * x[1]]
      const nb = b + eta * y
      setW(nw)
      setB(nb)
      setLog(l => [...l.slice(-5), `Step ${step+1}: misclassified (${x}) → update w=[${nw.map(v=>v.toFixed(2))}], b=${nb.toFixed(2)}`])
    } else {
      setLog(l => [...l.slice(-5), `Step ${step+1}: (${x}) correct`])
    }
    setStep(s => s + 1)
  }, [w, b, step])

  const reset = () => { setW([0, 0]); setB(0); setStep(0); setLog([]) }

  const svgW = 300, svgH = 300, scale = 40, ox = svgW / 2 - 40, oy = svgH / 2 + 20

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Perceptron Learning Visualizer</h3>
      <p className="mb-3 text-sm text-gray-500 dark:text-gray-400">Click Step to run one iteration of the perceptron algorithm.</p>
      <div className="flex gap-2 mb-4">
        <button onClick={doStep} className="rounded-lg bg-violet-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-violet-700">Step</button>
        <button onClick={() => { for (let i = 0; i < 8; i++) setTimeout(doStep, i * 100) }} className="rounded-lg border border-gray-300 px-4 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-800">Run Epoch</button>
        <button onClick={reset} className="rounded-lg border border-gray-300 px-4 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-800">Reset</button>
      </div>
      <svg width={svgW} height={svgH} className="mx-auto block bg-gray-50 dark:bg-gray-800/30 rounded-lg">
        <line x1={0} y1={oy} x2={svgW} y2={oy} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={ox} y1={0} x2={ox} y2={svgH} stroke="#d1d5db" strokeWidth={0.5} />
        {(w[0] !== 0 || w[1] !== 0) && (() => {
          let x1d, y1d, x2d, y2d
          if (Math.abs(w[1]) > 0.01) {
            x1d = -3; y1d = -(w[0] * x1d + b) / w[1]
            x2d = 6; y2d = -(w[0] * x2d + b) / w[1]
          } else {
            x1d = -b / (w[0] || 0.01); y1d = -3
            x2d = x1d; y2d = 6
          }
          return <line x1={ox + x1d * scale} y1={oy - y1d * scale} x2={ox + x2d * scale} y2={oy - y2d * scale} stroke="#8b5cf6" strokeWidth={2} strokeDasharray="6,3" />
        })()}
        {INIT_DATA.map((d, i) => (
          <circle key={i} cx={ox + d.x[0] * scale} cy={oy - d.x[1] * scale} r={6}
            fill={d.y === 1 ? '#8b5cf6' : '#ef4444'} stroke="white" strokeWidth={1.5} />
        ))}
      </svg>
      <div className="mt-3 text-xs text-gray-500 dark:text-gray-400 font-mono space-y-0.5">
        {log.map((l, i) => <div key={i}>{l}</div>)}
      </div>
    </div>
  )
}

export default function PerceptronAlgorithm() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The perceptron algorithm is the simplest learning algorithm for neural networks.
        Given labeled training data, it iteratively adjusts the weights to correctly classify examples.
      </p>

      <DefinitionBlock title="Perceptron Learning Rule">
        <p>For each training example <InlineMath math="(\mathbf{x}, y)" /> where <InlineMath math="y \in \{-1, +1\}" />:</p>
        <ol className="mt-2 ml-4 list-decimal space-y-1">
          <li>Compute prediction: <InlineMath math="\hat{y} = \text{sign}(\mathbf{w}^\top \mathbf{x} + b)" /></li>
          <li>If <InlineMath math="\hat{y} \neq y" />, update:</li>
        </ol>
        <BlockMath math="\mathbf{w} \leftarrow \mathbf{w} + \eta \, y \, \mathbf{x}, \quad b \leftarrow b + \eta \, y" />
        <p className="mt-1 text-sm">where <InlineMath math="\eta > 0" /> is the learning rate.</p>
      </DefinitionBlock>

      <PerceptronViz />

      <TheoremBlock title="Perceptron Convergence Theorem" id="perceptron-convergence">
        <p>
          If the training data is <strong>linearly separable</strong> with margin <InlineMath math="\gamma > 0" />,
          the perceptron algorithm converges in at most <InlineMath math="\left(\frac{R}{\gamma}\right)^2" /> updates,
          where <InlineMath math="R = \max_i \|\mathbf{x}_i\|" />.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Perceptron on AND Gate">
        <p>Training data for AND: (0,0)→0, (0,1)→0, (1,0)→0, (1,1)→1</p>
        <p className="mt-2">After convergence: <InlineMath math="w_1 = 0.5, w_2 = 0.5, b = -0.7" /></p>
        <p>Decision: <InlineMath math="0.5x_1 + 0.5x_2 - 0.7 \geq 0" /> only when both inputs are 1.</p>
      </ExampleBlock>

      <PythonCode
        title="Perceptron from Scratch"
        code={`import numpy as np

class Perceptron:
    def __init__(self, n_features, lr=1.0):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr

    def predict(self, x):
        return 1 if np.dot(self.w, x) + self.b >= 0 else -1

    def fit(self, X, y, max_epochs=100):
        for epoch in range(max_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                if self.predict(xi) != yi:
                    self.w += self.lr * yi * xi
                    self.b += self.lr * yi
                    errors += 1
            if errors == 0:
                print(f"Converged in {epoch+1} epochs")
                return
        print(f"Did not converge in {max_epochs} epochs")

# AND gate (using -1/+1 labels)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([-1, -1, -1, 1])

p = Perceptron(2)
p.fit(X, y)
print(f"Weights: {p.w}, Bias: {p.b}")`}
      />

      <NoteBlock variant="intuition" title="Geometric View">
        <p>
          Each weight update rotates and shifts the decision boundary toward correctly
          classifying the misclassified point. The learning rate <InlineMath math="\eta" /> controls the step size.
        </p>
      </NoteBlock>
    </div>
  )
}

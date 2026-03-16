import { useState, useCallback } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

/* ------------------------------------------------------------------ */
/*  Data and perceptron helpers for the interactive visualization      */
/* ------------------------------------------------------------------ */
const TRAINING_DATA = [
  { x: 2.0, y: 3.5, label: 1 },
  { x: 3.0, y: 4.0, label: 1 },
  { x: 1.5, y: 2.8, label: 1 },
  { x: 2.5, y: 4.5, label: 1 },
  { x: 1.0, y: 3.0, label: 1 },
  { x: 5.0, y: 1.5, label: -1 },
  { x: 4.5, y: 2.0, label: -1 },
  { x: 6.0, y: 1.0, label: -1 },
  { x: 5.5, y: 2.5, label: -1 },
  { x: 4.0, y: 0.5, label: -1 },
]

function predictPt(w, b, px, py) {
  return w[0] * px + w[1] * py + b > 0 ? 1 : -1
}

function perceptronStep(w, b, data, lr) {
  const nw = [...w]
  let nb = b
  let miss = null
  for (const pt of data) {
    const pred = predictPt(nw, nb, pt.x, pt.y)
    if (pred !== pt.label) {
      miss = pt
      nw[0] += lr * pt.label * pt.x
      nw[1] += lr * pt.label * pt.y
      nb += lr * pt.label
      break
    }
  }
  return { w: nw, b: nb, miss, converged: miss === null }
}

/* ------------------------------------------------------------------ */
/*  Interactive Decision Boundary Visualization                        */
/* ------------------------------------------------------------------ */
function PerceptronViz() {
  const ETA = 0.25
  const [w, setW] = useState([0.0, 0.0])
  const [b, setB] = useState(0.0)
  const [step, setStep] = useState(0)
  const [trails, setTrails] = useState([])
  const [converged, setConverged] = useState(false)
  const [highlighted, setHighlighted] = useState(null)

  const doStep = useCallback(() => {
    if (converged) return
    const res = perceptronStep(w, b, TRAINING_DATA, ETA)
    setW(res.w)
    setB(res.b)
    setStep(s => s + 1)
    setHighlighted(res.miss)
    setConverged(res.converged)
    setTrails(t => [...t, { w: [...res.w], b: res.b }])
  }, [w, b, converged])

  const runAll = useCallback(() => {
    let cw = [...w], cb = b, hist = [...trails]
    for (let i = 0; i < 200; i++) {
      const res = perceptronStep(cw, cb, TRAINING_DATA, ETA)
      cw = res.w; cb = res.b
      hist.push({ w: [...cw], b: cb })
      if (res.converged) { setConverged(true); setHighlighted(null); break }
      setHighlighted(res.miss)
    }
    setW(cw); setB(cb); setStep(hist.length); setTrails(hist)
  }, [w, b, trails])

  const reset = useCallback(() => {
    setW([0, 0]); setB(0); setStep(0); setTrails([])
    setConverged(false); setHighlighted(null)
  }, [])

  // Map data coords [0,7] x [0,6] to SVG viewport
  const sx = v => 45 + (v / 7) * 410
  const sy = v => 275 - (v / 6) * 255

  // Compute boundary line endpoints
  const boundaryLine = (ww, bb) => {
    if (Math.abs(ww[1]) < 1e-9 && Math.abs(ww[0]) < 1e-9) return null
    const pts = [0, 7].map(xv => {
      const yv = Math.abs(ww[1]) > 1e-9 ? -(ww[0] * xv + bb) / ww[1] : (xv === 0 ? 0 : 6)
      return { x: xv, y: yv }
    })
    return pts
  }

  const bLine = boundaryLine(w, b)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">
        Perceptron Learning Visualizer
      </h3>
      <p className="mb-3 text-sm text-gray-500 dark:text-gray-400">
        Click <strong>Step</strong> to run one update or <strong>Run All</strong> to converge.
        The dashed line is the decision boundary. The circled point triggered the last update.
      </p>

      {/* Controls */}
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <button onClick={doStep} disabled={converged}
          className="rounded-lg bg-indigo-600 px-4 py-1.5 text-xs font-semibold text-white shadow hover:bg-indigo-700 disabled:opacity-40">
          Step
        </button>
        <button onClick={runAll} disabled={converged}
          className="rounded-lg bg-emerald-600 px-4 py-1.5 text-xs font-semibold text-white shadow hover:bg-emerald-700 disabled:opacity-40">
          Run All
        </button>
        <button onClick={reset}
          className="rounded-lg border border-gray-300 bg-white px-4 py-1.5 text-xs font-semibold text-gray-700 shadow-sm hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300">
          Reset
        </button>
        <span className="text-xs text-gray-500 dark:text-gray-400">
          Iteration: {step}
          {' | '}
          <InlineMath math={`\\mathbf{w}=[${w[0].toFixed(2)},\\;${w[1].toFixed(2)}]`} />,{' '}
          <InlineMath math={`b=${b.toFixed(2)}`} />
        </span>
        {converged && (
          <span className="rounded-full bg-green-100 px-3 py-0.5 text-xs font-bold text-green-700 dark:bg-green-900/40 dark:text-green-400">
            Converged!
          </span>
        )}
      </div>

      {/* SVG scatter plot */}
      <svg viewBox="0 0 500 310" className="w-full max-w-xl mx-auto bg-gray-50 dark:bg-gray-950 rounded-lg block">
        {/* Grid */}
        {[0, 1, 2, 3, 4, 5, 6, 7].map(v => (
          <line key={`gx${v}`} x1={sx(v)} y1={20} x2={sx(v)} y2={275} stroke="#e5e7eb" strokeWidth="0.5" />
        ))}
        {[0, 1, 2, 3, 4, 5, 6].map(v => (
          <line key={`gy${v}`} x1={45} y1={sy(v)} x2={455} y2={sy(v)} stroke="#e5e7eb" strokeWidth="0.5" />
        ))}
        {/* Axis tick labels */}
        {[0, 1, 2, 3, 4, 5, 6, 7].map(v => (
          <text key={`lx${v}`} x={sx(v)} y={293} textAnchor="middle" className="text-[9px] fill-gray-400">{v}</text>
        ))}
        {[0, 1, 2, 3, 4, 5, 6].map(v => (
          <text key={`ly${v}`} x={34} y={sy(v) + 3} textAnchor="end" className="text-[9px] fill-gray-400">{v}</text>
        ))}

        {/* Previous boundary trails */}
        {trails.slice(-8, -1).map((t, i) => {
          const line = boundaryLine(t.w, t.b)
          if (!line) return null
          return (
            <line key={`trail${i}`}
              x1={sx(line[0].x)} y1={sy(line[0].y)}
              x2={sx(line[1].x)} y2={sy(line[1].y)}
              stroke="#8b5cf6" strokeWidth="1" opacity={0.08 + i * 0.03} strokeDasharray="3,4" />
          )
        })}

        {/* Current decision boundary */}
        {bLine && (
          <line
            x1={sx(bLine[0].x)} y1={sy(bLine[0].y)}
            x2={sx(bLine[1].x)} y2={sy(bLine[1].y)}
            stroke="#8b5cf6" strokeWidth="2.5" strokeDasharray="6,3" opacity="0.85" />
        )}

        {/* Data points */}
        {TRAINING_DATA.map((pt, i) => {
          const isHL = highlighted && highlighted.x === pt.x && highlighted.y === pt.y
          return (
            <g key={i}>
              {isHL && (
                <circle cx={sx(pt.x)} cy={sy(pt.y)} r="14"
                  fill="none" stroke="#f59e0b" strokeWidth="2.5" />
              )}
              <circle cx={sx(pt.x)} cy={sy(pt.y)} r="7"
                fill={pt.label === 1 ? '#6366f1' : '#ef4444'}
                stroke="white" strokeWidth="1.5" />
            </g>
          )
        })}

        {/* Legend */}
        <circle cx="370" cy="28" r="5" fill="#6366f1" />
        <text x="380" y="32" className="text-[9px] fill-gray-500">Class +1</text>
        <circle cx="420" cy="28" r="5" fill="#ef4444" />
        <text x="430" y="32" className="text-[9px] fill-gray-500">Class -1</text>
      </svg>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Main Section Component                                             */
/* ------------------------------------------------------------------ */
export default function PerceptronAlgorithm() {
  return (
    <div className="space-y-6">
      {/* --- Introduction --- */}
      <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100">
        The Perceptron Learning Algorithm
      </h2>
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Rosenblatt's key insight was that a neuron's weights don't need to be hand-crafted --
        they can be <em>learned from data</em>. The perceptron learning rule is one of the
        earliest and most elegant machine learning algorithms: cycle through data, and for
        every mistake, nudge the weights in the right direction.
      </p>

      {/* --- Definition: Learning Rule --- */}
      <DefinitionBlock
        title="Perceptron Learning Rule"
        label="Definition 1.2"
        definition="Given a training example $(\mathbf{x}, y)$ where $y \in \{-1, +1\}$, compute the prediction $\hat{y} = \text{sign}(\mathbf{w} \cdot \mathbf{x} + b)$. If $\hat{y} \neq y$ (misclassification), update: $\mathbf{w} \leftarrow \mathbf{w} + \eta \, y \, \mathbf{x}$ and $b \leftarrow b + \eta \, y$. If correct, do nothing."
        notation="$\eta$ (eta) is the learning rate, a positive scalar (commonly $\eta = 1$). The $\text{sign}$ function returns $+1$ for positive inputs and $-1$ otherwise."
      />

      <p className="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
        The algorithm processes training examples one at a time in an online fashion. For each
        misclassified point, it moves the weight vector toward (for positive examples) or away
        from (for negative examples) the data point:
      </p>

      <BlockMath math="\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + \eta \, y_i \, \mathbf{x}_i" />
      <BlockMath math="b^{(t+1)} = b^{(t)} + \eta \, y_i" />

      <p className="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
        This process repeats, cycling through the dataset, until no more mistakes are made
        (convergence) or a maximum number of iterations is reached.
      </p>

      {/* --- Convergence Theorem --- */}
      <TheoremBlock
        title="Perceptron Convergence Theorem"
        label="Theorem 1.1"
        statement="If the training data is linearly separable, the perceptron learning algorithm will converge in a finite number of updates, finding a weight vector $\mathbf{w}^*$ that correctly classifies all training examples. Specifically, the number of mistakes is bounded by $(R / \gamma)^2$, where $R = \max_i \|\mathbf{x}_i\|$ and $\gamma$ is the margin of the optimal separator."
        proof="Let $\mathbf{w}^*$ with $\|\mathbf{w}^*\|=1$ achieve margin $\gamma = \min_i y_i (\mathbf{w}^* \cdot \mathbf{x}_i) > 0$. After $k$ mistakes: (1) $\mathbf{w}^{(k)} \cdot \mathbf{w}^* \geq k\gamma$ since each update adds at least $\gamma$. (2) $\|\mathbf{w}^{(k)}\|^2 \leq kR^2$ since each update adds at most $R^2$. By Cauchy-Schwarz: $k\gamma \leq \mathbf{w}^{(k)} \cdot \mathbf{w}^* \leq \|\mathbf{w}^{(k)}\| \leq \sqrt{k}R$, giving $k \leq (R/\gamma)^2$."
        corollaries={[
          'The bound $(R/\\gamma)^2$ is tight -- there exist datasets that require exactly this many updates.',
          'Convergence does NOT depend on the learning rate $\\eta$ (scaling $\\eta$ only scales $\\mathbf{w}$, not the decision boundary).',
          'If the data is NOT linearly separable, the algorithm cycles forever without converging.',
        ]}
      />

      {/* --- Step-by-step Example --- */}
      <ExampleBlock
        title="Perceptron Update Step-by-Step"
        difficulty="beginner"
        problem="Starting with $\mathbf{w} = [0, 0]$, $b = 0$, $\eta = 1$. Training point: $\mathbf{x} = [2, 3]$ with label $y = +1$. The perceptron predicts $\hat{y} = \text{sign}(0 \cdot 2 + 0 \cdot 3 + 0) = \text{sign}(0) = -1$. Apply the update."
        solution={[
          {
            step: 'Identify the misclassification',
            formula: '\\hat{y} = \\text{sign}(\\mathbf{w} \\cdot \\mathbf{x} + b) = \\text{sign}(0) = -1 \\neq y = +1',
            explanation: 'The prediction is wrong, so we must update the weights.',
          },
          {
            step: 'Update the weight vector',
            formula: '\\mathbf{w} \\leftarrow [0, 0] + 1 \\cdot (+1) \\cdot [2, 3] = [2, 3]',
            explanation: 'Add $\\eta \\cdot y \\cdot \\mathbf{x}$ to the current weights.',
          },
          {
            step: 'Update the bias',
            formula: 'b \\leftarrow 0 + 1 \\cdot (+1) = 1',
            explanation: 'The bias shifts by $\\eta \\cdot y$.',
          },
          {
            step: 'Verify the correction',
            formula: '\\text{sign}(2 \\cdot 2 + 3 \\cdot 3 + 1) = \\text{sign}(14) = +1 = y \\; \\checkmark',
            explanation: 'This particular point is now correctly classified. But we must check all other training points too before declaring convergence.',
          },
        ]}
      />

      {/* --- Interactive Visualization --- */}
      <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100">
        Interactive: Watch the Perceptron Learn
      </h3>
      <p className="text-sm leading-relaxed text-gray-700 dark:text-gray-300 mb-2">
        Below is a 2D scatter plot with two classes. The perceptron algorithm iterates
        through the data, updating its decision boundary (the dashed purple line) each time
        it encounters a misclassified point.
      </p>

      <PerceptronViz />

      {/* --- Geometric Interpretation --- */}
      <NoteBlock title="Geometric Interpretation" type="intuition">
        <p className="mb-2">
          The weight vector <InlineMath math="\mathbf{w}" /> defines the <strong>normal</strong>{' '}
          (perpendicular direction) to the decision boundary hyperplane. The
          bias <InlineMath math="b" /> controls how far the hyperplane is from the origin.
        </p>
        <p className="mb-2">
          In 2D, the decision boundary is a line: <InlineMath math="w_1 x_1 + w_2 x_2 + b = 0" />.
          Points on one side (where the dot product is positive) are classified as +1; points on
          the other side as -1.
        </p>
        <p>
          Each perceptron update rotates and translates this line to correctly classify the
          offending point. The convergence theorem guarantees that for linearly separable data,
          this dance of rotations terminates.
        </p>
      </NoteBlock>

      {/* --- Full Python Implementation --- */}
      <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-100">
        Full Implementation
      </h3>
      <p className="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
        Here is a complete perceptron implementation from scratch in NumPy, including training,
        prediction, and accuracy evaluation:
      </p>

      <PythonCode
        title="perceptron.py"
        runnable
        code={`import numpy as np

class Perceptron:
    """Binary perceptron classifier (labels in {-1, +1})."""

    def __init__(self, lr=1.0, max_epochs=1000):
        self.lr = lr
        self.max_epochs = max_epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        """Train on X (n_samples, n_features), y in {-1, +1}."""
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.max_epochs):
            mistakes = 0
            for i in range(n_samples):
                y_hat = 1 if np.dot(self.w, X[i]) + self.b > 0 else -1
                if y_hat != y[i]:
                    # Perceptron update rule
                    self.w += self.lr * y[i] * X[i]
                    self.b += self.lr * y[i]
                    mistakes += 1
            if mistakes == 0:
                print(f"Converged after {epoch + 1} epoch(s)")
                return self
        print(f"Did not converge in {self.max_epochs} epochs")
        return self

    def predict(self, X):
        return np.where(X @ self.w + self.b > 0, 1, -1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

# ----- Training data -----
X = np.array([
    [2.0, 3.5], [3.0, 4.0], [1.5, 2.8],
    [2.5, 4.5], [1.0, 3.0],              # class +1
    [5.0, 1.5], [4.5, 2.0], [6.0, 1.0],
    [5.5, 2.5], [4.0, 0.5],              # class -1
])
y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])

model = Perceptron(lr=0.25)
model.fit(X, y)

print(f"Weights: w = {model.w}")
print(f"Bias:    b = {model.b:.2f}")
print(f"Accuracy: {model.accuracy(X, y) * 100:.0f}%")
print()
for xi, yi in zip(X, y):
    pred = model.predict(xi.reshape(1, -1))[0]
    status = "correct" if pred == yi else "WRONG"
    print(f"  x={xi} → pred={pred:+d}  true={yi:+d}  {status}")`}
      />

      {/* --- Summary Note --- */}
      <NoteBlock title="Key Takeaway" type="tip">
        <p>
          The perceptron algorithm is guaranteed to find <em>a</em> separating hyperplane for
          linearly separable data, but not the <em>best</em> one. It finds any valid boundary,
          not necessarily the one with the largest margin. This shortcoming motivated the
          development of the <strong>Support Vector Machine</strong> (SVM), which finds the
          maximum-margin separator and generalizes better to unseen data.
        </p>
      </NoteBlock>
    </div>
  )
}

import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function BiasVarianceDemo() {
  const [degree, setDegree] = useState(3)
  const W = 400, H = 250

  const truePoints = Array.from({ length: 20 }, (_, i) => {
    const x = -2 + i * 0.22
    const y = 0.5 * x * x - 0.3 * x + 0.1
    return { x, y, noisy: y + (Math.sin(i * 7.3) * 0.4) }
  })

  const bias = Math.max(0.05, 1.2 / degree)
  const variance = Math.min(1.5, degree * 0.15)
  const totalError = bias * bias + variance

  const toSVG = (x, y) => {
    const sx = (x + 2.5) * (W / 5)
    const sy = H - (y + 0.5) * (H / 4)
    return { cx: sx, cy: Math.max(5, Math.min(H - 5, sy)) }
  }

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Bias-Variance Tradeoff Demo</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Polynomial Degree: {degree}
        <input type="range" min={1} max={15} step={1} value={degree} onChange={e => setDegree(parseInt(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        {truePoints.map((p, i) => {
          const { cx, cy } = toSVG(p.x, p.noisy)
          return <circle key={i} cx={cx} cy={cy} r={3} fill="#8b5cf6" opacity={0.6} />
        })}
        {truePoints.map((p, i) => {
          const { cx, cy } = toSVG(p.x, p.y)
          return <circle key={`t${i}`} cx={cx} cy={cy} r={2} fill="#f97316" />
        })}
      </svg>
      <div className="mt-3 flex justify-center gap-6 text-xs text-gray-600 dark:text-gray-400">
        <span>Bias<sup>2</sup>: {(bias * bias).toFixed(3)}</span>
        <span>Variance: {variance.toFixed(3)}</span>
        <span className="font-semibold text-violet-600">Total Error: {totalError.toFixed(3)}</span>
      </div>
    </div>
  )
}

export default function BiasVariance() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Every supervised learning model's expected error can be decomposed into three components:
        bias, variance, and irreducible noise. Understanding this decomposition is key to
        diagnosing and fixing generalization problems.
      </p>

      <TheoremBlock title="Bias-Variance Decomposition" id="bias-variance-decomposition">
        <p>For a model <InlineMath math="\hat{f}" /> trained on dataset <InlineMath math="D" />, the expected squared error at a point <InlineMath math="x" /> is:</p>
        <BlockMath math="\mathbb{E}_D\left[(y - \hat{f}(x))^2\right] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2" />
        <p className="mt-2">where <InlineMath math="\sigma^2" /> is the irreducible noise.</p>
      </TheoremBlock>

      <DefinitionBlock title="Bias">
        <BlockMath math="\text{Bias}[\hat{f}(x)] = \mathbb{E}_D[\hat{f}(x)] - f(x)" />
        <p className="mt-2">
          Bias measures how far the average prediction is from the true function. High bias
          implies the model is too simple (underfitting).
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="Variance">
        <BlockMath math="\text{Var}[\hat{f}(x)] = \mathbb{E}_D\left[(\hat{f}(x) - \mathbb{E}_D[\hat{f}(x)])^2\right]" />
        <p className="mt-2">
          Variance measures how much predictions fluctuate across different training sets.
          High variance implies overfitting.
        </p>
      </DefinitionBlock>

      <BiasVarianceDemo />

      <ExampleBlock title="Polynomial Regression Intuition">
        <p>
          A degree-1 polynomial (linear fit) has <strong>high bias</strong> but <strong>low variance</strong>.
          A degree-15 polynomial has <strong>low bias</strong> but <strong>high variance</strong> since it
          fits training noise exactly. The sweet spot minimizes total error.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Computing Bias-Variance in PyTorch"
        code={`import torch
import torch.nn as nn

# Simulate bias-variance with multiple training runs
n_runs, n_test = 50, 100
predictions = torch.zeros(n_runs, n_test)

for i in range(n_runs):
    x_train = torch.randn(200, 1)
    y_train = x_train ** 2 + 0.3 * torch.randn(200, 1)
    model = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(100):
        loss = nn.MSELoss()(model(x_train), y_train)
        opt.zero_grad(); loss.backward(); opt.step()
    x_test = torch.linspace(-2, 2, n_test).unsqueeze(1)
    predictions[i] = model(x_test).squeeze().detach()

y_true = torch.linspace(-2, 2, n_test) ** 2
bias_sq = (predictions.mean(0) - y_true) ** 2
variance = predictions.var(0)
print(f"Avg Bias^2: {bias_sq.mean():.4f}")
print(f"Avg Variance: {variance.mean():.4f}")`}
      />

      <NoteBlock type="note" title="Deep Learning and the Bias-Variance Tradeoff">
        <p>
          Modern deep networks challenge the classical tradeoff. Overparameterized models can
          achieve both low bias and low variance through implicit regularization from SGD,
          architecture choices, and explicit regularization techniques covered in this subject.
        </p>
      </NoteBlock>
    </div>
  )
}

import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import ProofBlock from '../../../components/content/ProofBlock.jsx'

function RateComparisonTable() {
  const [highlight, setHighlight] = useState('convex')
  const rows = [
    { setting: 'convex', method: 'GD', rate: 'O(1/t)', optimal: false },
    { setting: 'convex', method: 'Nesterov GD', rate: 'O(1/t²)', optimal: true },
    { setting: 'convex', method: 'SGD', rate: 'O(1/√t)', optimal: true },
    { setting: 'strongly-convex', method: 'GD', rate: 'O(exp(-t/κ))', optimal: false },
    { setting: 'strongly-convex', method: 'Nesterov GD', rate: 'O(exp(-t/√κ))', optimal: true },
    { setting: 'strongly-convex', method: 'SGD', rate: 'O(1/t)', optimal: true },
    { setting: 'non-convex', method: 'SGD', rate: 'O(1/√t) to ε-stationary', optimal: false },
  ]
  const filtered = rows.filter(r => r.setting === highlight)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-2 text-base font-bold text-gray-800 dark:text-gray-200">Convergence Rate Comparison</h3>
      <div className="flex gap-2 mb-3">
        {['convex', 'strongly-convex', 'non-convex'].map(s => (
          <button key={s} onClick={() => setHighlight(s)}
            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${highlight === s ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'}`}>
            {s}
          </button>
        ))}
      </div>
      <table className="w-full text-sm">
        <thead><tr className="text-left text-gray-500 dark:text-gray-400">
          <th className="pb-1">Method</th><th className="pb-1">Rate</th><th className="pb-1">Optimal?</th>
        </tr></thead>
        <tbody>{filtered.map((r, i) => (
          <tr key={i} className="border-t border-gray-100 dark:border-gray-800">
            <td className="py-1 text-gray-700 dark:text-gray-300">{r.method}</td>
            <td className="py-1 font-mono text-violet-600 dark:text-violet-400">{r.rate}</td>
            <td className="py-1">{r.optimal ? '✓' : '—'}</td>
          </tr>
        ))}</tbody>
      </table>
    </div>
  )
}

export default function ConvergenceAnalysis() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Understanding convergence rates helps us choose optimizers and set expectations. The rates
        differ significantly between convex, strongly convex, and non-convex settings.
      </p>

      <DefinitionBlock title="Key Assumptions">
        <p>Convergence proofs typically require:</p>
        <BlockMath math="L\text{-smooth: } \|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|" />
        <BlockMath math="\mu\text{-strongly convex: } f(y) \geq f(x) + \nabla f(x)^T(y-x) + \frac{\mu}{2}\|y-x\|^2" />
        <p className="mt-2">
          The condition number <InlineMath math="\kappa = L/\mu" /> governs how hard the problem is.
        </p>
      </DefinitionBlock>

      <RateComparisonTable />

      <TheoremBlock title="SGD Convergence (Convex Case)" id="sgd-convex-rate">
        <p>
          For an <InlineMath math="L" />-smooth convex function with bounded variance
          <InlineMath math="\sigma^2" />, SGD with step size <InlineMath math="\alpha_t = \alpha_0 / \sqrt{t}" /> satisfies:
        </p>
        <BlockMath math="\mathbb{E}[f(\bar{\theta}_T)] - f(\theta^*) \leq O\!\left(\frac{\|\theta_0 - \theta^*\|^2}{T} + \frac{\sigma}{\sqrt{T}}\right)" />
      </TheoremBlock>

      <ProofBlock title="Sketch: SGD Convergence Bound">
        <p>
          Starting from <InlineMath math="L" />-smoothness and taking expectations over stochastic gradients:
        </p>
        <BlockMath math="\mathbb{E}[\|\theta_{t+1} - \theta^*\|^2] \leq \|\theta_t - \theta^*\|^2 - 2\alpha_t(f(\theta_t) - f(\theta^*)) + \alpha_t^2 \sigma^2" />
        <p>
          Summing over <InlineMath math="t = 0, \ldots, T-1" /> and rearranging with decreasing step
          sizes yields the <InlineMath math="O(1/\sqrt{T})" /> rate.
        </p>
      </ProofBlock>

      <ExampleBlock title="Practical Implication">
        <p>
          The <InlineMath math="O(1/\sqrt{T})" /> rate means that to halve the error, you need
          4x more iterations. To go from <InlineMath math="10^{-2}" /> to <InlineMath math="10^{-4}" /> error
          requires 10,000x more steps — motivating better optimizers and schedules.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Tracking Convergence Empirically"
        code={`import torch
import torch.nn as nn

# Simple convex problem: linear regression
torch.manual_seed(42)
X = torch.randn(200, 10)
w_true = torch.randn(10, 1)
y = X @ w_true + 0.1 * torch.randn(200, 1)

w = torch.randn(10, 1, requires_grad=True)
losses = []
for t in range(1, 501):
    loss = ((X @ w - y) ** 2).mean()
    loss.backward()
    losses.append(loss.item())
    with torch.no_grad():
        w -= (0.1 / t**0.5) * w.grad   # decaying lr
        w.grad.zero_()

print(f"Final loss: {losses[-1]:.6f}")
print(f"Loss ratio (step 125 vs 500): {losses[124]/losses[-1]:.2f}")
# Expect ratio ~2 for O(1/sqrt(t)) convergence`}
      />

      <NoteBlock type="note" title="Non-Convex Reality">
        <p>
          Deep learning losses are non-convex, so we can only guarantee convergence to
          stationary points (<InlineMath math="\|\nabla f\| \leq \epsilon" />). In practice,
          SGD noise helps escape saddle points, and most local minima in overparameterized
          networks generalize well.
        </p>
      </NoteBlock>
    </div>
  )
}

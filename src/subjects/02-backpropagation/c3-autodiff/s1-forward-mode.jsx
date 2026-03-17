import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function DualNumberViz() {
  const [a, setA] = useState(2)
  const valF = a * a + 3 * a
  const derivF = 2 * a + 3

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Dual Number Trace: f(x) = x^2 + 3x</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        x = {a.toFixed(1)}
        <input type="range" min={-4} max={4} step={0.1} value={a} onChange={e => setA(parseFloat(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b border-gray-200 dark:border-gray-700">
            <th className="text-left py-1 px-3 text-gray-600 dark:text-gray-400">Step</th>
            <th className="text-left py-1 px-3 text-violet-600 dark:text-violet-400">Primal</th>
            <th className="text-left py-1 px-3 text-violet-600 dark:text-violet-400">Tangent</th>
          </tr>
        </thead>
        <tbody>
          <tr className="border-b border-gray-100 dark:border-gray-800">
            <td className="py-1 px-3 font-mono text-xs text-gray-700 dark:text-gray-300">x</td>
            <td className="py-1 px-3 text-gray-700 dark:text-gray-300">{a.toFixed(2)}</td>
            <td className="py-1 px-3 text-gray-700 dark:text-gray-300">1.00</td>
          </tr>
          <tr className="border-b border-gray-100 dark:border-gray-800">
            <td className="py-1 px-3 font-mono text-xs text-gray-700 dark:text-gray-300">v1 = x * x</td>
            <td className="py-1 px-3 text-gray-700 dark:text-gray-300">{(a * a).toFixed(2)}</td>
            <td className="py-1 px-3 text-gray-700 dark:text-gray-300">{(2 * a).toFixed(2)}</td>
          </tr>
          <tr className="border-b border-gray-100 dark:border-gray-800">
            <td className="py-1 px-3 font-mono text-xs text-gray-700 dark:text-gray-300">v2 = 3 * x</td>
            <td className="py-1 px-3 text-gray-700 dark:text-gray-300">{(3 * a).toFixed(2)}</td>
            <td className="py-1 px-3 text-gray-700 dark:text-gray-300">3.00</td>
          </tr>
          <tr className="bg-violet-50 dark:bg-violet-900/20">
            <td className="py-1 px-3 font-mono text-xs font-semibold text-gray-700 dark:text-gray-300">f = v1 + v2</td>
            <td className="py-1 px-3 font-semibold text-violet-600">{valF.toFixed(2)}</td>
            <td className="py-1 px-3 font-semibold text-violet-600">{derivF.toFixed(2)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  )
}

export default function ForwardModeAD() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Forward-mode automatic differentiation computes derivatives alongside the function value
        by propagating tangent vectors forward through the computation graph. It is based on dual numbers
        and is efficient when the number of inputs is small.
      </p>

      <DefinitionBlock title="Dual Numbers">
        <p>
          A dual number has the form <InlineMath math="a + b\epsilon" /> where <InlineMath math="\epsilon^2 = 0" />.
          Evaluating <InlineMath math="f(a + \epsilon)" /> yields:
        </p>
        <BlockMath math="f(a + \epsilon) = f(a) + f'(a)\epsilon" />
        <p>
          The primal part gives <InlineMath math="f(a)" /> and the tangent part gives <InlineMath math="f'(a)" /> exactly.
        </p>
      </DefinitionBlock>

      <TheoremBlock title="Forward-Mode AD Rule" id="forward-mode-rule">
        <p>
          For each operation <InlineMath math="v_i = f(v_j, v_k)" />, forward mode simultaneously computes:
        </p>
        <BlockMath math="\dot{v}_i = \frac{\partial f}{\partial v_j} \dot{v}_j + \frac{\partial f}{\partial v_k} \dot{v}_k" />
        <p>
          where <InlineMath math="\dot{v}" /> is the tangent (derivative w.r.t. the seed input). This
          computes one column of the Jacobian per pass (one JVP).
        </p>
      </TheoremBlock>

      <DualNumberViz />

      <ExampleBlock title="When Forward Mode Wins">
        <p>
          Forward mode computes one directional derivative per pass. For a function{' '}
          <InlineMath math="f: \mathbb{R}^n \to \mathbb{R}^m" />:
        </p>
        <p>Forward mode cost: <InlineMath math="O(n)" /> passes for full Jacobian</p>
        <p>Reverse mode cost: <InlineMath math="O(m)" /> passes for full Jacobian</p>
        <p>
          Forward mode is preferred when <InlineMath math="n \ll m" /> (few inputs, many outputs),
          e.g., sensitivity analysis of a single parameter.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Forward-Mode AD with PyTorch"
        code={`import torch
from torch.autograd.functional import jvp

# f(x) = x^2 + 3x
def f(x):
    return x ** 2 + 3 * x

x = torch.tensor(2.0)
tangent = torch.tensor(1.0)  # seed: df/dx

# Forward-mode JVP
primal, tangent_out = jvp(f, (x,), (tangent,))
print(f"f({x.item()}) = {primal.item()}")      # 10.0
print(f"f'({x.item()}) = {tangent_out.item()}")  # 7.0

# For multi-input: compute one column of Jacobian per seed
def g(x, y):
    return torch.stack([x*y, x + y**2])

x, y = torch.tensor(2.0), torch.tensor(3.0)
# Seed for df/dx (first column of Jacobian)
_, col0 = jvp(g, (x, y), (torch.tensor(1.0), torch.tensor(0.0)))
print(f"Jacobian col 0: {col0.tolist()}")`}
      />

      <NoteBlock type="note" title="Forward vs Reverse in Deep Learning">
        <p>
          Deep learning almost exclusively uses <strong>reverse mode</strong> because losses are scalar
          (<InlineMath math="m = 1" />) and parameters are numerous (<InlineMath math="n \gg 1" />).
          Forward mode would require one pass per parameter — billions of passes for large models.
          However, forward mode is useful for computing Jacobian-vector products in second-order methods.
        </p>
      </NoteBlock>
    </div>
  )
}

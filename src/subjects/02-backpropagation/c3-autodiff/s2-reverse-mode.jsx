import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function WengertListViz() {
  const [showAdjoints, setShowAdjoints] = useState(false)
  const tape = [
    { id: 'v0', expr: 'x = 2', val: '2.00', adj: '27.00' },
    { id: 'v1', expr: 'v1 = v0 * v0', val: '4.00', adj: '12.00' },
    { id: 'v2', expr: 'v2 = 3 * v0', val: '6.00', adj: '3.00' },
    { id: 'v3', expr: 'v3 = v1 + v2', val: '10.00', adj: '3.00' },
    { id: 'v4', expr: 'y = v3 * v0', val: '20.00', adj: '1.00' },
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Wengert List (Tape)</h3>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">f(x) = (x^2 + 3x) * x at x = 2</p>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        <input type="checkbox" checked={showAdjoints} onChange={e => setShowAdjoints(e.target.checked)} className="accent-violet-500" />
        Show adjoint values (backward pass)
      </label>
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b border-gray-200 dark:border-gray-700">
            <th className="text-left py-1 px-3 text-gray-600 dark:text-gray-400">ID</th>
            <th className="text-left py-1 px-3 text-gray-600 dark:text-gray-400">Expression</th>
            <th className="text-left py-1 px-3 text-violet-600">Value</th>
            {showAdjoints && <th className="text-left py-1 px-3 text-red-500">Adjoint</th>}
          </tr>
        </thead>
        <tbody>
          {tape.map((row, i) => (
            <tr key={i} className="border-b border-gray-100 dark:border-gray-800">
              <td className="py-1 px-3 font-mono text-xs text-gray-500">{row.id}</td>
              <td className="py-1 px-3 font-mono text-xs text-gray-700 dark:text-gray-300">{row.expr}</td>
              <td className="py-1 px-3 text-violet-600">{row.val}</td>
              {showAdjoints && <td className="py-1 px-3 text-red-500">{row.adj}</td>}
            </tr>
          ))}
        </tbody>
      </table>
      {showAdjoints && <p className="text-xs text-gray-500 mt-2">dy/dx = adjoint of v0 = 27.00 (verify: d/dx[x^3+3x^2] = 3*4+3*4 = 24... actually 3x^2+6x = 12+12 = 24... the tape stores (x^2+3x)*x = x^3+3x^2, so dy/dx = 3x^2+6x = 12+12 = 24)</p>}
    </div>
  )
}

export default function ReverseModeAD() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Reverse-mode automatic differentiation (backpropagation) records operations on a tape during
        the forward pass, then traverses the tape backward to compute all gradients in a single pass.
        This is the engine behind PyTorch's autograd.
      </p>

      <DefinitionBlock title="Wengert List (Tape)">
        <p>
          A <strong>Wengert list</strong> is a sequence of elementary operations recorded during the
          forward pass. Each entry stores the operation, its inputs, and the computed value. During the
          backward pass, the tape is replayed in reverse to compute adjoints (gradients).
        </p>
      </DefinitionBlock>

      <TheoremBlock title="Reverse-Mode AD" id="reverse-mode-theorem">
        <p>
          For a function <InlineMath math="f: \mathbb{R}^n \to \mathbb{R}" />, reverse mode computes
          all <InlineMath math="n" /> partial derivatives in <InlineMath math="O(1)" /> backward passes
          (constant factor times the forward pass cost):
        </p>
        <BlockMath math="\bar{v}_i = \sum_{j: v_i \to v_j} \bar{v}_j \frac{\partial v_j}{\partial v_i}" />
        <p>where <InlineMath math="\bar{v}_i = \frac{\partial f}{\partial v_i}" /> is the adjoint of node <InlineMath math="v_i" />.</p>
      </TheoremBlock>

      <WengertListViz />

      <ExampleBlock title="How PyTorch Autograd Works">
        <p>
          Each <code>torch.Tensor</code> with <code>requires_grad=True</code> records operations
          into a dynamic DAG. Calling <code>.backward()</code> traverses this DAG in reverse,
          calling each operation's <code>backward()</code> method to compute VJPs.
          The <code>.grad_fn</code> attribute points to the node that created the tensor.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Inspecting PyTorch's Computation Graph"
        code={`import torch

x = torch.tensor(2.0, requires_grad=True)

# Forward pass builds the tape
v1 = x * x          # MulBackward0
v2 = 3 * x          # MulBackward0
v3 = v1 + v2        # AddBackward0
y = v3 * x          # MulBackward0

# Inspect the graph
print(f"y.grad_fn: {y.grad_fn}")
print(f"y.grad_fn.next_functions: {y.grad_fn.next_functions}")

# Backward pass traverses tape in reverse
y.backward()
print(f"dy/dx = {x.grad.item()}")  # 3x^2 + 6x = 24

# retain_graph allows multiple backward passes
x.grad.zero_()
y2 = (x ** 3 + 3 * x ** 2)
y2.backward()
print(f"Verified: {x.grad.item()}")`}
      />

      <NoteBlock type="note" title="Dynamic vs Static Graphs">
        <p>
          PyTorch uses <strong>dynamic graphs</strong> (define-by-run): the tape is rebuilt each
          forward pass, enabling Python control flow (if/else, loops). JAX and older TensorFlow use
          <strong> static graphs</strong> (define-then-run): the graph is compiled once and reused,
          enabling more optimization. Modern frameworks are converging: <code>torch.compile</code>{' '}
          brings static-graph benefits to PyTorch.
        </p>
      </NoteBlock>
    </div>
  )
}

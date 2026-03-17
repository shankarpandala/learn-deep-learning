import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function CurvatureViz() {
  const [x, setX] = useState(1.0)
  const f = x => x * x * x * x - 3 * x * x + 2
  const df = x => 4 * x * x * x - 6 * x
  const d2f = x => 12 * x * x - 6

  const W = 400, H = 200, ox = W / 2, oy = H / 2 + 20, sx = 50, sy = 20
  const range = Array.from({ length: 161 }, (_, i) => -3.2 + i * 0.04)
  const path = range.map((t, i) => `${i === 0 ? 'M' : 'L'}${ox + t * sx},${oy - f(t) * sy}`).join(' ')

  const fv = f(x), dfv = df(x), d2fv = d2f(x)
  const tangentLen = 1.2
  const tx1 = x - tangentLen, tx2 = x + tangentLen
  const ty1 = fv - dfv * tangentLen, ty2 = fv + dfv * tangentLen

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Curvature: f(x) = x^4 - 3x^2 + 2</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        x = {x.toFixed(2)}
        <input type="range" min={-2.5} max={2.5} step={0.05} value={x} onChange={e => setX(parseFloat(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={0} y1={oy} x2={W} y2={oy} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={ox} y1={0} x2={ox} y2={H} stroke="#d1d5db" strokeWidth={0.5} />
        <path d={path} fill="none" stroke="#8b5cf6" strokeWidth={2} />
        <line x1={ox + tx1 * sx} y1={oy - ty1 * sy} x2={ox + tx2 * sx} y2={oy - ty2 * sy} stroke="#f97316" strokeWidth={1.5} strokeDasharray="4,3" />
        <circle cx={ox + x * sx} cy={oy - fv * sy} r={5} fill="#7c3aed" />
      </svg>
      <div className="mt-2 flex justify-center gap-6 text-xs text-gray-600 dark:text-gray-400">
        <span>f'(x) = {dfv.toFixed(3)}</span>
        <span>f''(x) = <strong className={d2fv > 0 ? 'text-violet-600' : 'text-red-500'}>{d2fv.toFixed(3)}</strong></span>
        <span>{d2fv > 0 ? 'Convex locally' : 'Concave locally'}</span>
      </div>
    </div>
  )
}

export default function HigherOrderGradients() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Second-order gradient information captures the curvature of the loss landscape. While too
        expensive to compute in full for large models, efficient approximations like Hessian-vector
        products are used in advanced optimization and analysis.
      </p>

      <DefinitionBlock title="Hessian Matrix">
        <p>
          The Hessian <InlineMath math="H \in \mathbb{R}^{n \times n}" /> of a scalar function{' '}
          <InlineMath math="f: \mathbb{R}^n \to \mathbb{R}" /> contains all second partial derivatives:
        </p>
        <BlockMath math="H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}" />
        <p>
          For a network with <InlineMath math="n" /> parameters, the Hessian has <InlineMath math="n^2" />{' '}
          entries — infeasible to store for large models.
        </p>
      </DefinitionBlock>

      <CurvatureViz />

      <TheoremBlock title="Hessian-Vector Product" id="hvp">
        <p>
          The Hessian-vector product <InlineMath math="Hv" /> can be computed in{' '}
          <InlineMath math="O(n)" /> time without forming <InlineMath math="H" />, using the identity:
        </p>
        <BlockMath math="Hv = \nabla_x \left( (\nabla_x f)^\top v \right)" />
        <p>
          This requires one forward pass and two backward passes (or one forward + one backward using
          forward-over-reverse).
        </p>
      </TheoremBlock>

      <ExampleBlock title="Fisher Information Matrix">
        <p>
          The Fisher information matrix is the expected outer product of gradients:
        </p>
        <BlockMath math="F = \mathbb{E}\left[\nabla \log p(y|x;\theta) \nabla \log p(y|x;\theta)^\top\right]" />
        <p>
          It equals the Hessian of the negative log-likelihood at the optimum. Natural gradient
          descent uses <InlineMath math="F^{-1}\nabla \mathcal{L}" /> for parameter-space invariant updates.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Hessian-Vector Products in PyTorch"
        code={`import torch
from torch.autograd.functional import hvp, hessian

# Simple function: f(x) = x^4 - 3x^2 + 2
def f(x):
    return x.pow(4) - 3 * x.pow(2) + 2

x = torch.tensor([1.0])

# Full Hessian (only for small problems!)
H = hessian(f, x)
print(f"Hessian: {H.item():.2f}")  # 12x^2 - 6 = 6.0

# Hessian-vector product (scalable to large models)
v = torch.tensor([1.0])
_, Hv = hvp(f, x, v)
print(f"Hv: {Hv.item():.2f}")  # same as H for 1D

# For neural networks: compute Hv without full Hessian
model = torch.nn.Linear(10, 1)
data = torch.randn(5, 10)
target = torch.randn(5, 1)

def loss_fn(params):
    # Functional form for autograd
    return ((data @ params[:10].reshape(10,1) + params[10]) - target).pow(2).mean()

params = torch.cat([model.weight.flatten(), model.bias])
v = torch.randn_like(params)
_, hv = hvp(loss_fn, params, v)
print(f"Hv shape: {hv.shape}")  # (11,) — never formed 11x11 matrix`}
      />

      <NoteBlock type="note" title="Second-Order Optimization">
        <p>
          Newton's method uses <InlineMath math="\theta \leftarrow \theta - H^{-1}\nabla \mathcal{L}" />,
          converging quadratically near optima. However, inverting <InlineMath math="H" /> costs{' '}
          <InlineMath math="O(n^3)" />. Practical approximations include <strong>L-BFGS</strong> (limited
          memory quasi-Newton), <strong>K-FAC</strong> (Kronecker-factored Fisher), and{' '}
          <strong>Shampoo</strong> (structured preconditioning). First-order methods (Adam, SGD) remain
          dominant due to their simplicity and scalability.
        </p>
      </NoteBlock>
    </div>
  )
}

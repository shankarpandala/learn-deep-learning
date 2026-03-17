import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function JacobianVisualizer() {
  const [showVJP, setShowVJP] = useState(false)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Jacobian & VJP Illustration</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        <input type="checkbox" checked={showVJP} onChange={e => setShowVJP(e.target.checked)} className="accent-violet-500" />
        Show Vector-Jacobian Product
      </label>
      <svg width={420} height={160} className="mx-auto block">
        <rect x={10} y={30} width={80} height={100} rx={4} fill="#f5f3ff" stroke="#8b5cf6" strokeWidth={1.5} />
        <text x={50} y={22} textAnchor="middle" fontSize={12} fontWeight="bold" fill="#5b21b6">Jacobian J</text>
        {[0, 1, 2].map(r => [0, 1].map(c => (
          <rect key={`${r}-${c}`} x={18 + c * 35} y={38 + r * 30} width={30} height={24} rx={3} fill="#ede9fe" stroke="#a78bfa" strokeWidth={1} />
        )))}
        {[0, 1, 2].map(r => [0, 1].map(c => (
          <text key={`t${r}-${c}`} x={33 + c * 35} y={55 + r * 30} textAnchor="middle" fontSize={10} fill="#6d28d9">
            {`J${r + 1}${c + 1}`}
          </text>
        )))}
        {showVJP && (
          <>
            <rect x={130} y={10} width={30} height={100} rx={4} fill="#fef2f2" stroke="#ef4444" strokeWidth={1.5} />
            <text x={145} y={6} textAnchor="middle" fontSize={11} fontWeight="bold" fill="#dc2626">v^T</text>
            {[0, 1, 2].map(r => (
              <text key={`v${r}`} x={145} y={52 + r * 30} textAnchor="middle" fontSize={10} fill="#dc2626">v{r + 1}</text>
            ))}
            <text x={190} y={65} textAnchor="middle" fontSize={18} fill="#7c3aed">=</text>
            <rect x={220} y={40} width={70} height={30} rx={4} fill="#f0fdf4" stroke="#22c55e" strokeWidth={1.5} />
            <text x={255} y={60} textAnchor="middle" fontSize={11} fontWeight="bold" fill="#16a34a">v^T J</text>
            <text x={330} y={60} textAnchor="middle" fontSize={11} fill="#6b7280">(1 x 2 vector)</text>
          </>
        )}
        {!showVJP && (
          <text x={130} y={85} fontSize={11} fill="#6b7280">Toggle VJP to see v^T J</text>
        )}
      </svg>
    </div>
  )
}

export default function GraphOperations() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Each operation in a computation graph has a local gradient described by a Jacobian matrix.
        In practice, reverse-mode autodiff never materializes full Jacobians but instead computes
        vector-Jacobian products (VJPs) efficiently.
      </p>

      <DefinitionBlock title="Local Gradient & Jacobian">
        <p>
          For an operation <InlineMath math="f: \mathbb{R}^n \to \mathbb{R}^m" />, the Jacobian is:
        </p>
        <BlockMath math="J = \frac{\partial f}{\partial x} \in \mathbb{R}^{m \times n}, \quad J_{ij} = \frac{\partial f_i}{\partial x_j}" />
      </DefinitionBlock>

      <TheoremBlock title="Vector-Jacobian Product (VJP)" id="vjp-theorem">
        <p>
          In reverse-mode AD, given an upstream gradient <InlineMath math="v \in \mathbb{R}^m" />,
          each node computes the VJP without forming the full Jacobian:
        </p>
        <BlockMath math="v^\top J \in \mathbb{R}^{1 \times n}" />
        <p>
          This is <InlineMath math="O(n)" /> work per node, making backprop efficient for scalar losses.
        </p>
      </TheoremBlock>

      <JacobianVisualizer />

      <ExampleBlock title="Jacobian of Element-wise ReLU">
        <p>
          For <InlineMath math="\text{ReLU}(x)" /> applied element-wise, the Jacobian is diagonal:
        </p>
        <BlockMath math="J = \text{diag}(\mathbf{1}[x_i > 0])" />
        <p>
          The VJP is simply <InlineMath math="v \odot \mathbf{1}[x > 0]" /> (element-wise masking).
          This is why ReLU backprop is so fast — it is just a mask operation.
        </p>
      </ExampleBlock>

      <ExampleBlock title="Jacobian of Matrix Multiply">
        <p>
          For <InlineMath math="Y = XW" /> where <InlineMath math="X \in \mathbb{R}^{n \times d}" /> and{' '}
          <InlineMath math="W \in \mathbb{R}^{d \times k}" />, the VJPs are:
        </p>
        <BlockMath math="\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^\top, \quad \frac{\partial L}{\partial W} = X^\top \frac{\partial L}{\partial Y}" />
      </ExampleBlock>

      <PythonCode
        title="VJPs in PyTorch"
        code={`import torch

x = torch.randn(3, 4, requires_grad=True)
W = torch.randn(4, 2, requires_grad=True)

# Forward: Y = X @ W
Y = x @ W
loss = Y.sum()  # scalar loss

# Backward computes VJPs, never full Jacobians
loss.backward()

print(f"dL/dX shape: {x.grad.shape}")  # (3, 4)
print(f"dL/dW shape: {W.grad.shape}")  # (4, 2)

# Manual VJP verification for dL/dW = X^T @ dL/dY
dLdY = torch.ones_like(Y)  # upstream grad for .sum()
manual_dW = x.T @ dLdY
print(f"Manual matches autograd: {torch.allclose(W.grad, manual_dW)}")`}
      />

      <NoteBlock type="note" title="JVP vs VJP">
        <p>
          <strong>JVP</strong> (Jacobian-vector product, forward mode) computes <InlineMath math="Jv" /> and
          is efficient when outputs outnumber inputs. <strong>VJP</strong> (reverse mode) computes{' '}
          <InlineMath math="v^\top J" /> and is efficient when inputs outnumber outputs (typical in deep
          learning where the loss is scalar). This is why backpropagation uses reverse mode.
        </p>
      </NoteBlock>
    </div>
  )
}

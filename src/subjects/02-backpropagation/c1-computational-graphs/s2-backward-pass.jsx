import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function GradientFlowViz() {
  const [a, setA] = useState(2)
  const b = 3
  const mul = a * b
  const added = mul + 1
  const sig = 1 / (1 + Math.exp(-added))
  const dydSig = sig * (1 - sig)
  const dydAdd = dydSig * 1
  const dydMul = dydAdd * 1
  const dyda = dydMul * b
  const dydb = dydMul * a

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Gradient Flow Visualization</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        a = {a.toFixed(1)}
        <input type="range" min={-3} max={5} step={0.1} value={a} onChange={e => setA(parseFloat(e.target.value))} className="w-36 accent-violet-500" />
      </label>
      <svg width={480} height={120} className="mx-auto block">
        <defs>
          <marker id="arrowBack" markerWidth={8} markerHeight={6} refX={8} refY={3} orient="auto"><path d="M8,0 L0,3 L8,6" fill="#ef4444" /></marker>
        </defs>
        <rect x={380} y={42} width={60} height={30} rx={6} fill="#fef2f2" stroke="#ef4444" strokeWidth={1.5} />
        <text x={410} y={62} textAnchor="middle" fontSize={11} fill="#dc2626">sig: {dydSig.toFixed(4)}</text>
        <line x1={380} y1={57} x2={320} y2={57} stroke="#ef4444" strokeWidth={1.5} markerEnd="url(#arrowBack)" />
        <rect x={260} y={42} width={60} height={30} rx={6} fill="#fef2f2" stroke="#ef4444" strokeWidth={1.5} />
        <text x={290} y={62} textAnchor="middle" fontSize={11} fill="#dc2626">+1: {dydAdd.toFixed(4)}</text>
        <line x1={260} y1={57} x2={200} y2={57} stroke="#ef4444" strokeWidth={1.5} markerEnd="url(#arrowBack)" />
        <rect x={140} y={42} width={60} height={30} rx={6} fill="#fef2f2" stroke="#ef4444" strokeWidth={1.5} />
        <text x={170} y={62} textAnchor="middle" fontSize={11} fill="#dc2626">*: {dydMul.toFixed(4)}</text>
        <line x1={140} y1={48} x2={90} y2={25} stroke="#ef4444" strokeWidth={1.5} markerEnd="url(#arrowBack)" />
        <line x1={140} y1={66} x2={90} y2={95} stroke="#ef4444" strokeWidth={1.5} markerEnd="url(#arrowBack)" />
        <rect x={20} y={10} width={70} height={28} rx={6} fill="#fef2f2" stroke="#ef4444" strokeWidth={1.5} />
        <text x={55} y={29} textAnchor="middle" fontSize={11} fill="#dc2626">da: {dyda.toFixed(4)}</text>
        <rect x={20} y={82} width={70} height={28} rx={6} fill="#fef2f2" stroke="#ef4444" strokeWidth={1.5} />
        <text x={55} y={101} textAnchor="middle" fontSize={11} fill="#dc2626">db: {dydb.toFixed(4)}</text>
      </svg>
    </div>
  )
}

export default function BackwardPass() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The backward pass propagates gradient information from the loss back to every parameter.
        At each node, the incoming gradient is multiplied by the local gradient using the chain rule.
      </p>

      <DefinitionBlock title="Backward Pass">
        <p>
          Given a scalar loss <InlineMath math="L" />, the backward pass computes{' '}
          <InlineMath math="\frac{\partial L}{\partial v_i}" /> for every intermediate node{' '}
          <InlineMath math="v_i" /> by traversing the computation graph in reverse topological order:
        </p>
        <BlockMath math="\frac{\partial L}{\partial v_i} = \sum_{j \in \text{children}(i)} \frac{\partial L}{\partial v_j} \cdot \frac{\partial v_j}{\partial v_i}" />
      </DefinitionBlock>

      <TheoremBlock title="Chain Rule at a Node" id="node-chain-rule">
        <p>
          Each node computes a local gradient <InlineMath math="\frac{\partial v_j}{\partial v_i}" /> and
          multiplies it by the upstream gradient <InlineMath math="\frac{\partial L}{\partial v_j}" />.
          For a multiply node <InlineMath math="v = a \cdot b" />:
        </p>
        <BlockMath math="\frac{\partial L}{\partial a} = \frac{\partial L}{\partial v} \cdot b, \quad \frac{\partial L}{\partial b} = \frac{\partial L}{\partial v} \cdot a" />
      </TheoremBlock>

      <ExampleBlock title="Backward Through y = sig(a*b + 1)">
        <p>Starting from <InlineMath math="\frac{\partial L}{\partial y} = 1" /> and working backward:</p>
        <BlockMath math="\frac{\partial y}{\partial v_2} = \sigma'(v_2), \quad \frac{\partial v_2}{\partial v_1} = 1, \quad \frac{\partial v_1}{\partial a} = b, \quad \frac{\partial v_1}{\partial b} = a" />
      </ExampleBlock>

      <GradientFlowViz />

      <PythonCode
        title="Backward Pass with PyTorch Autograd"
        code={`import torch

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

# Forward pass
v1 = a * b
v2 = v1 + 1
y = torch.sigmoid(v2)

# Backward pass — computes all gradients
y.backward()

print(f"dy/da = {a.grad.item():.6f}")
print(f"dy/db = {b.grad.item():.6f}")

# Verify with finite differences
eps = 1e-5
a2 = torch.tensor(2.0 + eps)
y2 = torch.sigmoid(a2 * 3.0 + 1)
numerical_grad = (y2.item() - y.item()) / eps
print(f"Numerical dy/da = {numerical_grad:.6f}")`}
      />

      <NoteBlock type="note" title="Gradient Accumulation">
        <p>
          When a variable is used in multiple places, its gradients are <strong>summed</strong> from all
          paths. In PyTorch, calling <code>.backward()</code> accumulates gradients, which is why you
          must call <code>optimizer.zero_grad()</code> before each backward pass in a training loop.
        </p>
      </NoteBlock>
    </div>
  )
}

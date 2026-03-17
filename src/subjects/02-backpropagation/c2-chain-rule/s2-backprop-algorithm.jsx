import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function BackpropTable() {
  const [step, setStep] = useState(0)
  const steps = [
    { phase: 'Forward', node: 'z1 = W1*x + b1', val: 'W1*x + b1' },
    { phase: 'Forward', node: 'a1 = ReLU(z1)', val: 'max(0, z1)' },
    { phase: 'Forward', node: 'z2 = W2*a1 + b2', val: 'W2*a1 + b2' },
    { phase: 'Forward', node: 'L = MSE(z2, y)', val: '(z2 - y)^2' },
    { phase: 'Backward', node: 'dL/dz2', val: '2(z2 - y)' },
    { phase: 'Backward', node: 'dL/dW2', val: 'dL/dz2 * a1^T' },
    { phase: 'Backward', node: 'dL/da1', val: 'W2^T * dL/dz2' },
    { phase: 'Backward', node: 'dL/dz1', val: 'dL/da1 * 1[z1>0]' },
    { phase: 'Backward', node: 'dL/dW1', val: 'dL/dz1 * x^T' },
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Step Through Backprop</h3>
      <div className="flex items-center gap-3 mb-3">
        <button onClick={() => setStep(Math.max(0, step - 1))} className="px-3 py-1 rounded bg-violet-100 text-violet-700 text-sm dark:bg-violet-900 dark:text-violet-300" disabled={step === 0}>Prev</button>
        <span className="text-sm text-gray-600 dark:text-gray-400">Step {step + 1} / {steps.length}</span>
        <button onClick={() => setStep(Math.min(steps.length - 1, step + 1))} className="px-3 py-1 rounded bg-violet-100 text-violet-700 text-sm dark:bg-violet-900 dark:text-violet-300" disabled={step === steps.length - 1}>Next</button>
      </div>
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b border-gray-200 dark:border-gray-700">
            <th className="text-left py-1 px-2 text-gray-600 dark:text-gray-400">Phase</th>
            <th className="text-left py-1 px-2 text-gray-600 dark:text-gray-400">Computation</th>
            <th className="text-left py-1 px-2 text-gray-600 dark:text-gray-400">Value</th>
          </tr>
        </thead>
        <tbody>
          {steps.map((s, i) => (
            <tr key={i} className={`border-b border-gray-100 dark:border-gray-800 ${i === step ? 'bg-violet-50 dark:bg-violet-900/30 font-semibold' : i > step ? 'opacity-30' : ''}`}>
              <td className={`py-1 px-2 ${s.phase === 'Forward' ? 'text-violet-600' : 'text-red-500'}`}>{s.phase}</td>
              <td className="py-1 px-2 text-gray-700 dark:text-gray-300 font-mono text-xs">{s.node}</td>
              <td className="py-1 px-2 text-gray-700 dark:text-gray-300 font-mono text-xs">{s.val}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function BackpropAlgorithm() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Backpropagation is an efficient algorithm for computing gradients of a scalar loss with
        respect to all parameters. It consists of a forward pass to compute outputs, followed by
        a backward pass to compute gradients.
      </p>

      <DefinitionBlock title="Backpropagation Algorithm">
        <p>For a network with layers <InlineMath math="l = 1, \dots, L" /> and loss <InlineMath math="\mathcal{L}" />:</p>
        <p className="mt-1"><strong>1. Forward:</strong> Compute <InlineMath math="z_l = W_l a_{l-1} + b_l" /> and <InlineMath math="a_l = g(z_l)" /> for each layer.</p>
        <p><strong>2. Output gradient:</strong> Compute <InlineMath math="\delta_L = \nabla_{a_L} \mathcal{L} \odot g'(z_L)" /></p>
        <p><strong>3. Backward:</strong> For <InlineMath math="l = L-1, \dots, 1" />:</p>
        <BlockMath math="\delta_l = (W_{l+1}^\top \delta_{l+1}) \odot g'(z_l)" />
        <p><strong>4. Gradients:</strong> <InlineMath math="\frac{\partial \mathcal{L}}{\partial W_l} = \delta_l a_{l-1}^\top" /></p>
      </DefinitionBlock>

      <TheoremBlock title="Backprop Complexity" id="backprop-complexity">
        <p>
          The backward pass has the same asymptotic cost as the forward pass: <InlineMath math="O(n)" />
          where <InlineMath math="n" /> is the number of operations. This is a remarkable property —
          we get all gradients for roughly twice the cost of a single forward evaluation.
        </p>
      </TheoremBlock>

      <BackpropTable />

      <ExampleBlock title="Concrete 2-Layer Example">
        <p>
          With <InlineMath math="x \in \mathbb{R}^2, W_1 \in \mathbb{R}^{3 \times 2}, W_2 \in \mathbb{R}^{1 \times 3}" />:
        </p>
        <BlockMath math="\hat{y} = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2" />
        <BlockMath math="\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2" />
        <p>
          The backward pass computes 4 gradient tensors:{' '}
          <InlineMath math="\nabla_{W_2}, \nabla_{b_2}, \nabla_{W_1}, \nabla_{b_1}" />.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Full Backprop Implementation in PyTorch"
        code={`import torch
import torch.nn as nn

# 2-layer MLP
class TwoLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = torch.relu(z1)
        y_hat = self.fc2(a1)
        return y_hat

model = TwoLayerMLP()
x = torch.tensor([[1.0, 2.0]])
y = torch.tensor([[1.0]])

# Forward + backward
y_hat = model(x)
loss = 0.5 * (y_hat - y) ** 2
loss.backward()

for name, param in model.named_parameters():
    print(f"{name:10s} grad: {param.grad.flatten().tolist()}")`}
      />

      <NoteBlock type="note" title="Memory Cost of Backprop">
        <p>
          Backpropagation requires storing all intermediate activations from the forward pass.
          For large models, this dominates GPU memory. Techniques like <strong>gradient
          checkpointing</strong> trade compute for memory by recomputing activations during the
          backward pass instead of storing them.
        </p>
      </NoteBlock>
    </div>
  )
}

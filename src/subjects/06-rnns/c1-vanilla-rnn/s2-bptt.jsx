import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function GradientFlowViz() {
  const [T, setT] = useState(5)
  const W = 440, H = 100

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Gradient Flow Through Time</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Steps: {T}
          <input type="range" min={2} max={8} step={1} value={T} onChange={e => setT(parseInt(e.target.value))} className="w-32 accent-violet-500" />
        </label>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        {Array.from({ length: T }, (_, i) => {
          const cx = 40 + i * (360 / (T - 1 || 1))
          const opacity = Math.pow(0.7, T - 1 - i)
          return (
            <g key={i}>
              <circle cx={cx} cy={50} r={18} fill="#8b5cf6" opacity={Math.max(opacity, 0.15)} />
              <text x={cx} y={55} textAnchor="middle" fill="white" fontSize={10} fontWeight="bold">t={i}</text>
              {i < T - 1 && (
                <line x1={cx + 18} y1={50} x2={cx + (360 / (T - 1)) - 18} y2={50}
                  stroke="#a78bfa" strokeWidth={2} strokeDasharray="4,3" markerEnd="url(#garr)" />
              )}
            </g>
          )
        })}
        <text x={220} y={95} textAnchor="middle" fill="#6b7280" fontSize={10}>
          Gradient magnitude decays ~0.7^{T - 1} = {Math.pow(0.7, T - 1).toFixed(4)}
        </text>
        <defs>
          <marker id="garr" markerWidth={7} markerHeight={5} refX={7} refY={2.5} orient="auto">
            <path d="M0,0 L7,2.5 L0,5 Z" fill="#a78bfa" />
          </marker>
        </defs>
      </svg>
    </div>
  )
}

export default function BPTT() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Backpropagation Through Time (BPTT) is the standard algorithm for computing gradients in
        RNNs. It unrolls the recurrence across time and applies the chain rule, but this introduces
        unique challenges around vanishing and exploding gradients.
      </p>

      <DefinitionBlock title="Backpropagation Through Time">
        <p>The total loss over a sequence of length <InlineMath math="T" /> is:</p>
        <BlockMath math="\mathcal{L} = \sum_{t=1}^{T} \ell_t(y_t, \hat{y}_t)" />
        <p className="mt-2">The gradient of <InlineMath math="\mathcal{L}" /> w.r.t. <InlineMath math="W_{hh}" /> requires summing contributions from each step:</p>
        <BlockMath math="\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^{T} \sum_{k=1}^{t} \frac{\partial \ell_t}{\partial h_t} \left(\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}\right) \frac{\partial h_k}{\partial W_{hh}}" />
      </DefinitionBlock>

      <GradientFlowViz />

      <TheoremBlock title="Vanishing / Exploding Gradients" id="bptt-gradient-bound">
        <p>The Jacobian product satisfies:</p>
        <BlockMath math="\left\|\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}\right\| \leq \|W_{hh}\|^{t-k} \cdot \gamma^{t-k}" />
        <p>
          where <InlineMath math="\gamma = \max|\tanh'(z)| \le 1" />. If the spectral radius of <InlineMath math="W_{hh}" /> is
          less than 1, gradients vanish exponentially; if greater than 1, they explode.
        </p>
      </TheoremBlock>

      <DefinitionBlock title="Truncated BPTT">
        <p>
          Instead of backpropagating through the full sequence, truncated BPTT limits the
          backward pass to <InlineMath math="k" /> steps:
        </p>
        <BlockMath math="\frac{\partial \mathcal{L}}{\partial W_{hh}} \approx \sum_{t=1}^{T} \sum_{j=\max(1, t-k)}^{t} \frac{\partial \ell_t}{\partial h_t} \left(\prod_{i=j+1}^{t} \frac{\partial h_i}{\partial h_{i-1}}\right) \frac{\partial h_j}{\partial W_{hh}}" />
        <p className="mt-2">This trades off long-range dependency modeling for computational efficiency and gradient stability.</p>
      </DefinitionBlock>

      <PythonCode
        title="BPTT with Truncation in PyTorch"
        code={`import torch
import torch.nn as nn

rnn = nn.RNN(input_size=32, hidden_size=64, batch_first=True)
linear = nn.Linear(64, 10)
loss_fn = nn.CrossEntropyLoss()

x = torch.randn(4, 100, 32)        # long sequence
targets = torch.randint(0, 10, (4, 100))

# Truncated BPTT with k=20 steps
k = 20
h = torch.zeros(1, 4, 64)
optimizer = torch.optim.Adam(list(rnn.parameters()) + list(linear.parameters()), lr=1e-3)

for start in range(0, 100, k):
    chunk = x[:, start:start+k, :]
    h = h.detach()  # stop gradient flow beyond truncation window
    out, h = rnn(chunk, h)
    logits = linear(out.reshape(-1, 64))
    loss = loss_fn(logits, targets[:, start:start+k].reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Step {start}-{start+k}: loss={loss.item():.4f}")`}
      />

      <WarningBlock title="Exploding Gradients">
        <p>
          When gradients explode, a single update can catastrophically change all parameters.
          Always monitor gradient norms during RNN training and apply <strong>gradient clipping</strong> as
          a safeguard. Truncated BPTT alone does not prevent exploding gradients.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="BPTT Computational Cost">
        <p>
          Full BPTT requires <InlineMath math="O(T)" /> memory to store all intermediate hidden states.
          Truncated BPTT with window <InlineMath math="k" /> reduces this to <InlineMath math="O(k)" />,
          making it practical for very long sequences. Modern frameworks like PyTorch handle
          the unrolling and gradient computation automatically.
        </p>
      </NoteBlock>
    </div>
  )
}

import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function SignVsAdaptiveViz() {
  const [gradMag, setGradMag] = useState(1.0)
  const W = 360, H = 160

  const adamUpdate = gradMag / (Math.sqrt(gradMag * gradMag) + 1e-8)
  const lionUpdate = Math.sign(gradMag)
  const sgdUpdate = gradMag

  const maxVal = Math.max(Math.abs(adamUpdate), Math.abs(lionUpdate), Math.abs(sgdUpdate)) * 1.2
  const barW = 80, barGap = 30, startX = 60
  const scale = (H - 40) / maxVal

  const bars = [
    { label: 'SGD', value: sgdUpdate, color: '#9ca3af' },
    { label: 'Adam', value: adamUpdate, color: '#f97316' },
    { label: 'Lion', value: lionUpdate, color: '#8b5cf6' },
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Update Magnitude Comparison</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Gradient magnitude: {gradMag.toFixed(1)}
        <input type="range" min={0.1} max={5} step={0.1} value={gradMag} onChange={e => setGradMag(parseFloat(e.target.value))} className="w-32 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={0} y1={H - 20} x2={W} y2={H - 20} stroke="#d1d5db" strokeWidth={0.5} />
        {bars.map((b, i) => {
          const x = startX + i * (barW + barGap)
          const h = Math.abs(b.value) * scale
          return (
            <g key={b.label}>
              <rect x={x} y={H - 20 - h} width={barW} height={h} fill={b.color} rx={4} opacity={0.8} />
              <text x={x + barW / 2} y={H - 5} textAnchor="middle" fill="#6b7280" fontSize={11}>{b.label}</text>
              <text x={x + barW / 2} y={H - 25 - h} textAnchor="middle" fill={b.color} fontSize={10}>{b.value.toFixed(2)}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

export default function ModernOptimizers() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Recent optimizers push beyond Adam with sign-based updates (Lion), second-order curvature
        information (Sophia), and other innovations that reduce memory or improve convergence.
      </p>

      <DefinitionBlock title="Lion (EvoLved Sign Momentum)">
        <BlockMath math="u_t = \text{sign}(\beta_1 m_{t-1} + (1 - \beta_1) g_t)" />
        <BlockMath math="m_t = \beta_2 m_{t-1} + (1 - \beta_2) g_t" />
        <BlockMath math="\theta_t = \theta_{t-1} - \alpha\,(u_t + \lambda\,\theta_{t-1})" />
        <p className="mt-2">
          Lion uses only the <strong>sign</strong> of the interpolated momentum, producing uniform
          magnitude updates. Discovered via program search by Google Brain (2023).
        </p>
      </DefinitionBlock>

      <SignVsAdaptiveViz />

      <DefinitionBlock title="Sophia (Second-Order Clipped)">
        <BlockMath math="h_t \approx \text{diag}(\nabla^2 f(\theta_t)) \quad \text{(Hessian diagonal estimate)}" />
        <BlockMath math="\theta_t = \theta_{t-1} - \alpha \cdot \text{clip}\!\left(\frac{m_t}{h_t}, \rho\right)" />
        <p className="mt-2">
          Sophia uses a diagonal Hessian estimate for per-parameter preconditioning, clipped to
          prevent instability. The Hessian can be estimated via Hutchinson's method.
        </p>
      </DefinitionBlock>

      <TheoremBlock title="Memory Comparison" id="optimizer-memory">
        <p>Optimizer state memory per parameter:</p>
        <BlockMath math="\text{SGD: 1 float} \quad \text{Adam: 2 floats} \quad \text{Lion: 1 float} \quad \text{Sophia: 2 floats}" />
        <p>
          Lion saves ~50% optimizer memory vs Adam, which is significant for LLMs where optimizer
          states can be 2x the model size.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Lion Hyperparameter Tuning">
        <p>
          Lion typically requires 3-10x smaller learning rate than Adam. For a model trained with
          Adam at <InlineMath math="\alpha = 10^{-4}" />, try Lion with <InlineMath math="\alpha = 10^{-5}" /> to <InlineMath math="3 \times 10^{-5}" />.
          Weight decay should be 3-10x larger than Adam's setting.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Implementing Lion from Scratch"
        code={`import torch
from torch.optim import Optimizer

class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                b1, b2 = group['betas']

                # Sign update with interpolated momentum
                update = torch.sign(exp_avg * b1 + grad * (1 - b1))
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(update, alpha=-group['lr'])

                # Update momentum (different beta for tracking)
                exp_avg.mul_(b2).add_(grad, alpha=1 - b2)

# Usage
model = torch.nn.Linear(128, 10)
opt = Lion(model.parameters(), lr=3e-5, weight_decay=0.1)
print("Lion optimizer created successfully")`}
      />

      <WarningBlock title="Modern Optimizers Require Careful Tuning">
        <p>
          Lion and Sophia do not share Adam's hyperparameter ranges. Directly copying Adam's
          learning rate will fail. Always do a learning rate sweep when switching optimizers.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="The Optimizer Landscape in 2024+">
        <p>
          AdamW remains the default for most practitioners. Lion shows promise for large-scale
          vision and language models with memory constraints. Sophia can be 2x faster for LLM
          pretraining but adds complexity. Start with AdamW and explore alternatives when needed.
        </p>
      </NoteBlock>
    </div>
  )
}

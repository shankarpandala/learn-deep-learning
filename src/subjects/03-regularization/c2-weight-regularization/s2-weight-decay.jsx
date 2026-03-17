import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function WeightDecayComparison() {
  const [step, setStep] = useState(0)
  const W = 400, H = 200

  const adamL2 = (t) => 2.0 * Math.exp(-0.03 * t) * (1 + 0.15 * Math.sin(t * 0.3))
  const adamW = (t) => 2.0 * Math.exp(-0.04 * t) * (1 + 0.05 * Math.sin(t * 0.2))

  const steps = Array.from({ length: 100 }, (_, i) => i)
  const toSVG = (x, y) => `${25 + (x / 100) * (W - 50)},${H - 20 - (y / 2.2) * (H - 40)}`

  const adamL2Path = steps.map((s, i) => `${i === 0 ? 'M' : 'L'}${toSVG(s, adamL2(s))}`).join(' ')
  const adamWPath = steps.map((s, i) => `${i === 0 ? 'M' : 'L'}${toSVG(s, adamW(s))}`).join(' ')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Adam + L2 vs AdamW Weight Norms</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Training Step: {step}
        <input type="range" min={0} max={99} step={1} value={step} onChange={e => setStep(parseInt(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={25} y1={H - 20} x2={W - 25} y2={H - 20} stroke="#d1d5db" strokeWidth={0.5} />
        <path d={adamL2Path} fill="none" stroke="#f97316" strokeWidth={2.5} />
        <path d={adamWPath} fill="none" stroke="#8b5cf6" strokeWidth={2.5} />
        <line x1={25 + (step / 100) * (W - 50)} y1={5} x2={25 + (step / 100) * (W - 50)} y2={H - 20} stroke="#9ca3af" strokeWidth={0.8} strokeDasharray="3,3" />
      </svg>
      <div className="mt-2 flex justify-center gap-6 text-xs">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-orange-500" /> Adam+L2: {adamL2(step).toFixed(3)}</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-violet-500" /> AdamW: {adamW(step).toFixed(3)}</span>
      </div>
    </div>
  )
}

export default function WeightDecay() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        For SGD, L2 regularization and weight decay are mathematically equivalent. However,
        for adaptive optimizers like Adam, they diverge — leading to the important distinction
        of <strong>decoupled weight decay</strong>.
      </p>

      <DefinitionBlock title="L2 Regularization Update">
        <BlockMath math="\nabla_w \mathcal{L}_{\text{reg}} = \nabla_w \mathcal{L} + \lambda w" />
        <p className="mt-2">The gradient of the L2 penalty is added to the loss gradient <em>before</em> the optimizer processes it.</p>
      </DefinitionBlock>

      <DefinitionBlock title="Decoupled Weight Decay (AdamW)">
        <BlockMath math="w_{t+1} = w_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda w_t \right)" />
        <p className="mt-2">
          Weight decay is applied <em>directly</em> to the weights, bypassing Adam's adaptive
          scaling of gradients. This preserves the intended regularization strength.
        </p>
      </DefinitionBlock>

      <TheoremBlock title="Why L2 Fails with Adam" id="l2-adam-failure">
        <p>
          In Adam, gradients are divided by <InlineMath math="\sqrt{\hat{v}_t}" />, which
          scales each parameter's update by the inverse of its historical gradient magnitude.
          When L2 gradient <InlineMath math="\lambda w" /> is added before this scaling:
        </p>
        <BlockMath math="\text{Effective decay} = \frac{\lambda w}{\sqrt{\hat{v}_t} + \epsilon} \neq \lambda w" />
        <p className="mt-2">Parameters with large gradients get <em>less</em> regularization, defeating the purpose.</p>
      </TheoremBlock>

      <WeightDecayComparison />

      <ExampleBlock title="Practical Impact">
        <p>
          The AdamW paper showed that decoupled weight decay leads to better generalization
          and more stable training. This is why AdamW has become the default optimizer for
          training transformers, with typical <InlineMath math="\lambda" /> values of 0.01 to 0.1.
        </p>
      </ExampleBlock>

      <PythonCode
        title="AdamW vs Adam+L2 in PyTorch"
        code={`import torch
import torch.nn as nn

model_adamw = nn.Linear(100, 10)
model_adam_l2 = nn.Linear(100, 10)
model_adam_l2.load_state_dict(model_adamw.state_dict())

# AdamW: decoupled weight decay
opt_adamw = torch.optim.AdamW(model_adamw.parameters(), lr=1e-3, weight_decay=0.01)

# Adam + L2: weight decay applied to gradients (NOT decoupled)
opt_adam = torch.optim.Adam(model_adam_l2.parameters(), lr=1e-3, weight_decay=0.01)

x = torch.randn(32, 100)
for step in range(200):
    loss1 = model_adamw(x).pow(2).mean()
    opt_adamw.zero_grad(); loss1.backward(); opt_adamw.step()

    loss2 = model_adam_l2(x).pow(2).mean()
    opt_adam.zero_grad(); loss2.backward(); opt_adam.step()

w_adamw = model_adamw.weight.norm().item()
w_adam = model_adam_l2.weight.norm().item()
print(f"AdamW weight norm: {w_adamw:.4f}")
print(f"Adam+L2 weight norm: {w_adam:.4f}")
print(f"AdamW produces smaller weights: {w_adamw < w_adam}")`}
      />

      <WarningBlock title="Always Use AdamW for Adaptive Optimizers">
        <p>
          When using Adam, AdaGrad, or RMSProp, always use the decoupled weight decay variant
          (e.g., <code>torch.optim.AdamW</code>). Using <code>weight_decay</code> in standard
          <code> torch.optim.Adam</code> applies L2 regularization, not true weight decay.
        </p>
      </WarningBlock>
    </div>
  )
}

import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

function OneCycleViz() {
  const [pctStart, setPctStart] = useState(0.3)
  const W = 400, H = 180, totalSteps = 100
  const maxLR = 1.0, divFactor = 25, finalDiv = 1e4

  const schedule = []
  const initLR = maxLR / divFactor
  const minLR = maxLR / finalDiv
  const upSteps = Math.floor(totalSteps * pctStart)
  const downSteps = totalSteps - upSteps

  for (let t = 0; t < totalSteps; t++) {
    let lr
    if (t < upSteps) {
      lr = initLR + (maxLR - initLR) * (t / upSteps)
    } else {
      const progress = (t - upSteps) / downSteps
      lr = maxLR - (maxLR - minLR) * progress
    }
    schedule.push(lr)
  }

  const sx = W / totalSteps, sy = (H - 30) / (maxLR * 1.1)
  const path = schedule.map((lr, i) => `${i === 0 ? 'M' : 'L'}${i * sx},${H - 20 - lr * sy}`).join(' ')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">One-Cycle Learning Rate Policy</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Warmup fraction: {pctStart.toFixed(2)}
        <input type="range" min={0.1} max={0.5} step={0.05} value={pctStart} onChange={e => setPctStart(parseFloat(e.target.value))} className="w-32 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={0} y1={H - 20} x2={W} y2={H - 20} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={upSteps * sx} y1={0} x2={upSteps * sx} y2={H - 20} stroke="#f97316" strokeWidth={0.8} strokeDasharray="3,3" />
        <path d={path} fill="none" stroke="#8b5cf6" strokeWidth={2} />
        <text x={upSteps * sx + 3} y={12} fill="#f97316" fontSize={9}>peak</text>
      </svg>
    </div>
  )
}

export default function CyclicLR() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Cyclic learning rate policies oscillate the learning rate between bounds. The one-cycle
        policy, proposed by Leslie Smith, enables super-convergence: training in dramatically
        fewer iterations.
      </p>

      <DefinitionBlock title="Cyclic Learning Rate">
        <BlockMath math="\alpha_t = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min}) \cdot \max(0, 1 - |t / T_{\text{half}} - 1|)" />
        <p className="mt-2">
          The learning rate linearly increases from <InlineMath math="\alpha_{\min}" /> to
          <InlineMath math="\alpha_{\max}" />, then linearly decreases back, repeating
          cyclically. This triangular wave can help explore the loss landscape.
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="One-Cycle Policy">
        <p>A single cycle of: warm up to peak LR, then anneal down to a very small value.</p>
        <BlockMath math="\text{Phase 1: } \alpha_{\min} \to \alpha_{\max} \quad (\sim 30\% \text{ of training})" />
        <BlockMath math="\text{Phase 2: } \alpha_{\max} \to \alpha_{\min}/10^4 \quad (\sim 70\% \text{ of training})" />
      </DefinitionBlock>

      <OneCycleViz />

      <TheoremBlock title="Super-Convergence" id="super-convergence">
        <p>
          With the one-cycle policy, certain architectures can be trained in 1/5 to 1/10 of the
          usual number of epochs. The high learning rate phase acts as regularization (similar to
          large noise), while the final low LR phase fine-tunes to a sharp minimum.
        </p>
      </TheoremBlock>

      <ExampleBlock title="LR Range Test (Smith's Method)">
        <p>
          To find the optimal max LR: start with a very small LR and exponentially increase it
          over one epoch. Plot loss vs LR. The optimal max LR is typically where loss is still
          decreasing but before it diverges — usually one order of magnitude before the minimum.
        </p>
      </ExampleBlock>

      <PythonCode
        title="One-Cycle Policy in PyTorch"
        code={`import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# One-cycle policy
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.1, total_steps=1000,
    pct_start=0.3,        # 30% warmup
    div_factor=25,         # initial_lr = max_lr / 25
    final_div_factor=1e4,  # final_lr = initial_lr / 1e4
    anneal_strategy='cos'
)

# LR Range Test
def lr_range_test(model, data_loader, start_lr=1e-7, end_lr=10, steps=100):
    lrs, losses_list = [], []
    lr = start_lr
    mult = (end_lr / start_lr) ** (1 / steps)
    opt = torch.optim.SGD(model.parameters(), lr=start_lr)
    for i, (x, y) in zip(range(steps), data_loader):
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        opt.step(); opt.zero_grad()
        lrs.append(lr)
        losses_list.append(loss.item())
        lr *= mult
        for pg in opt.param_groups: pg['lr'] = lr
    return lrs, losses_list

print("One-cycle scheduler created with 1000 steps")`}
      />

      <NoteBlock type="note" title="When to Use One-Cycle">
        <p>
          One-cycle works best with SGD+momentum for CNNs and can dramatically reduce training
          time. For Transformers with Adam, cosine annealing with warmup is usually preferred.
          Always run the LR range test first to find the right peak learning rate.
        </p>
      </NoteBlock>
    </div>
  )
}

import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function ScheduleViz() {
  const [warmup, setWarmup] = useState(10)
  const [decayType, setDecayType] = useState('step')
  const W = 400, H = 180, totalSteps = 100
  const peakLR = 1.0

  const schedule = []
  for (let t = 0; t < totalSteps; t++) {
    let lr
    if (t < warmup) {
      lr = peakLR * (t + 1) / warmup
    } else if (decayType === 'step') {
      lr = t < 50 ? peakLR : t < 75 ? peakLR * 0.1 : peakLR * 0.01
    } else {
      const decay = Math.exp(-0.03 * (t - warmup))
      lr = peakLR * decay
    }
    schedule.push(lr)
  }

  const sx = W / totalSteps, sy = (H - 30) / (peakLR * 1.1)
  const path = schedule.map((lr, i) => `${i === 0 ? 'M' : 'L'}${i * sx},${H - 20 - lr * sy}`).join(' ')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Learning Rate Schedule</h3>
      <div className="flex items-center gap-4 mb-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Warmup: {warmup}
          <input type="range" min={0} max={30} step={1} value={warmup} onChange={e => setWarmup(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <div className="flex gap-2">
          {['step', 'exponential'].map(d => (
            <button key={d} onClick={() => setDecayType(d)}
              className={`px-3 py-1 rounded text-xs font-medium ${decayType === d ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'}`}>
              {d}
            </button>
          ))}
        </div>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={0} y1={H - 20} x2={W} y2={H - 20} stroke="#d1d5db" strokeWidth={0.5} />
        {warmup > 0 && <line x1={warmup * sx} y1={0} x2={warmup * sx} y2={H - 20} stroke="#f97316" strokeWidth={0.8} strokeDasharray="3,3" />}
        <path d={path} fill="none" stroke="#8b5cf6" strokeWidth={2} />
        {warmup > 0 && <text x={warmup * sx + 3} y={12} fill="#f97316" fontSize={9}>warmup end</text>}
      </svg>
    </div>
  )
}

export default function WarmupDecay() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Learning rate scheduling is crucial for training stability and final performance.
        Warmup prevents early instability, while decay ensures convergence to good minima.
      </p>

      <DefinitionBlock title="Linear Warmup">
        <BlockMath math="\alpha_t = \alpha_{\max} \cdot \frac{t}{T_{\text{warmup}}}, \quad t \leq T_{\text{warmup}}" />
        <p className="mt-2">
          Gradients are noisy and poorly conditioned early in training. Warmup lets statistics
          in Adam/BatchNorm stabilize before applying the full learning rate.
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="Step & Exponential Decay">
        <BlockMath math="\text{Step: } \alpha_t = \alpha_0 \cdot \gamma^{\lfloor t / s \rfloor}" />
        <BlockMath math="\text{Exponential: } \alpha_t = \alpha_0 \cdot e^{-\lambda t}" />
        <p className="mt-2">
          Step decay drops the learning rate by factor <InlineMath math="\gamma" /> every
          <InlineMath math="s" /> epochs. Exponential decay provides smoother reduction.
        </p>
      </DefinitionBlock>

      <ScheduleViz />

      <ExampleBlock title="Why Warmup Helps Adam">
        <p>
          At step 1, Adam's bias-corrected second moment <InlineMath math="\hat{v}_1 = g_1^2" /> is
          based on a single gradient, making the adaptive ratio highly unreliable. With warmup,
          by the time the full learning rate is reached, the moment estimates have accumulated
          enough data to be meaningful.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Warmup + Step Decay Schedule"
        code={`import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

model = nn.Linear(128, 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

warmup_steps = 1000
total_steps = 50000

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps            # linear warmup
    elif step < 30000:
        return 1.0                             # constant
    elif step < 40000:
        return 0.1                             # first drop
    else:
        return 0.01                            # second drop

scheduler = LambdaLR(optimizer, lr_lambda)

# Training loop pattern
for step in range(100):
    loss = model(torch.randn(8, 128)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

print(f"Final LR: {scheduler.get_last_lr()[0]:.6f}")`}
      />

      <WarningBlock title="Step Scheduler Before or After Optimizer?">
        <p>
          In PyTorch, always call <code>scheduler.step()</code> after <code>optimizer.step()</code>.
          Calling it before can skip the first learning rate value and lead to unexpected behavior.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Practical Guidelines">
        <p>
          <strong>Warmup duration</strong>: 1-5% of total training steps for Transformers, less
          for CNNs. <strong>Decay</strong>: step decay at 30% and 60% of training is a classic
          recipe for image classification. For language models, cosine decay is more common.
        </p>
      </NoteBlock>
    </div>
  )
}

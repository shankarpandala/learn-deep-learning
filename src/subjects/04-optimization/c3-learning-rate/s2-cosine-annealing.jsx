import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function CosineViz() {
  const [restarts, setRestarts] = useState(1)
  const [minLR, setMinLR] = useState(0.0)
  const W = 400, H = 180, totalSteps = 120
  const peakLR = 1.0

  const schedule = []
  const periodLen = Math.floor(totalSteps / restarts)
  for (let t = 0; t < totalSteps; t++) {
    const tInPeriod = t % periodLen
    const lr = minLR + 0.5 * (peakLR - minLR) * (1 + Math.cos(Math.PI * tInPeriod / periodLen))
    schedule.push(lr)
  }

  const sx = W / totalSteps, sy = (H - 30) / (peakLR * 1.1)
  const path = schedule.map((lr, i) => `${i === 0 ? 'M' : 'L'}${i * sx},${H - 20 - lr * sy}`).join(' ')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Cosine Annealing with Warm Restarts</h3>
      <div className="flex items-center gap-4 mb-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Restarts: {restarts}
          <input type="range" min={1} max={6} step={1} value={restarts} onChange={e => setRestarts(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Min LR: {minLR.toFixed(2)}
          <input type="range" min={0} max={0.3} step={0.01} value={minLR} onChange={e => setMinLR(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={0} y1={H - 20} x2={W} y2={H - 20} stroke="#d1d5db" strokeWidth={0.5} />
        <path d={path} fill="none" stroke="#8b5cf6" strokeWidth={2} />
        {restarts > 1 && Array.from({ length: restarts - 1 }, (_, i) => (
          <line key={i} x1={(i + 1) * periodLen * sx} y1={0} x2={(i + 1) * periodLen * sx} y2={H - 20} stroke="#f97316" strokeWidth={0.8} strokeDasharray="3,3" />
        ))}
      </svg>
      <div className="mt-2 text-center text-xs text-gray-500 dark:text-gray-400">
        Period length: {periodLen} steps {restarts > 1 && '| Orange lines = restart points'}
      </div>
    </div>
  )
}

export default function CosineAnnealing() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Cosine annealing smoothly decays the learning rate following a cosine curve. Combined with
        warm restarts (SGDR), it can escape local minima and build snapshot ensembles.
      </p>

      <DefinitionBlock title="Cosine Annealing Schedule">
        <BlockMath math="\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})\left(1 + \cos\!\left(\frac{\pi \, t}{T}\right)\right)" />
        <p className="mt-2">
          The learning rate starts at <InlineMath math="\alpha_{\max}" />, smoothly decreases to
          <InlineMath math="\alpha_{\min}" /> following a half cosine curve over <InlineMath math="T" /> steps.
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="SGDR: Warm Restarts">
        <BlockMath math="\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})\left(1 + \cos\!\left(\frac{\pi \, T_{\text{cur}}}{T_i}\right)\right)" />
        <p className="mt-2">
          After each period <InlineMath math="T_i" />, the learning rate jumps back
          to <InlineMath math="\alpha_{\max}" />. Period lengths can increase
          with <InlineMath math="T_i = T_0 \cdot T_{\text{mult}}^i" />.
        </p>
      </DefinitionBlock>

      <CosineViz />

      <TheoremBlock title="Why Cosine Works" id="cosine-intuition">
        <p>
          The cosine schedule spends most of its time at moderate learning rates (the flat part
          of the cosine near 0 and <InlineMath math="\pi" />). Compared to linear decay, it provides
          more aggressive early reduction and gentler final convergence, matching the typical
          optimization landscape of neural networks.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Snapshot Ensembles">
        <p>
          With warm restarts, save model weights at the end of each cosine cycle (the minimum LR
          point). Average predictions from these snapshots for a free ensemble that typically
          improves accuracy by 0.5-1% without extra training cost.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Cosine Annealing in PyTorch"
        code={`import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

model = nn.Linear(128, 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Simple cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# Or with warm restarts (SGDR)
scheduler_wr = CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

lrs = []
for step in range(100):
    optimizer.step()
    scheduler.step()
    lrs.append(scheduler.get_last_lr()[0])

print(f"LR at step 0: {lrs[0]:.6f}")
print(f"LR at step 25: {lrs[25]:.6f}")
print(f"LR at step 49: {lrs[49]:.6f}")`}
      />

      <NoteBlock type="note" title="Cosine is the Default for LLMs">
        <p>
          Nearly all modern LLM training runs (GPT, LLaMA, etc.) use cosine decay, typically
          with linear warmup for the first 1-2% of steps and a minimum LR of 10% of the
          peak rate. This has become the de facto standard schedule.
        </p>
      </NoteBlock>
    </div>
  )
}

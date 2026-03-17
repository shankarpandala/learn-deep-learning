import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function LRDecayViz() {
  const [steps, setSteps] = useState(30)
  const [rho, setRho] = useState(0.9)
  const W = 400, H = 200

  const adagradLR = []
  const rmspropLR = []
  let accum = 0, emaG = 0
  const baseLR = 0.1
  for (let t = 1; t <= 60; t++) {
    const g = 1.0 + 0.3 * Math.sin(t * 0.5)
    accum += g * g
    emaG = rho * emaG + (1 - rho) * g * g
    adagradLR.push(baseLR / (Math.sqrt(accum) + 1e-8))
    rmspropLR.push(baseLR / (Math.sqrt(emaG) + 1e-8))
  }

  const maxLR = Math.max(...rmspropLR, ...adagradLR) * 1.1
  const sx = W / 62, sy = H / maxLR

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Effective Learning Rate Over Time</h3>
      <div className="flex items-center gap-4 mb-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Steps: {steps}
          <input type="range" min={5} max={60} step={1} value={steps} onChange={e => setSteps(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          ρ = {rho.toFixed(2)}
          <input type="range" min={0.5} max={0.999} step={0.01} value={rho} onChange={e => setRho(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={0} y1={H - 1} x2={W} y2={H - 1} stroke="#d1d5db" strokeWidth={0.5} />
        {adagradLR.slice(0, steps).map((lr, i, a) => i > 0 && (
          <line key={`a-${i}`} x1={(i) * sx} y1={H - a[i-1] * sy} x2={(i + 1) * sx} y2={H - lr * sy} stroke="#9ca3af" strokeWidth={1.8} />
        ))}
        {rmspropLR.slice(0, steps).map((lr, i, a) => i > 0 && (
          <line key={`r-${i}`} x1={(i) * sx} y1={H - a[i-1] * sy} x2={(i + 1) * sx} y2={H - lr * sy} stroke="#8b5cf6" strokeWidth={2} />
        ))}
      </svg>
      <div className="mt-2 flex justify-center gap-6 text-xs">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-violet-500" /> RMSProp</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-gray-400" /> AdaGrad</span>
      </div>
    </div>
  )
}

export default function AdagradRMSProp() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        AdaGrad introduced per-parameter adaptive learning rates by accumulating squared gradients.
        RMSProp fixes AdaGrad's aggressive decay by using an exponential moving average instead.
      </p>

      <DefinitionBlock title="AdaGrad">
        <BlockMath math="G_t = G_{t-1} + g_t \odot g_t" />
        <BlockMath math="\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{G_t} + \epsilon} \odot g_t" />
        <p className="mt-2">
          Each parameter gets its own effective learning rate that decreases as its cumulative
          gradient grows. Parameters with sparse or small gradients retain larger learning rates.
        </p>
      </DefinitionBlock>

      <WarningBlock title="AdaGrad's Learning Rate Decay">
        <p>
          Since <InlineMath math="G_t" /> only grows, the effective learning rate monotonically
          decreases to zero. For non-convex problems (deep learning), this causes premature
          convergence — training effectively stops too early.
        </p>
      </WarningBlock>

      <DefinitionBlock title="RMSProp (Hinton, 2012)">
        <BlockMath math="v_t = \rho \, v_{t-1} + (1 - \rho)\, g_t^2" />
        <BlockMath math="\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{v_t} + \epsilon} \odot g_t" />
        <p className="mt-2">
          The decay factor <InlineMath math="\rho" /> (typically 0.9 or 0.99) keeps a moving
          window of recent gradient magnitudes, preventing the denominator from growing unboundedly.
        </p>
      </DefinitionBlock>

      <LRDecayViz />

      <ExampleBlock title="Sparse Features">
        <p>
          In NLP with one-hot embeddings, common words get frequent gradient updates while rare
          words get few. AdaGrad/RMSProp automatically give rare words larger effective learning
          rates, helping them learn from limited data.
        </p>
      </ExampleBlock>

      <PythonCode
        title="AdaGrad & RMSProp in PyTorch"
        code={`import torch
import torch.optim as optim
import torch.nn as nn

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
x = torch.randn(32, 784)
target = torch.randint(0, 10, (32,))

# AdaGrad
opt_ada = optim.Adagrad(model.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss()(model(x), target)
loss.backward(); opt_ada.step(); opt_ada.zero_grad()
print(f"AdaGrad loss: {loss.item():.4f}")

# RMSProp — usually preferred for deep learning
opt_rms = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)
loss = nn.CrossEntropyLoss()(model(x), target)
loss.backward(); opt_rms.step(); opt_rms.zero_grad()
print(f"RMSProp loss: {loss.item():.4f}")`}
      />

      <NoteBlock type="note" title="AdaGrad Still Shines for Sparse Problems">
        <p>
          Despite its limitations in deep learning, AdaGrad remains the optimizer of choice for
          sparse problems like recommendation systems and click-through-rate prediction, where its
          decaying rate naturally handles frequently occurring features.
        </p>
      </NoteBlock>
    </div>
  )
}

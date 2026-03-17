import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function BiasCorrection() {
  const [beta1, setBeta1] = useState(0.9)
  const W = 380, H = 180
  const steps = 20

  const uncorrected = []
  const corrected = []
  let m = 0
  const trueGrad = 1.0
  for (let t = 1; t <= steps; t++) {
    m = beta1 * m + (1 - beta1) * trueGrad
    const mHat = m / (1 - Math.pow(beta1, t))
    uncorrected.push(m)
    corrected.push(mHat)
  }

  const sx = W / (steps + 2), sy = H / 1.5

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Bias Correction Effect</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        β₁ = {beta1.toFixed(2)}
        <input type="range" min={0.5} max={0.999} step={0.01} value={beta1} onChange={e => setBeta1(parseFloat(e.target.value))} className="w-32 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={sx} y1={H - trueGrad * sy} x2={W} y2={H - trueGrad * sy} stroke="#d1d5db" strokeWidth={0.8} strokeDasharray="4,4" />
        <text x={W - 60} y={H - trueGrad * sy - 5} fill="#9ca3af" fontSize={10}>true mean</text>
        {uncorrected.map((v, i, a) => i > 0 && (
          <line key={`u-${i}`} x1={(i) * sx + sx} y1={H - a[i-1] * sy} x2={(i + 1) * sx + sx} y2={H - v * sy} stroke="#9ca3af" strokeWidth={1.5} />
        ))}
        {corrected.map((v, i, a) => i > 0 && (
          <line key={`c-${i}`} x1={(i) * sx + sx} y1={H - a[i-1] * sy} x2={(i + 1) * sx + sx} y2={H - v * sy} stroke="#8b5cf6" strokeWidth={2} />
        ))}
      </svg>
      <div className="mt-2 flex justify-center gap-6 text-xs">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-violet-500" /> Bias-corrected</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-gray-400" /> Uncorrected</span>
      </div>
    </div>
  )
}

export default function AdamOptimizer() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Adam combines the momentum idea (first moment) with adaptive learning rates (second moment),
        making it the most widely used optimizer in deep learning. AdamW improves it with properly
        decoupled weight decay.
      </p>

      <DefinitionBlock title="Adam (Adaptive Moment Estimation)">
        <BlockMath math="m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(first moment)}" />
        <BlockMath math="v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(second moment)}" />
        <BlockMath math="\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(bias correction)}" />
        <BlockMath math="\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t" />
      </DefinitionBlock>

      <BiasCorrection />

      <DefinitionBlock title="AdamW (Decoupled Weight Decay)">
        <BlockMath math="\theta_t = \theta_{t-1} - \alpha\!\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\,\theta_{t-1}\right)" />
        <p className="mt-2">
          In Adam with L2 regularization, the penalty gradient is also adapted, weakening it
          for parameters with large gradients. AdamW applies weight decay directly to the
          parameters, outside the adaptive mechanism, giving proper regularization.
        </p>
      </DefinitionBlock>

      <TheoremBlock title="Default Hyperparameters" id="adam-defaults">
        <p>The recommended defaults from the original paper (Kingma & Ba, 2015):</p>
        <BlockMath math="\alpha = 0.001, \quad \beta_1 = 0.9, \quad \beta_2 = 0.999, \quad \epsilon = 10^{-8}" />
        <p>
          For AdamW, typical weight decay <InlineMath math="\lambda \in [0.01, 0.1]" />.
          Many LLM training runs use <InlineMath math="\beta_2 = 0.95" /> for stability.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Adam vs AdamW on Regularized Models">
        <p>
          With <InlineMath math="\lambda = 0.01" /> weight decay on a Transformer, AdamW typically
          achieves 1-3% better validation accuracy because the regularization is not distorted
          by the adaptive scaling. This difference grows with model size.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Adam & AdamW in PyTorch"
        code={`import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
x, target = torch.randn(32, 784), torch.randint(0, 10, (32,))

# AdamW — preferred for most tasks
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01
)

loss = nn.CrossEntropyLoss()(model(x), target)
loss.backward()
optimizer.step()
optimizer.zero_grad()
print(f"Loss: {loss.item():.4f}")

# Check parameter norms — AdamW regularizes these
norms = [p.norm().item() for p in model.parameters()]
print(f"Param norms: {[f'{n:.3f}' for n in norms]}")`}
      />

      <WarningBlock title="Adam Can Generalize Poorly">
        <p>
          Adam sometimes converges to sharper minima than SGD with momentum, leading to worse
          generalization. AdamW mitigates this, and combining Adam with learning rate warmup
          and cosine decay further helps. For vision tasks, SGD+momentum often still wins.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="When to Use Adam vs SGD">
        <p>
          <strong>Adam/AdamW</strong>: Transformers, NLP, generative models, quick prototyping.
          <strong> SGD+momentum</strong>: CNNs for image classification (often better generalization
          with proper tuning). When in doubt, start with AdamW.
        </p>
      </NoteBlock>
    </div>
  )
}

import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function mse(e) { return e * e }
function mae(e) { return Math.abs(e) }
function huber(e, delta) { return Math.abs(e) <= delta ? 0.5 * e * e : delta * (Math.abs(e) - 0.5 * delta) }

function LossPlot() {
  const [delta, setDelta] = useState(1.0)
  const W = 420, H = 250, ox = W / 2, oy = H * 0.8, sx = 30, sy = 15

  const range = Array.from({ length: 161 }, (_, i) => -4 + i * 0.05)
  const toSVG = (x, y) => `${ox + x * sx},${oy - y * sy}`

  const msePath = range.map((x, i) => `${i === 0 ? 'M' : 'L'}${toSVG(x, Math.min(mse(x), 12))}`).join(' ')
  const maePath = range.map((x, i) => `${i === 0 ? 'M' : 'L'}${toSVG(x, mae(x))}`).join(' ')
  const huberPath = range.map((x, i) => `${i === 0 ? 'M' : 'L'}${toSVG(x, huber(x, delta))}`).join(' ')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Loss Function Comparison</h3>
      <label className="flex items-center gap-2 mb-3 text-sm text-gray-600 dark:text-gray-400">
        Huber δ = {delta.toFixed(1)}
        <input type="range" min={0.1} max={3} step={0.1} value={delta} onChange={e => setDelta(parseFloat(e.target.value))} className="w-32 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={0} y1={oy} x2={W} y2={oy} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={ox} y1={0} x2={ox} y2={H} stroke="#d1d5db" strokeWidth={0.5} />
        <text x={ox + 4 * sx + 5} y={oy + 15} fontSize={10} fill="#9ca3af">error</text>
        <path d={msePath} fill="none" stroke="#8b5cf6" strokeWidth={2.5} />
        <path d={maePath} fill="none" stroke="#f97316" strokeWidth={2.5} />
        <path d={huberPath} fill="none" stroke="#10b981" strokeWidth={2.5} />
      </svg>
      <div className="mt-2 flex justify-center gap-4 text-xs">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-violet-500" /> MSE</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-orange-500" /> MAE</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-emerald-500" /> Huber</span>
      </div>
    </div>
  )
}

export default function RegressionLosses() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Loss functions quantify the gap between predictions and true values. The choice of loss
        function fundamentally shapes what a neural network learns and how it behaves with outliers.
      </p>

      <DefinitionBlock title="Mean Squared Error (MSE)">
        <BlockMath math="\mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2" />
        <p className="mt-2">Gradient: <InlineMath math="\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)" /></p>
      </DefinitionBlock>

      <DefinitionBlock title="Mean Absolute Error (MAE)">
        <BlockMath math="\mathcal{L}_{\text{MAE}} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|" />
        <p className="mt-2">Gradient: <InlineMath math="\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{1}{n}\text{sign}(\hat{y}_i - y_i)" /></p>
      </DefinitionBlock>

      <DefinitionBlock title="Huber Loss (Smooth L1)">
        <BlockMath math="\mathcal{L}_\delta(e) = \begin{cases} \frac{1}{2}e^2 & \text{if } |e| \leq \delta \\ \delta(|e| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}" />
        <p className="mt-2">Combines the best of both: quadratic near zero, linear for large errors.</p>
      </DefinitionBlock>

      <LossPlot />

      <TheoremBlock title="MSE as Maximum Likelihood" id="mse-mle">
        <p>
          MSE is equivalent to maximum likelihood estimation under the assumption
          of Gaussian noise:
        </p>
        <BlockMath math="y_i = f(\mathbf{x}_i) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)" />
        <p>
          Minimizing MSE = maximizing the log-likelihood of the Gaussian model.
          Similarly, MAE corresponds to Laplacian noise assumptions.
        </p>
      </TheoremBlock>

      <WarningBlock title="MSE Gradient Explosion">
        <p>
          For large errors, the MSE gradient <InlineMath math="2(\hat{y} - y)" /> grows linearly,
          potentially causing unstable training. Huber loss caps the gradient
          at <InlineMath math="\delta" />, providing stability while maintaining sensitivity near zero.
        </p>
      </WarningBlock>

      <ExampleBlock title="Numerical Comparison">
        <p>For true value <InlineMath math="y = 3.0" /> and predictions:</p>
        <div className="mt-2 overflow-x-auto">
          <table className="text-sm border-collapse">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="py-1 pr-4">ŷ</th><th className="py-1 pr-4">Error</th><th className="py-1 pr-4">MSE</th><th className="py-1 pr-4">MAE</th><th className="py-1">Huber(δ=1)</th>
              </tr>
            </thead>
            <tbody className="text-gray-700 dark:text-gray-300">
              <tr><td className="py-1 pr-4">2.5</td><td className="py-1 pr-4">0.5</td><td className="py-1 pr-4">0.25</td><td className="py-1 pr-4">0.5</td><td className="py-1">0.125</td></tr>
              <tr><td className="py-1 pr-4">1.0</td><td className="py-1 pr-4">2.0</td><td className="py-1 pr-4">4.0</td><td className="py-1 pr-4">2.0</td><td className="py-1">1.5</td></tr>
              <tr><td className="py-1 pr-4">10.0</td><td className="py-1 pr-4">7.0</td><td className="py-1 pr-4">49.0</td><td className="py-1 pr-4">7.0</td><td className="py-1">6.5</td></tr>
            </tbody>
          </table>
        </div>
      </ExampleBlock>

      <PythonCode
        title="Loss Functions in PyTorch"
        code={`import torch
import torch.nn as nn

y_true = torch.tensor([3.0, 1.0, 4.0, 2.0])
y_pred = torch.tensor([2.5, 1.5, 3.0, 2.5])

mse = nn.MSELoss()
mae = nn.L1Loss()
huber = nn.SmoothL1Loss(beta=1.0)  # Huber with delta=1

print(f"MSE:   {mse(y_pred, y_true):.4f}")
print(f"MAE:   {mae(y_pred, y_true):.4f}")
print(f"Huber: {huber(y_pred, y_true):.4f}")

# Custom Huber with adjustable delta
def huber_loss(pred, target, delta=1.0):
    error = pred - target
    abs_error = torch.abs(error)
    quadratic = 0.5 * error ** 2
    linear = delta * (abs_error - 0.5 * delta)
    return torch.where(abs_error <= delta, quadratic, linear).mean()`}
      />

      <NoteBlock type="tip" title="Choosing a Regression Loss">
        <p>
          <strong>MSE</strong>: Default choice; penalizes large errors heavily.
          <strong> MAE</strong>: Robust to outliers; constant gradient magnitude.
          <strong> Huber</strong>: Best of both worlds; use when data may have outliers.
          In deep learning, Huber loss is used in DQN (reinforcement learning) and
          object detection regression heads.
        </p>
      </NoteBlock>
    </div>
  )
}

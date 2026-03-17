import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ProbabilisticForecastViz() {
  const [uncertainty, setUncertainty] = useState(1.0)
  const W = 400, H = 160, N = 40, split = 25

  const actual = Array.from({ length: N }, (_, i) => 2 * Math.sin(i * 0.3) + 0.5 * Math.cos(i * 0.7))
  const yMin = -4, yMax = 4
  const toSVG = (i, v) => `${(i / N) * W},${H - ((v - yMin) / (yMax - yMin)) * H}`

  const histPath = actual.slice(0, split).map((v, i) => `${i === 0 ? 'M' : 'L'}${toSVG(i, v)}`).join(' ')
  const meanPath = actual.slice(split - 1).map((v, i) => `${i === 0 ? 'M' : 'L'}${toSVG(split - 1 + i, v + Math.sin(i * 0.2) * 0.2)}`).join(' ')

  const bandPoints = actual.slice(split - 1).map((v, i) => {
    const x = ((split - 1 + i) / N) * W
    const m = v + Math.sin(i * 0.2) * 0.2
    const spread = uncertainty * (0.3 + i * 0.08)
    return { x, upper: H - ((m + spread - yMin) / (yMax - yMin)) * H, lower: H - ((m - spread - yMin) / (yMax - yMin)) * H }
  })
  const bandPath = bandPoints.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.upper}`).join(' ') + bandPoints.reverse().map((p) => `L${p.x},${p.lower}`).join(' ') + 'Z'

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Probabilistic Forecast</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Uncertainty scale: {uncertainty.toFixed(1)}
        <input type="range" min={0.2} max={3} step={0.1} value={uncertainty} onChange={e => setUncertainty(parseFloat(e.target.value))} className="w-32 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <rect x={(split / N) * W} y={0} width={W - (split / N) * W} height={H} fill="#8b5cf6" opacity={0.05} />
        <path d={bandPath} fill="#8b5cf6" opacity={0.2} />
        <path d={histPath} fill="none" stroke="#6b7280" strokeWidth={2} />
        <path d={meanPath} fill="none" stroke="#8b5cf6" strokeWidth={2} />
        <line x1={(split / N) * W} y1={0} x2={(split / N) * W} y2={H} stroke="#9ca3af" strokeDasharray="4,3" strokeWidth={1} />
        <text x={(split / N) * W + 4} y={12} className="text-[10px] fill-gray-400">forecast</text>
      </svg>
      <div className="mt-2 flex justify-center gap-4 text-xs text-gray-500">
        <span>Prediction intervals widen with horizon — reflecting growing uncertainty</span>
      </div>
    </div>
  )
}

export default function DeepARProbabilisticForecasting() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        DeepAR is an autoregressive RNN model that produces probabilistic forecasts by
        parameterizing a likelihood function at each time step. It learns across many
        related time series, sharing patterns via global parameters.
      </p>

      <DefinitionBlock title="DeepAR Model">
        <p>At each step <InlineMath math="t" />, an LSTM computes hidden state <InlineMath math="h_t" /> and outputs distribution parameters:</p>
        <BlockMath math="h_t = \text{LSTM}(h_{t-1},\; [x_{t-1},\; \mathbf{c}_t])" />
        <BlockMath math="\mu_t, \sigma_t = \text{MLP}(h_t), \quad z_t \sim \mathcal{N}(\mu_t, \sigma_t^2)" />
        <p className="mt-2">where <InlineMath math="\mathbf{c}_t" /> contains covariates (time features, static embeddings).</p>
      </DefinitionBlock>

      <ProbabilisticForecastViz />

      <TheoremBlock title="Negative Log-Likelihood Loss" id="deepar-loss">
        <p>DeepAR is trained by maximizing the log-likelihood of the observed data:</p>
        <BlockMath math="\mathcal{L} = -\sum_{t=1}^{T} \log p(x_t \mid \mu_t, \sigma_t) = \sum_{t=1}^{T}\left[\log \sigma_t + \frac{(x_t - \mu_t)^2}{2\sigma_t^2}\right] + C" />
        <p>For count data, a negative binomial likelihood replaces the Gaussian.</p>
      </TheoremBlock>

      <ExampleBlock title="Quantile Regression Alternative">
        <p>Instead of parametric distributions, predict quantiles directly. The pinball loss for quantile <InlineMath math="q" />:</p>
        <BlockMath math="\mathcal{L}_q(y, \hat{y}) = \begin{cases} q(y - \hat{y}) & \text{if } y \geq \hat{y} \\ (1-q)(\hat{y} - y) & \text{if } y < \hat{y} \end{cases}" />
      </ExampleBlock>

      <PythonCode
        title="DeepAR-Style Probabilistic Forecaster"
        code={`import torch
import torch.nn as nn

class SimpleDeepAR(nn.Module):
    def __init__(self, input_dim=1, hidden=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers, batch_first=True)
        self.mu_head = nn.Linear(hidden, 1)
        self.sigma_head = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())

    def forward(self, x):  # x: (B, T, 1)
        h, _ = self.lstm(x)
        mu = self.mu_head(h)        # (B, T, 1)
        sigma = self.sigma_head(h)  # (B, T, 1), always positive
        return mu, sigma

    def loss(self, x):
        mu, sigma = self.forward(x[:, :-1, :])  # teacher forcing
        target = x[:, 1:, :]
        nll = torch.log(sigma) + 0.5 * ((target - mu) / sigma)**2
        return nll.mean()

model = SimpleDeepAR()
x = torch.randn(16, 48, 1)  # batch=16, seq=48
print(f"NLL loss: {model.loss(x).item():.4f}")`}
      />

      <NoteBlock type="note" title="Multi-Series Training">
        <p>
          DeepAR's key advantage is training a single global model across thousands of related
          time series. Each series gets a learned embedding vector, allowing the model to share
          seasonal and trend patterns while adapting to individual series characteristics.
        </p>
      </NoteBlock>
    </div>
  )
}

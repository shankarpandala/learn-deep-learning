import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function PredictionHorizonDemo() {
  const [horizon, setHorizon] = useState(5)
  const [method, setMethod] = useState('deterministic')

  const methods = {
    deterministic: { name: 'Deterministic', blur: horizon * 3, desc: 'Single prediction, increasingly blurry' },
    stochastic: { name: 'Stochastic (VAE)', blur: Math.min(horizon, 3), desc: 'Samples diverse futures, sharper' },
    diffusion: { name: 'Diffusion', blur: Math.min(horizon, 2), desc: 'High quality, temporally consistent' },
  }

  const m = methods[method]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Video Prediction Quality vs Horizon</h3>
      <div className="flex flex-wrap gap-4 mb-4">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Prediction horizon: {horizon} frames
          <input type="range" min={1} max={20} step={1} value={horizon} onChange={e => setHorizon(Number(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <div className="flex gap-2">
          {Object.entries(methods).map(([key, val]) => (
            <button key={key} onClick={() => setMethod(key)}
              className={`rounded-lg px-3 py-1 text-xs font-medium ${method === key ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'}`}>
              {val.name}
            </button>
          ))}
        </div>
      </div>
      <div className="flex gap-1 items-end">
        {Array.from({ length: 10 }).map((_, i) => {
          const isContext = i < 5
          const predIdx = i - 5
          const quality = isContext ? 100 : Math.max(10, 100 - m.blur * (predIdx + 1) * 2)
          return (
            <div key={i} className="flex-1 flex flex-col items-center">
              <div className={`w-full rounded-t ${isContext ? 'bg-violet-500' : quality > 60 ? 'bg-violet-400' : quality > 30 ? 'bg-violet-300' : 'bg-violet-200'}`}
                style={{ height: `${quality}px`, opacity: isContext ? 1 : 0.5 + quality / 200 }} />
              <span className="text-xs text-gray-500 mt-1">{isContext ? `c${i}` : `p${predIdx}`}</span>
            </div>
          )
        })}
      </div>
      <p className="text-xs text-gray-500 mt-2">{m.desc}. Estimated quality at horizon {horizon}: {Math.max(10, 100 - m.blur * horizon * 2).toFixed(0)}%</p>
    </div>
  )
}

export default function VideoPrediction() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Video prediction generates future frames given past observations, testing a model's
        understanding of scene dynamics, physics, and object permanence. It serves as both
        a self-supervised learning task and a core component of world models.
      </p>

      <DefinitionBlock title="Video Prediction Problem">
        <p>Given context frames <InlineMath math="x_{1:T}" />, predict future frames:</p>
        <BlockMath math="P(x_{T+1:T+K} | x_{1:T})" />
        <p className="mt-2">
          The future is inherently uncertain: a ball at the edge of a table might fall or stay.
          Deterministic models produce blurry averages of possible futures, motivating
          stochastic approaches that model the full distribution of outcomes.
        </p>
      </DefinitionBlock>

      <PredictionHorizonDemo />

      <TheoremBlock title="Stochastic Video Prediction" id="stochastic-prediction">
        <p>
          SVG (Stochastic Video Generation) uses a learned prior to sample diverse futures:
        </p>
        <BlockMath math="x_{t+1} = g_\theta(x_t, z_t), \quad z_t \sim q_\phi(z_t | x_{1:t+1}) \text{ (training)}" />
        <p className="mt-1">
          At test time, <InlineMath math="z_t \sim p_\psi(z_t | x_{1:t})" /> is sampled from a learned
          prior. The KL divergence between posterior and prior ensures the prior can generate
          meaningful latent codes:
        </p>
        <BlockMath math="\mathcal{L} = \|x_{t+1} - \hat{x}_{t+1}\|^2 + \beta \, D_\text{KL}(q_\phi \| p_\psi)" />
      </TheoremBlock>

      <ExampleBlock title="Evaluation Metrics">
        <p>Video prediction quality is measured by multiple complementary metrics:</p>
        <ul className="list-disc pl-5 mt-2 space-y-1">
          <li><strong>PSNR/SSIM:</strong> Pixel-level quality (penalizes blur)</li>
          <li><strong>FVD (Frechet Video Distance):</strong> Distribution-level quality using I3D features</li>
          <li><strong>LPIPS:</strong> Perceptual quality from deep feature distances</li>
        </ul>
        <BlockMath math="\text{FVD} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})" />
      </ExampleBlock>

      <PythonCode
        title="Simple Video Prediction Model"
        code={`import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=pad)
        self.hidden_ch = hidden_ch

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new

class VideoPredictionModel(nn.Module):
    def __init__(self, channels=3, hidden=64):
        super().__init__()
        self.encoder = nn.Conv2d(channels, hidden, 3, stride=2, padding=1)
        self.lstm = ConvLSTMCell(hidden, hidden)
        self.decoder = nn.ConvTranspose2d(hidden, channels, 4, stride=2, padding=1)

    def forward(self, context, n_future=5):
        B, T, C, H, W = context.shape
        h = torch.zeros(B, 64, H // 2, W // 2, device=context.device)
        c = torch.zeros_like(h)

        # Encode context frames
        for t in range(T):
            enc = torch.relu(self.encoder(context[:, t]))
            h, c = self.lstm(enc, (h, c))

        # Predict future frames
        predictions = []
        for _ in range(n_future):
            h, c = self.lstm(h, (h, c))
            pred = torch.sigmoid(self.decoder(h))
            predictions.append(pred)

        return torch.stack(predictions, dim=1)

model = VideoPredictionModel()
context = torch.rand(2, 5, 3, 64, 64)  # 5 context frames
future = model(context, n_future=10)
print(f"Context: {context.shape}")     # [2, 5, 3, 64, 64]
print(f"Predicted: {future.shape}")    # [2, 10, 3, 64, 64]`}
      />

      <NoteBlock type="note" title="World Models and Video Prediction">
        <p>
          Video prediction is a core capability of <strong>world models</strong> for autonomous agents.
          Models like GAIA-1 (Wayve) and DreamerV3 learn to predict future observations from
          actions, enabling planning in imagination. The transition from pixel prediction to
          latent-space prediction dramatically improves both quality and computational efficiency.
        </p>
      </NoteBlock>
    </div>
  )
}

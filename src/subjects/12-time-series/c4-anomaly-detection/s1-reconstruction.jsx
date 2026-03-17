import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ReconstructionErrorViz() {
  const [threshold, setThreshold] = useState(1.2)
  const N = 60, W = 420, H = 160

  const original = Array.from({ length: N }, (_, i) => Math.sin(i * 0.3) + 0.3 * Math.cos(i * 0.7))
  const anomalyIdx = [18, 19, 42, 43, 44]
  const withAnomaly = original.map((v, i) => anomalyIdx.includes(i) ? v + 2.5 : v)
  const reconstructed = original.map((v, i) => v + (Math.sin(i * 11.3) * 0.1))
  const errors = withAnomaly.map((v, i) => Math.abs(v - reconstructed[i]))

  const yMin = -2, yMax = 4
  const toSVG = (i, v) => `${(i / N) * W},${H * 0.6 - ((v - yMin) / (yMax - yMin)) * H * 0.6}`
  const origPath = withAnomaly.map((v, i) => `${i === 0 ? 'M' : 'L'}${toSVG(i, v)}`).join(' ')
  const recPath = reconstructed.map((v, i) => `${i === 0 ? 'M' : 'L'}${toSVG(i, v)}`).join(' ')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Reconstruction-Based Anomaly Detection</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Threshold: {threshold.toFixed(1)}
        <input type="range" min={0.3} max={3} step={0.1} value={threshold} onChange={e => setThreshold(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <path d={origPath} fill="none" stroke="#6b7280" strokeWidth={1.5} />
        <path d={recPath} fill="none" stroke="#8b5cf6" strokeWidth={1.5} strokeDasharray="4,3" />
        {errors.map((e, i) => {
          const x = (i / N) * W, barH = (e / 3) * 40
          const isAnomaly = e > threshold
          return <rect key={i} x={x} y={H - barH} width={W / N - 1} height={barH} fill={isAnomaly ? '#ef4444' : '#d1d5db'} opacity={0.7} rx={1} />
        })}
        <line x1={0} y1={H - (threshold / 3) * 40} x2={W} y2={H - (threshold / 3) * 40} stroke="#ef4444" strokeWidth={1} strokeDasharray="5,3" />
        <text x={W - 4} y={H - (threshold / 3) * 40 - 4} textAnchor="end" className="text-[9px] fill-red-500">threshold</text>
      </svg>
      <div className="mt-2 flex justify-center gap-4 text-xs text-gray-500">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-gray-500" /> Original</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-violet-500" style={{ borderBottom: '1px dashed' }} /> Reconstructed</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 bg-red-400 rounded" /> Anomaly</span>
      </div>
    </div>
  )
}

export default function ReconstructionBasedDetection() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Reconstruction-based anomaly detection trains an autoencoder on normal data. At
        inference, anomalies produce high reconstruction error because the model has
        never learned to reproduce abnormal patterns.
      </p>

      <DefinitionBlock title="Reconstruction Error Anomaly Score">
        <p>An autoencoder <InlineMath math="f_\theta" /> maps a window <InlineMath math="\mathbf{x}" /> to its reconstruction <InlineMath math="\hat{\mathbf{x}}" />. The anomaly score is:</p>
        <BlockMath math="a(\mathbf{x}) = \|\mathbf{x} - f_\theta(\mathbf{x})\|_2^2" />
        <p className="mt-2">A point is flagged as anomalous if <InlineMath math="a(\mathbf{x}) > \tau" />, where <InlineMath math="\tau" /> is a threshold set on validation data.</p>
      </DefinitionBlock>

      <ReconstructionErrorViz />

      <TheoremBlock title="LSTM-Autoencoder Architecture" id="lstm-ae">
        <p>The encoder LSTM compresses the input window into a fixed-size latent vector:</p>
        <BlockMath math="\mathbf{z} = \text{LSTM}_{\text{enc}}(\mathbf{x}_1, \ldots, \mathbf{x}_T) \in \mathbb{R}^d" />
        <p>The decoder LSTM reconstructs the sequence from <InlineMath math="\mathbf{z}" />:</p>
        <BlockMath math="\hat{\mathbf{x}}_t = \text{MLP}(\text{LSTM}_{\text{dec}}(\mathbf{z}, \hat{\mathbf{x}}_{t-1}))" />
      </TheoremBlock>

      <ExampleBlock title="Threshold Selection Strategies">
        <p>
          Common approaches: (1) fixed percentile (e.g., 99th) of training reconstruction errors,
          (2) mean + <InlineMath math="k\sigma" /> of training errors, or (3) learned threshold via
          a small labeled validation set. Dynamic thresholds that adapt over time handle
          concept drift better than static ones.
        </p>
      </ExampleBlock>

      <PythonCode
        title="LSTM Autoencoder for Anomaly Detection"
        code={`import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden=64, latent=32, n_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden, n_layers, batch_first=True)
        self.compress = nn.Linear(hidden, latent)
        self.expand = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, n_layers, batch_first=True)
        self.output = nn.Linear(hidden, input_dim)

    def forward(self, x):  # x: (B, T, 1)
        _, (h_n, _) = self.encoder(x)
        z = self.compress(h_n[-1])               # (B, latent)
        z_exp = self.expand(z).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(z_exp)
        return self.output(dec_out)               # (B, T, 1)

# Train on normal data, detect anomalies via reconstruction error
model = LSTMAutoencoder()
normal_data = torch.sin(torch.linspace(0, 10*3.14, 200)).reshape(1, 200, 1)
recon = model(normal_data)
error = ((normal_data - recon)**2).mean(dim=-1).squeeze()
threshold = error.mean() + 3 * error.std()
print(f"Threshold: {threshold.item():.4f}")
print(f"Anomalous steps: {(error > threshold).sum().item()}")`}
      />

      <NoteBlock type="note" title="VAE for Richer Anomaly Scores">
        <p>
          Variational autoencoders (VAEs) provide a principled anomaly score via the ELBO:
          both reconstruction error and KL divergence from the prior contribute. Points
          that map to unusual latent regions (high KL) are anomalous even if reconstruction
          appears acceptable.
        </p>
      </NoteBlock>
    </div>
  )
}

import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function BottleneckVisualizer() {
  const [bottleneck, setBottleneck] = useState(8)
  const inputDim = 64
  const compressionRatio = (bottleneck / inputDim * 100).toFixed(1)

  const layers = [inputDim, 32, bottleneck, 32, inputDim]
  const maxH = 160
  const W = 400

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Autoencoder Bottleneck</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Latent dim: {bottleneck}
        <input type="range" min={2} max={32} step={1} value={bottleneck} onChange={e => setBottleneck(parseInt(e.target.value))} className="w-40 accent-violet-500" />
        <span className="text-xs">({compressionRatio}% of input)</span>
      </label>
      <svg width={W} height={maxH + 30} className="mx-auto block">
        {layers.map((dim, i) => {
          const x = 40 + i * 80
          const h = (dim / inputDim) * maxH
          const y = (maxH - h) / 2 + 10
          const color = i === 2 ? '#8b5cf6' : '#a78bfa'
          return (
            <g key={i}>
              <rect x={x} y={y} width={30} height={h} rx={4} fill={color} opacity={0.8} />
              <text x={x + 15} y={maxH + 25} textAnchor="middle" className="text-xs fill-gray-500">{dim}</text>
              {i < layers.length - 1 && <line x1={x + 30} y1={maxH / 2 + 10} x2={x + 80} y2={maxH / 2 + 10} stroke="#d1d5db" strokeWidth={1} />}
            </g>
          )
        })}
        <text x={40 + 15} y={maxH + 25} textAnchor="middle" className="text-[10px] fill-gray-400">input</text>
        <text x={40 + 2 * 80 + 15} y={maxH + 25} textAnchor="middle" className="text-[10px] fill-violet-500 font-bold">z</text>
      </svg>
    </div>
  )
}

export default function Autoencoder() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        An autoencoder learns to compress input data into a lower-dimensional latent representation and
        then reconstruct it. The bottleneck forces the network to discover meaningful structure in the data.
      </p>

      <DefinitionBlock title="Autoencoder">
        <p>An autoencoder consists of an encoder <InlineMath math="f_\theta" /> and decoder <InlineMath math="g_\phi" />:</p>
        <BlockMath math="\mathbf{z} = f_\theta(\mathbf{x}), \quad \hat{\mathbf{x}} = g_\phi(\mathbf{z})" />
        <p className="mt-2">Training minimizes reconstruction loss:</p>
        <BlockMath math="\mathcal{L}(\theta, \phi) = \| \mathbf{x} - g_\phi(f_\theta(\mathbf{x})) \|^2" />
      </DefinitionBlock>

      <BottleneckVisualizer />

      <ExampleBlock title="Dimensionality Reduction vs PCA">
        <p>
          A single-layer linear autoencoder with MSE loss learns the same subspace as PCA.
          However, deep nonlinear autoencoders capture manifolds that PCA cannot:
        </p>
        <BlockMath math="\text{PCA: } \mathbf{z} = W^\top \mathbf{x}, \quad \text{AE: } \mathbf{z} = f_\theta(\mathbf{x})" />
      </ExampleBlock>

      <PythonCode
        title="Simple Autoencoder in PyTorch"
        code={`import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),  # pixel values in [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

model = Autoencoder(latent_dim=16)
x = torch.randn(8, 784)
x_hat, z = model(x)
print(f"Input: {x.shape}, Latent: {z.shape}, Recon: {x_hat.shape}")
loss = nn.MSELoss()(x_hat, torch.sigmoid(x))
print(f"Reconstruction loss: {loss.item():.4f}")`}
      />

      <NoteBlock type="note" title="Denoising Autoencoders">
        <p>
          A <strong>denoising autoencoder</strong> (DAE) corrupts the input with noise and trains the network
          to reconstruct the clean version. This prevents the autoencoder from learning the identity function
          and encourages robust feature extraction: <InlineMath math="\mathcal{L} = \|\mathbf{x} - g_\phi(f_\theta(\tilde{\mathbf{x}}))\|^2" />.
        </p>
      </NoteBlock>

      <NoteBlock type="note" title="Applications">
        <p>
          Autoencoders power anomaly detection (high reconstruction error signals anomalies),
          data compression, feature learning for downstream tasks, and serve as building blocks
          for variational autoencoders and diffusion models.
        </p>
      </NoteBlock>
    </div>
  )
}

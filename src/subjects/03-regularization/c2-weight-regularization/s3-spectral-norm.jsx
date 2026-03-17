import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function PowerIterationDemo() {
  const [iterations, setIterations] = useState(1)
  const W = [[2, 1], [1, 3]]

  const powerIterate = (n) => {
    let u = [1 / Math.sqrt(2), 1 / Math.sqrt(2)]
    let sigma = 0
    for (let i = 0; i < n; i++) {
      const v = [W[0][0] * u[0] + W[0][1] * u[1], W[1][0] * u[0] + W[1][1] * u[1]]
      sigma = Math.sqrt(v[0] * v[0] + v[1] * v[1])
      u = [v[0] / sigma, v[1] / sigma]
    }
    return { sigma, u }
  }

  const { sigma, u } = powerIterate(iterations)
  const trueSigma = 3.618

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Power Iteration for Spectral Norm</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Iterations: {iterations}
        <input type="range" min={1} max={20} step={1} value={iterations} onChange={e => setIterations(parseInt(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <div className="grid grid-cols-2 gap-4 text-sm text-gray-700 dark:text-gray-300">
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3">
          <p className="font-semibold text-violet-700 dark:text-violet-300">Estimated <InlineMath math="\sigma_1" /></p>
          <p className="text-2xl font-mono">{sigma.toFixed(4)}</p>
          <p className="text-xs text-gray-500">True value: {trueSigma.toFixed(3)}</p>
        </div>
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3">
          <p className="font-semibold text-violet-700 dark:text-violet-300">Top Singular Vector</p>
          <p className="text-lg font-mono">[{u[0].toFixed(4)}, {u[1].toFixed(4)}]</p>
          <p className="text-xs text-gray-500">Error: {Math.abs(sigma - trueSigma).toFixed(6)}</p>
        </div>
      </div>
    </div>
  )
}

export default function SpectralNorm() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Spectral normalization constrains the Lipschitz constant of each layer by normalizing
        weight matrices by their largest singular value. This is critical for training stable
        GANs and controlling function smoothness.
      </p>

      <DefinitionBlock title="Spectral Norm">
        <BlockMath math="\sigma(W) = \max_{\|h\|_2 \leq 1} \|Wh\|_2 = \sigma_1(W)" />
        <p className="mt-2">
          The spectral norm of a matrix is its largest singular value, which equals
          the maximum factor by which the matrix can stretch a vector.
        </p>
      </DefinitionBlock>

      <TheoremBlock title="Lipschitz Constraint via Spectral Normalization" id="spectral-lipschitz">
        <p>
          Spectral normalization replaces <InlineMath math="W" /> with <InlineMath math="\bar{W}" />:
        </p>
        <BlockMath math="\bar{W} = \frac{W}{\sigma(W)}" />
        <p className="mt-2">
          For a network <InlineMath math="f = f_L \circ \cdots \circ f_1" /> with each layer spectrally
          normalized, the global Lipschitz constant is bounded:
        </p>
        <BlockMath math="\|f(x) - f(y)\|_2 \leq \prod_{l=1}^L \sigma(\bar{W}_l) = 1" />
      </TheoremBlock>

      <PowerIterationDemo />

      <ExampleBlock title="Why One Step Suffices">
        <p>
          In practice, a single power iteration step per training step is sufficient because
          the weight matrix changes slowly between updates. The singular vector estimate from
          the previous step is a warm start, converging quickly to the true value.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Spectral Normalization in PyTorch"
        code={`import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# Apply spectral normalization to a layer
layer = nn.Linear(64, 32)
sn_layer = spectral_norm(layer)

# Check that spectral norm is approximately 1
x = torch.randn(16, 64)
W = sn_layer.weight
U, S, V = torch.linalg.svd(W)
print(f"Largest singular value: {S[0].item():.4f}")  # ~1.0

# GAN discriminator with spectral normalization
class SNDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(784, 256)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 1)),
        )
    def forward(self, x):
        return self.net(x)

disc = SNDiscriminator()
print(f"Discriminator params: {sum(p.numel() for p in disc.parameters())}")`}
      />

      <NoteBlock type="note" title="Beyond GANs">
        <p>
          Spectral normalization is also used in diffusion models, contrastive learning,
          and any setting where controlling the Lipschitz constant of a network is desirable.
          It can be combined with other regularization techniques like dropout and weight decay.
        </p>
      </NoteBlock>
    </div>
  )
}

import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function CouplingLayerViz() {
  const [split, setSplit] = useState(4)
  const dim = 8

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Affine Coupling Layer</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Split position: {split}/{dim}
        <input type="range" min={1} max={dim - 1} step={1} value={split} onChange={e => setSplit(parseInt(e.target.value))} className="w-32 accent-violet-500" />
      </label>
      <div className="flex gap-1 justify-center mb-2">
        {Array.from({ length: dim }, (_, i) => (
          <div key={i} className={`w-8 h-8 rounded flex items-center justify-center text-xs text-white font-mono ${i < split ? 'bg-violet-500' : 'bg-orange-400'}`}>
            z{i + 1}
          </div>
        ))}
      </div>
      <div className="text-center text-xs text-gray-500">
        <span className="text-violet-600">Identity path ({split}d)</span>
        {' | '}
        <span className="text-orange-600">Transformed path ({dim - split}d) via s,t networks</span>
      </div>
      <p className="text-xs text-gray-400 text-center mt-1">
        Jacobian is triangular with diagonal [1,...,1, exp(s_{'{'}d+1{'}'}), ..., exp(s_D)]
      </p>
    </div>
  )
}

export default function CouplingFlows() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Coupling layers split the input and transform one part conditioned on the other, yielding
        triangular Jacobians with cheap determinants. RealNVP and Glow build powerful flows from
        these simple building blocks.
      </p>

      <DefinitionBlock title="Affine Coupling Layer (RealNVP)">
        <p>Split input <InlineMath math="\mathbf{z} = [\mathbf{z}_{1:d}, \mathbf{z}_{d+1:D}]" />:</p>
        <BlockMath math="\mathbf{y}_{1:d} = \mathbf{z}_{1:d}" />
        <BlockMath math="\mathbf{y}_{d+1:D} = \mathbf{z}_{d+1:D} \odot \exp(s(\mathbf{z}_{1:d})) + t(\mathbf{z}_{1:d})" />
        <p className="mt-2">
          where <InlineMath math="s, t" /> are arbitrary neural networks. The Jacobian determinant is simply:
        </p>
        <BlockMath math="\log|\det J| = \sum_{j=d+1}^{D} s_j(\mathbf{z}_{1:d})" />
      </DefinitionBlock>

      <CouplingLayerViz />

      <ExampleBlock title="Glow: Generative Flow with 1x1 Convolutions">
        <p>
          Glow extends RealNVP with three innovations per step: (1) actnorm (data-dependent initialization),
          (2) invertible 1x1 convolution for channel permutation (replacing fixed shuffling), and
          (3) affine coupling layers. The 1x1 conv has Jacobian:
        </p>
        <BlockMath math="\log|\det J| = h \cdot w \cdot \log|\det \mathbf{W}|" />
        <p className="mt-1">where <InlineMath math="h, w" /> are spatial dimensions and <InlineMath math="\mathbf{W}" /> is the weight matrix.</p>
      </ExampleBlock>

      <PythonCode
        title="RealNVP Coupling Layer"
        code={`import torch
import torch.nn as nn

class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        half = dim // 2
        self.net = nn.Sequential(
            nn.Linear(half, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, (dim - half) * 2),  # s and t
        )
        self.half = half

    def forward(self, z):
        z1, z2 = z[:, :self.half], z[:, self.half:]
        st = self.net(z1)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s)  # clamp scale for stability
        y2 = z2 * torch.exp(s) + t
        y = torch.cat([z1, y2], dim=-1)
        log_det = s.sum(dim=-1)
        return y, log_det

    def inverse(self, y):
        y1, y2 = y[:, :self.half], y[:, self.half:]
        st = self.net(y1)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s)
        z2 = (y2 - t) * torch.exp(-s)
        return torch.cat([y1, z2], dim=-1)

layer = AffineCoupling(dim=16)
z = torch.randn(8, 16)
y, log_det = layer(z)
z_recon = layer.inverse(y)
print(f"Reconstruction error: {(z - z_recon).abs().max():.2e}")
print(f"Log-det Jacobian: {log_det[:3].tolist()}")`}
      />

      <WarningBlock title="Alternating Splits Are Essential">
        <p>
          A single coupling layer leaves half the dimensions unchanged. Flows must alternate which
          dimensions are identity vs transformed. Without this, the model cannot learn arbitrary
          distributions — the unchanged dimensions remain exactly Gaussian.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Masked Autoregressive Flows (MAF)">
        <p>
          MAF uses autoregressive conditioning: each <InlineMath math="x_i" /> depends on all
          previous <InlineMath math="x_{'{<i}'}" />. This is more expressive than coupling layers
          but sampling requires <InlineMath math="D" /> sequential passes. Inverse Autoregressive
          Flow (IAF) reverses this trade-off: fast sampling, slow density evaluation.
        </p>
      </NoteBlock>
    </div>
  )
}

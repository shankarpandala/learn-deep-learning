import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function CollapseViz() {
  const [method, setMethod] = useState('none')
  const methods = {
    none: { spread: 5, label: 'No prevention', desc: 'All representations collapse to a single point.' },
    negatives: { spread: 60, label: 'Negative pairs', desc: 'Contrastive loss pushes negatives apart.' },
    momentum: { spread: 55, label: 'Momentum encoder', desc: 'Slow-moving target prevents rapid collapse.' },
    variance: { spread: 58, label: 'Variance regularization', desc: 'Explicit loss term maintains variance.' },
  }
  const m = methods[method]

  const points = Array.from({ length: 20 }, (_, i) => {
    const angle = (i / 20) * Math.PI * 2
    const r = m.spread * (0.5 + 0.5 * Math.sin(i * 1.7))
    return { x: 150 + r * Math.cos(angle), y: 75 + r * Math.sin(angle) * 0.6 }
  })

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-2 text-base font-bold text-gray-800 dark:text-gray-200">Representational Collapse</h3>
      <div className="flex gap-2 mb-3 flex-wrap">
        {Object.entries(methods).map(([key, v]) => (
          <button key={key} onClick={() => setMethod(key)}
            className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${method === key ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400'}`}>
            {v.label}
          </button>
        ))}
      </div>
      <svg width={300} height={150} className="mx-auto block">
        {points.map((p, i) => (
          <circle key={i} cx={p.x} cy={p.y} r={4} fill="#8b5cf6" opacity={0.7} />
        ))}
      </svg>
      <p className="text-xs text-gray-500 text-center mt-1">{m.desc}</p>
    </div>
  )
}

export default function CollapsePrevention() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The central challenge in self-supervised learning is representational collapse: the model
        learns to map all inputs to the same representation, achieving zero loss trivially. Multiple
        strategies have been developed to prevent this failure mode.
      </p>

      <DefinitionBlock title="Representational Collapse">
        <p>Collapse occurs when the encoder produces constant or low-rank representations:</p>
        <BlockMath math="f_\theta(\mathbf{x}) \approx \mathbf{c} \quad \forall\, \mathbf{x} \in \mathcal{X}" />
        <p className="mt-2">
          This is a trivial solution to many self-supervised objectives. For instance, with a similarity
          loss <InlineMath math="\mathcal{L} = -\text{sim}(f(\mathbf{x}), f(\mathbf{x}'))" />, constant output
          gives perfect similarity with zero learning.
        </p>
      </DefinitionBlock>

      <CollapseViz />

      <WarningBlock title="Dimensional Collapse">
        <p>
          Even without full collapse, <strong>dimensional collapse</strong> can occur: representations
          occupy a low-dimensional subspace of the embedding space. The effective rank of the
          representation matrix drops, wasting capacity. This is harder to detect than full collapse.
        </p>
      </WarningBlock>

      <ExampleBlock title="Four Strategies to Prevent Collapse">
        <p className="space-y-2">
          <strong>1. Contrastive (negative pairs):</strong> Push apart representations of different images.
          SimCLR, MoCo use large sets of negatives.<br/>
          <strong>2. Momentum encoder:</strong> BYOL, MoCo use a slowly-updated target network, preventing
          the representations from changing too rapidly.<br/>
          <strong>3. Variance/covariance regularization:</strong> VICReg explicitly penalizes low variance
          and high covariance between embedding dimensions.<br/>
          <strong>4. Centering + sharpening:</strong> DINO centers the teacher output and sharpens it,
          preventing any single dimension from dominating.
        </p>
      </ExampleBlock>

      <PythonCode
        title="VICReg Loss: Variance-Invariance-Covariance"
        code={`import torch
import torch.nn.functional as F

def vicreg_loss(z1, z2, lam=25.0, mu=25.0, nu=1.0):
    """VICReg: Variance-Invariance-Covariance Regularization."""
    B, D = z1.shape

    # Invariance: MSE between positive pairs
    inv_loss = F.mse_loss(z1, z2)

    # Variance: std of each dimension should be >= 1
    std_z1 = z1.std(dim=0)
    std_z2 = z2.std(dim=0)
    var_loss = (F.relu(1 - std_z1).mean() + F.relu(1 - std_z2).mean())

    # Covariance: off-diagonal elements should be 0
    z1_centered = z1 - z1.mean(dim=0)
    z2_centered = z2 - z2.mean(dim=0)
    cov1 = (z1_centered.T @ z1_centered) / (B - 1)
    cov2 = (z2_centered.T @ z2_centered) / (B - 1)
    # Zero out diagonal, penalize off-diagonal
    mask = ~torch.eye(D, dtype=bool)
    cov_loss = (cov1[mask].pow(2).mean() + cov2[mask].pow(2).mean())

    return lam * inv_loss + mu * var_loss + nu * cov_loss

z1 = torch.randn(64, 128)
z2 = z1 + 0.1 * torch.randn_like(z1)  # positive pairs
loss = vicreg_loss(z1, z2)
print(f"VICReg loss: {loss.item():.3f}")
print(f"Embedding std: {z1.std(dim=0).mean():.3f} (target >= 1.0)")`}
      />

      <NoteBlock type="note" title="Barlow Twins: Redundancy Reduction">
        <p>
          Barlow Twins takes a complementary approach: the cross-correlation matrix between two
          augmented views should be the identity matrix. This simultaneously prevents collapse
          (diagonal elements = 1) and redundancy (off-diagonal = 0):
          <InlineMath math="\mathcal{L} = \sum_i (1 - C_{ii})^2 + \lambda \sum_{i \neq j} C_{ij}^2" />.
        </p>
      </NoteBlock>
    </div>
  )
}

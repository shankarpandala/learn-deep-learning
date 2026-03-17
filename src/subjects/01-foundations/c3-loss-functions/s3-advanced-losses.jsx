import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function EmbeddingViz() {
  const [margin, setMargin] = useState(1.0)
  const [anchorPos, setAnchorPos] = useState({ x: 200, y: 130 })
  const W = 420, H = 260

  const anchor = anchorPos
  const positive = { x: anchor.x + 40, y: anchor.y - 30 }
  const negative = { x: anchor.x + 120, y: anchor.y + 50 }
  const dPos = Math.sqrt((positive.x - anchor.x) ** 2 + (positive.y - anchor.y) ** 2)
  const dNeg = Math.sqrt((negative.x - anchor.x) ** 2 + (negative.y - anchor.y) ** 2)
  const tripletLoss = Math.max(0, dPos - dNeg + margin * 50)
  const marginRadius = dPos + margin * 50

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Triplet Loss Embedding Space</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          margin = {margin.toFixed(1)}
          <input type="range" min={0.1} max={3.0} step={0.1} value={margin} onChange={e => setMargin(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <span className="text-sm text-gray-500 dark:text-gray-400">
          d(a,p)={dPos.toFixed(0)} | d(a,n)={dNeg.toFixed(0)} | loss={tripletLoss.toFixed(0)}
        </span>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <circle cx={anchor.x} cy={anchor.y} r={marginRadius} fill="none" stroke="#8b5cf6" strokeWidth={1} strokeDasharray="4,4" opacity={0.4} />
        <line x1={anchor.x} y1={anchor.y} x2={positive.x} y2={positive.y} stroke="#10b981" strokeWidth={2} />
        <line x1={anchor.x} y1={anchor.y} x2={negative.x} y2={negative.y} stroke="#ef4444" strokeWidth={2} />
        <circle cx={anchor.x} cy={anchor.y} r={8} fill="#8b5cf6" />
        <circle cx={positive.x} cy={positive.y} r={8} fill="#10b981" />
        <circle cx={negative.x} cy={negative.y} r={8} fill="#ef4444" />
        <text x={anchor.x} y={anchor.y - 14} fontSize={11} fill="#8b5cf6" textAnchor="middle" fontWeight="bold">Anchor</text>
        <text x={positive.x} y={positive.y - 14} fontSize={11} fill="#10b981" textAnchor="middle" fontWeight="bold">Positive</text>
        <text x={negative.x} y={negative.y - 14} fontSize={11} fill="#ef4444" textAnchor="middle" fontWeight="bold">Negative</text>
        <text x={W / 2} y={H - 8} fontSize={10} fill="#9ca3af" textAnchor="middle">Dashed circle = margin boundary from anchor through positive</text>
      </svg>
    </div>
  )
}

export default function AdvancedLosses() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Contrastive and triplet losses learn embedding spaces where similar items are close
        and dissimilar items are far apart. These losses are foundational for metric learning,
        face recognition, and modern self-supervised learning.
      </p>

      <DefinitionBlock title="Contrastive Loss (Hadsell et al., 2006)">
        <BlockMath math="\mathcal{L} = \frac{1}{2}y \cdot d^2 + \frac{1}{2}(1-y) \cdot \max(0, m - d)^2" />
        <p className="mt-2">
          For a pair of embeddings with distance <InlineMath math="d = \|f(x_1) - f(x_2)\|_2" />,
          label <InlineMath math="y=1" /> for similar pairs and <InlineMath math="y=0" /> for
          dissimilar. The margin <InlineMath math="m > 0" /> defines the minimum separation
          for dissimilar pairs.
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="Triplet Loss (Schroff et al., 2015)">
        <BlockMath math="\mathcal{L} = \max\bigl(0,\; \|f(a) - f(p)\|_2^2 - \|f(a) - f(n)\|_2^2 + \alpha\bigr)" />
        <p className="mt-2">
          Given an anchor <InlineMath math="a" />, a positive <InlineMath math="p" /> (same class),
          and a negative <InlineMath math="n" /> (different class), we push the negative further
          away than the positive by at least margin <InlineMath math="\alpha" />.
        </p>
      </DefinitionBlock>

      <EmbeddingViz />

      <TheoremBlock title="Hard Negative Mining" id="hard-negative">
        <p>
          Random triplet selection leads to many trivial triplets where the loss is already zero.
          Hard negative mining selects the hardest negative for each anchor:
        </p>
        <BlockMath math="n^* = \arg\min_{n:\, y_n \neq y_a} \|f(a) - f(n)\|_2" />
        <p>
          Semi-hard negatives satisfy <InlineMath math="\|f(a) - f(p)\|_2 < \|f(a) - f(n)\|_2 < \|f(a) - f(p)\|_2 + \alpha" /> and
          often yield more stable training than pure hard negatives.
        </p>
      </TheoremBlock>

      <DefinitionBlock title="InfoNCE Loss (van den Oord et al., 2018)">
        <BlockMath math="\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}" />
        <p className="mt-2">
          Generalizes contrastive learning to multiple negatives. The temperature <InlineMath math="\tau" /> controls
          the sharpness of the distribution. This loss is used in SimCLR, MoCo, and CLIP.
          It is equivalent to a <InlineMath math="(2N-1)" />-way cross-entropy over cosine similarities.
        </p>
      </DefinitionBlock>

      <WarningBlock title="Collapse in Contrastive Learning">
        <p>
          Without careful design, the encoder can collapse to a constant embedding where all
          outputs are identical (trivially minimizing positive distances). Strategies to prevent
          collapse include using a momentum encoder (MoCo), stop-gradient (BYOL/SimSiam),
          or large batch sizes with diverse negatives (SimCLR).
        </p>
      </WarningBlock>

      <PythonCode
        title="Triplet & InfoNCE Loss in PyTorch"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

# Triplet loss (built-in)
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
anchor = torch.randn(32, 128)
positive = anchor + 0.1 * torch.randn(32, 128)
negative = torch.randn(32, 128)
loss_t = triplet_loss(anchor, positive, negative)
print(f"Triplet loss: {loss_t:.4f}")

# InfoNCE loss (simplified SimCLR-style)
def info_nce_loss(z_i, z_j, temperature=0.07):
    """z_i, z_j: [B, D] augmented pair embeddings."""
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    B = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)            # [2B, D]
    sim = z @ z.T / temperature                  # [2B, 2B]
    # Mask out self-similarity
    mask = ~torch.eye(2 * B, dtype=bool, device=z.device)
    sim = sim.masked_select(mask).view(2 * B, -1)
    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([torch.arange(B) + B - 1,
                        torch.arange(B)], dim=0)
    return F.cross_entropy(sim, labels)

z1 = torch.randn(64, 128)
z2 = z1 + 0.1 * torch.randn(64, 128)
loss_nce = info_nce_loss(z1, z2)
print(f"InfoNCE loss: {loss_nce:.4f}")`}
      />

      <ExampleBlock title="Face Verification with Triplet Loss">
        <p>
          In FaceNet, the embedding space is trained so that the L2 distance between faces
          of the same person is small. At inference, a simple threshold on distance determines
          identity:
        </p>
        <BlockMath math="\text{same person} \iff \|f(x_1) - f(x_2)\|_2 < \theta" />
        <p>
          Typical embedding dimension is 128, and the threshold <InlineMath math="\theta" /> is
          tuned on a validation set for the desired false-accept / false-reject trade-off.
        </p>
      </ExampleBlock>

      <NoteBlock type="tip" title="When to Use Which Loss">
        <p>
          <strong>Contrastive</strong>: Simple pair-based tasks (signature verification).
          <strong> Triplet</strong>: Fine-grained recognition (faces, products) with online mining.
          <strong> InfoNCE</strong>: Self-supervised pretraining with large batches.
          For modern representation learning, InfoNCE-based approaches (SimCLR, CLIP) have
          largely superseded pairwise and triplet formulations.
        </p>
      </NoteBlock>
    </div>
  )
}

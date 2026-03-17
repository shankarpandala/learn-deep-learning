import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function PredictiveTaskViz() {
  const [maskRatio, setMaskRatio] = useState(0.5)
  const gridSize = 8
  const totalCells = gridSize * gridSize
  const maskedCount = Math.round(totalCells * maskRatio)

  const [seed] = useState(() => Array.from({ length: totalCells }, () => Math.random()))
  const masked = seed.map((v, i) => {
    const threshold = seed.slice().sort()[maskedCount - 1] || 0
    return v <= threshold
  })

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Predictive Masking</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Mask ratio: {(maskRatio * 100).toFixed(0)}%
        <input type="range" min={0.1} max={0.9} step={0.05} value={maskRatio} onChange={e => setMaskRatio(parseFloat(e.target.value))} className="w-40 accent-violet-500" />
        <span className="text-xs">({maskedCount}/{totalCells} patches masked)</span>
      </label>
      <div className="flex justify-center">
        <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${gridSize}, 1fr)` }}>
          {Array.from({ length: totalCells }, (_, i) => (
            <div key={i} className={`w-5 h-5 rounded-sm ${masked[i] ? 'bg-gray-300 dark:bg-gray-600' : 'bg-violet-400'}`} />
          ))}
        </div>
      </div>
      <div className="flex justify-center gap-4 text-xs mt-2">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 bg-violet-400 rounded-sm" /> Visible</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 bg-gray-300 rounded-sm" /> Masked (predict)</span>
      </div>
    </div>
  )
}

export default function PredictiveLearning() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Predictive self-supervision trains models to predict missing or future parts of the input.
        This paradigm, inspired by language model pre-training (predicting next tokens), has become
        the dominant approach in self-supervised visual representation learning.
      </p>

      <DefinitionBlock title="Predictive Self-Supervised Learning">
        <p>Given input <InlineMath math="\mathbf{x}" />, partition into visible context <InlineMath math="\mathbf{x}_v" /> and target <InlineMath math="\mathbf{x}_t" />:</p>
        <BlockMath math="\mathcal{L} = \mathbb{E}\left[d\left(g_\phi(f_\theta(\mathbf{x}_v)),\; \mathbf{x}_t\right)\right]" />
        <p className="mt-2">
          The distance <InlineMath math="d" /> can operate in pixel space (MSE), token space (cross-entropy),
          or representation space (feature regression).
        </p>
      </DefinitionBlock>

      <PredictiveTaskViz />

      <TheoremBlock title="Information-Theoretic Perspective" id="info-theory">
        <p>
          Predictive learning maximizes a lower bound on the mutual information between visible
          and target parts:
        </p>
        <BlockMath math="I(\mathbf{x}_v; \mathbf{x}_t) \geq -\mathcal{L}_{\text{pred}} + H(\mathbf{x}_t)" />
        <p className="mt-2">
          Better prediction (lower loss) implies more information captured about the data structure.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Context Prediction (Doersch et al., 2015)">
        <p>
          One of the earliest visual predictive tasks: given a center patch, predict the relative
          position (1 of 8 neighbors) of a second patch. This teaches the network about spatial
          layouts and object parts — for example, an eye patch is typically above a mouth patch.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Relative Position Prediction"
        code={`import torch
import torch.nn as nn

class RelativePositionNet(nn.Module):
    def __init__(self, backbone_dim=512):
        super().__init__()
        # Two patch embeddings are concatenated
        self.position_head = nn.Sequential(
            nn.Linear(backbone_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 8),  # 8 relative positions
        )

    def forward(self, patch_center, patch_neighbor):
        return self.position_head(torch.cat([patch_center, patch_neighbor], dim=-1))

def extract_patch_pairs(images, patch_size=64, gap=32):
    """Extract center patch and one of 8 neighboring patches."""
    B, C, H, W = images.shape
    # Center patch coordinates
    cy = H // 2 - patch_size // 2
    cx = W // 2 - patch_size // 2
    center = images[:, :, cy:cy+patch_size, cx:cx+patch_size]

    # Random neighbor direction (0-7: TL, T, TR, L, R, BL, B, BR)
    direction = torch.randint(0, 8, (B,))
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    neighbors, labels = [], direction
    for i in range(B):
        dy, dx = offsets[direction[i]]
        ny = cy + dy * (patch_size + gap)
        nx = cx + dx * (patch_size + gap)
        ny, nx = max(0, ny), max(0, nx)
        neighbors.append(images[i, :, ny:ny+patch_size, nx:nx+patch_size])

    return center, torch.stack(neighbors), labels

print("Relative position: 8-class classification")
print("Forces learning spatial relationships between object parts")`}
      />

      <NoteBlock type="note" title="From Pixel Prediction to Feature Prediction">
        <p>
          Predicting raw pixels encourages learning low-level statistics (textures, edges) rather than
          high-level semantics. Modern approaches predict in <em>feature space</em> instead: MAE predicts
          normalized pixel patches, BEiT predicts discrete visual tokens, and data2vec predicts
          teacher network representations. This shift dramatically improves downstream performance.
        </p>
      </NoteBlock>
    </div>
  )
}

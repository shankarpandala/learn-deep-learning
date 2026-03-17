import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function MAEMaskViz() {
  const [maskRatio, setMaskRatio] = useState(0.75)
  const gridSize = 14
  const totalPatches = gridSize * gridSize
  const maskedCount = Math.round(totalPatches * maskRatio)

  const [patchOrder] = useState(() => {
    const indices = Array.from({ length: totalPatches }, (_, i) => i)
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]]
    }
    return indices
  })

  const maskedSet = new Set(patchOrder.slice(0, maskedCount))

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">MAE: Masked Image Patches</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Mask ratio: {(maskRatio * 100).toFixed(0)}%
        <input type="range" min={0.5} max={0.9} step={0.05} value={maskRatio} onChange={e => setMaskRatio(parseFloat(e.target.value))} className="w-40 accent-violet-500" />
        <span className="text-xs">({totalPatches - maskedCount} visible / {totalPatches} total)</span>
      </label>
      <div className="flex justify-center">
        <div className="grid gap-px" style={{ gridTemplateColumns: `repeat(${gridSize}, 1fr)` }}>
          {Array.from({ length: totalPatches }, (_, i) => (
            <div key={i} className={`w-3.5 h-3.5 ${maskedSet.has(i) ? 'bg-gray-200 dark:bg-gray-700' : 'bg-violet-400'}`} />
          ))}
        </div>
      </div>
      <p className="text-xs text-gray-500 text-center mt-2">
        Only {((1 - maskRatio) * 100).toFixed(0)}% of patches go through the encoder (huge compute savings)
      </p>
    </div>
  )
}

export default function MAE() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Masked Autoencoders (MAE) adapt the masked language modeling paradigm (BERT) to vision.
        By masking a very high proportion of image patches (75%) and reconstructing them, MAE learns
        powerful visual representations with remarkable compute efficiency.
      </p>

      <DefinitionBlock title="MAE Architecture">
        <p>MAE consists of an asymmetric encoder-decoder:</p>
        <ul className="list-disc ml-5 mt-2 space-y-1">
          <li><strong>Encoder</strong> (ViT): processes only <em>visible</em> patches (~25%), making it very fast</li>
          <li><strong>Decoder</strong> (lightweight): processes full set of tokens (visible + mask tokens)</li>
          <li><strong>Reconstruction target</strong>: per-patch normalized pixel values</li>
        </ul>
        <BlockMath math="\mathcal{L}_{\text{MAE}} = \frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} \| \hat{\mathbf{p}}_i - \mathbf{p}_i \|^2" />
        <p className="mt-1">Loss computed only on masked patches <InlineMath math="\mathcal{M}" />.</p>
      </DefinitionBlock>

      <MAEMaskViz />

      <TheoremBlock title="Why 75% Masking Works" id="high-masking">
        <p>
          Images have high spatial redundancy: neighboring patches are highly correlated.
          Low masking ratios allow the model to interpolate from nearby visible patches
          without learning semantics. At 75%, the task becomes truly challenging:
        </p>
        <BlockMath math="I(\mathbf{x}_{\text{visible}}; \mathbf{x}_{\text{masked}}) \ll I(\mathbf{x}; \mathbf{x})" />
        <p className="mt-2">
          The visible patches provide insufficient local information, forcing holistic understanding.
          This contrasts with BERT's 15% masking, since text tokens carry more information per token.
        </p>
      </TheoremBlock>

      <PythonCode
        title="MAE Core: Masking and Reconstruction"
        code={`import torch
import torch.nn as nn

class MAEEncoder(nn.Module):
    def __init__(self, num_patches=196, embed_dim=768, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_embed = nn.Linear(768, embed_dim)  # patch projection
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        # Transformer blocks would go here

    def random_masking(self, x):
        B, N, D = x.shape
        keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_keep = ids_shuffle[:, :keep]

        # Keep only visible patches
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        return x_visible, ids_keep, ids_shuffle

    def forward(self, x):
        # x: (B, num_patches, patch_dim)
        x = self.patch_embed(x) + self.pos_embed
        x_visible, ids_keep, ids_shuffle = self.random_masking(x)
        # Pass x_visible through transformer (not shown)
        return x_visible, ids_keep, ids_shuffle

# Efficiency: encoder processes only 25% of patches
encoder = MAEEncoder(num_patches=196, mask_ratio=0.75)
patches = torch.randn(4, 196, 768)
visible, keep_ids, _ = encoder(patches)
print(f"Input: {patches.shape[1]} patches")
print(f"Encoder processes: {visible.shape[1]} patches ({visible.shape[1]/196*100:.0f}%)")
print(f"3-4x faster than processing all patches!")`}
      />

      <ExampleBlock title="MAE Pre-training Results">
        <p>
          MAE pre-trained ViT-Large achieves 85.9% top-1 on ImageNet with fine-tuning, surpassing
          supervised pre-training. The decoder is discarded after pre-training, and only the
          encoder is used for downstream tasks. MAE is particularly effective for larger models
          (ViT-Huge: 86.9%) where labeled data is insufficient.
        </p>
      </ExampleBlock>

      <NoteBlock type="note" title="Pixel vs Feature Reconstruction">
        <p>
          MAE reconstructs normalized pixels, which works surprisingly well despite the concern that
          pixel prediction emphasizes low-level details. The high masking ratio forces semantic
          understanding regardless. BEiT (next section) reconstructs discrete visual tokens instead,
          providing a different inductive bias.
        </p>
      </NoteBlock>
    </div>
  )
}

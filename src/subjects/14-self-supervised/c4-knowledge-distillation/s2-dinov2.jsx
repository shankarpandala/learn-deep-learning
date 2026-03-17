import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ScalingViz() {
  const [modelSize, setModelSize] = useState(2)
  const models = [
    { name: 'ViT-S/14', params: 21, linear: 79.0, knn: 77.2 },
    { name: 'ViT-B/14', params: 86, linear: 82.1, knn: 80.1 },
    { name: 'ViT-L/14', params: 300, linear: 83.5, knn: 82.0 },
    { name: 'ViT-g/14', params: 1100, linear: 83.9, knn: 82.8 },
  ]
  const m = models[modelSize]
  const maxAcc = 84

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">DINOv2 Scaling Behavior</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Model: {m.name} ({m.params}M params)
        <input type="range" min={0} max={3} step={1} value={modelSize} onChange={e => setModelSize(parseInt(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <div className="flex gap-4 justify-center items-end h-28">
        <div className="flex flex-col items-center">
          <div className="w-14 bg-violet-500 rounded-t transition-all" style={{ height: `${(m.linear / maxAcc) * 90}px` }} />
          <span className="text-xs text-gray-500 mt-1">Linear</span>
          <span className="text-xs text-violet-600 font-semibold">{m.linear}%</span>
        </div>
        <div className="flex flex-col items-center">
          <div className="w-14 bg-violet-300 rounded-t transition-all" style={{ height: `${(m.knn / maxAcc) * 90}px` }} />
          <span className="text-xs text-gray-500 mt-1">k-NN</span>
          <span className="text-xs text-violet-600 font-semibold">{m.knn}%</span>
        </div>
      </div>
      <p className="text-xs text-gray-500 text-center mt-1">ImageNet top-1 accuracy (no fine-tuning)</p>
    </div>
  )
}

export default function DINOv2() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        DINOv2 scales self-supervised learning to produce all-purpose visual features that work
        across tasks without fine-tuning. By combining DINO self-distillation with iBOT masked
        modeling and careful data curation, DINOv2 creates foundation-level visual representations.
      </p>

      <DefinitionBlock title="DINOv2: Combined Objective">
        <p>DINOv2 combines two self-supervised losses:</p>
        <BlockMath math="\mathcal{L}_{\text{DINOv2}} = \mathcal{L}_{\text{DINO}}(\text{[CLS]}) + \mathcal{L}_{\text{iBOT}}(\text{patch tokens})" />
        <p className="mt-2">
          <InlineMath math="\mathcal{L}_{\text{DINO}}" />: Self-distillation on [CLS] token (global features).
          <InlineMath math="\mathcal{L}_{\text{iBOT}}" />: Masked image modeling on patch tokens (local features).
          The combination yields features strong for both image-level and dense prediction tasks.
        </p>
      </DefinitionBlock>

      <ScalingViz />

      <ExampleBlock title="Data Curation Pipeline">
        <p>
          DINOv2 curates a 142M image dataset (LVD-142M) through: (1) web crawling to collect
          candidate images, (2) deduplication using copy detection, (3) self-supervised retrieval
          to select images similar to curated datasets (ImageNet). This automated pipeline
          avoids manual annotation while ensuring data quality and diversity.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Using DINOv2 Features for Downstream Tasks"
        code={`import torch
import torch.nn as nn

# Load pre-trained DINOv2 (using torch.hub)
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

class DINOv2Wrapper:
    """Demonstrates DINOv2 feature extraction patterns."""
    def __init__(self, embed_dim=1024):
        self.embed_dim = embed_dim

    def extract_features(self, images):
        """Returns CLS token and patch tokens."""
        # In practice: features = model.forward_features(images)
        B = images.shape[0]
        cls_token = torch.randn(B, self.embed_dim)
        patch_tokens = torch.randn(B, 256, self.embed_dim)  # 16x16 patches
        return cls_token, patch_tokens

# Usage patterns (no fine-tuning needed!)
wrapper = DINOv2Wrapper()
images = torch.randn(4, 3, 224, 224)
cls_feat, patch_feat = wrapper.extract_features(images)

# 1. Image classification: linear probe on CLS token
classifier = nn.Linear(1024, 1000)
logits = classifier(cls_feat)

# 2. Semantic segmentation: linear probe on patch tokens
seg_head = nn.Linear(1024, 21)  # 21 classes
seg_map = seg_head(patch_feat)  # (B, 256, 21)

# 3. k-NN classification (no training at all!)
# Just compute cosine similarity to labeled reference features

print(f"CLS features: {cls_feat.shape} (for classification)")
print(f"Patch features: {patch_feat.shape} (for dense prediction)")
print("All from a single frozen backbone — no fine-tuning!")`}
      />

      <WarningBlock title="Compute Requirements">
        <p>
          DINOv2 ViT-g was trained on 142M images for 625K iterations on 140 A100 GPUs.
          Training from scratch is impractical for most researchers. However, the released
          pre-trained models serve as powerful frozen feature extractors for a wide range of tasks.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="DINOv2 as a Visual Foundation Model">
        <p>
          DINOv2 features rival or exceed supervised features (including CLIP) on many tasks
          without any task-specific training: depth estimation, semantic segmentation, image
          retrieval, and classification. This makes DINOv2 arguably the strongest general-purpose
          visual feature extractor as of its release.
        </p>
      </NoteBlock>
    </div>
  )
}

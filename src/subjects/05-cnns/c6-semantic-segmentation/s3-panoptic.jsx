import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function SegmentationTypes() {
  const [selected, setSelected] = useState('panoptic')
  const types = {
    semantic: {
      name: 'Semantic Segmentation',
      desc: 'Assigns a class label to every pixel. Does not distinguish between individual instances of the same class.',
      output: 'H x W label map',
      example: 'All "car" pixels get the same label, regardless of how many cars are present.',
    },
    instance: {
      name: 'Instance Segmentation',
      desc: 'Detects each object instance and produces a binary mask for each. Only for "thing" classes (countable objects).',
      output: 'N masks + N labels',
      example: 'Each car gets a unique mask and label. Background is not segmented.',
    },
    panoptic: {
      name: 'Panoptic Segmentation',
      desc: 'Unifies semantic and instance segmentation. Every pixel gets a class AND an instance ID.',
      output: 'H x W (class_id, instance_id)',
      example: 'Each car gets a unique instance, AND the sky, road get semantic labels.',
    },
  }
  const t = types[selected]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Segmentation Task Comparison</h3>
      <div className="flex gap-2 mb-4">
        {Object.entries(types).map(([key, val]) => (
          <button key={key} onClick={() => setSelected(key)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${selected === key ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="p-4 rounded-lg bg-violet-50 dark:bg-violet-900/20 space-y-2">
        <p className="font-bold text-violet-800 dark:text-violet-200">{t.name}</p>
        <p className="text-sm text-gray-700 dark:text-gray-300">{t.desc}</p>
        <p className="text-xs text-gray-500">Output format: <strong>{t.output}</strong></p>
        <p className="text-xs text-gray-500 italic">{t.example}</p>
      </div>
    </div>
  )
}

export default function PanopticSegmentation() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Panoptic segmentation unifies semantic segmentation ("stuff" classes like sky, road) and
        instance segmentation ("thing" classes like cars, people). Mask R-CNN handles instance
        segmentation, while panoptic methods extend this to cover every pixel.
      </p>

      <DefinitionBlock title="Mask R-CNN">
        <p>
          Mask R-CNN extends Faster R-CNN by adding a parallel mask prediction branch. For each
          detected object, it predicts a binary mask using a small FCN:
        </p>
        <BlockMath math="\mathcal{L} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{box}} + \mathcal{L}_{\text{mask}}" />
        <p className="mt-2">
          The mask branch outputs a <InlineMath math="C \times m \times m" /> mask for each RoI,
          where <InlineMath math="m = 28" /> typically. RoI Align is critical for spatial precision.
        </p>
      </DefinitionBlock>

      <SegmentationTypes />

      <TheoremBlock title="Panoptic Quality (PQ)" id="panoptic-quality">
        <p>The standard metric for panoptic segmentation decomposes into recognition and segmentation:</p>
        <BlockMath math="\text{PQ} = \underbrace{\frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}}_{\text{Recognition Quality (RQ)}} \times \underbrace{\frac{\sum_{(p,g) \in TP} \text{IoU}(p, g)}{|TP|}}_{\text{Segmentation Quality (SQ)}}" />
        <p className="mt-2">
          PQ = RQ x SQ, where RQ measures detection performance and SQ measures mask quality.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Stuff vs Things">
        <p>
          <strong>Things</strong>: countable objects (person, car, dog) requiring instance-level masks.{' '}
          <strong>Stuff</strong>: amorphous regions (sky, grass, road) requiring only class labels.
          Panoptic segmentation handles both in a unified framework, assigning every pixel a
          semantic class and, for thing classes, an instance ID.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Mask R-CNN with torchvision"
        code={`import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

# Load pretrained Mask R-CNN
model = maskrcnn_resnet50_fpn_v2(
    weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
model.eval()

# Inference
x = torch.randn(1, 3, 480, 640)
with torch.no_grad():
    preds = model(x)

# Each prediction contains boxes, labels, scores, and masks
print(f"Boxes: {preds[0]['boxes'].shape}")
print(f"Masks: {preds[0]['masks'].shape}")  # [N, 1, H, W]
print(f"Labels: {preds[0]['labels'][:5]}")

# Filter high-confidence masks
keep = preds[0]['scores'] > 0.5
masks = preds[0]['masks'][keep] > 0.5  # binarize
print(f"High-conf instances: {masks.shape[0]}")

# Combine into panoptic-style output
combined = torch.zeros(480, 640, dtype=torch.long)
for i, mask in enumerate(masks):
    combined[mask[0]] = i + 1  # instance IDs
print(f"Panoptic map: {combined.unique().shape[0]} segments")`}
      />

      <NoteBlock type="note" title="Modern Panoptic Architectures">
        <p>
          Panoptic FPN uses a shared FPN backbone with separate semantic and instance heads.
          MaskFormer and Mask2Former unify all segmentation tasks with a single mask
          classification approach using Transformer decoders, treating each segment (stuff or thing)
          as a binary mask predicted by a learned query.
        </p>
      </NoteBlock>
    </div>
  )
}

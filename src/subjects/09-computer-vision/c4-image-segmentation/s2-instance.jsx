import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function MaskDemo() {
  const [showMasks, setShowMasks] = useState(true)
  const [showBoxes, setShowBoxes] = useState(true)
  const W = 280, H = 180
  const instances = [
    { x: 30, y: 30, w: 70, h: 90, color: '#8b5cf6', label: 'Person 1' },
    { x: 120, y: 50, w: 60, h: 80, color: '#f97316', label: 'Person 2' },
    { x: 200, y: 60, w: 55, h: 70, color: '#22c55e', label: 'Dog' },
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Instance Segmentation Output</h3>
      <div className="flex gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          <input type="checkbox" checked={showMasks} onChange={e => setShowMasks(e.target.checked)} className="accent-violet-500" />
          Show masks
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          <input type="checkbox" checked={showBoxes} onChange={e => setShowBoxes(e.target.checked)} className="accent-violet-500" />
          Show boxes
        </label>
      </div>
      <svg width={W} height={H} className="mx-auto block bg-gray-50 dark:bg-gray-800 rounded">
        {instances.map((inst, i) => (
          <g key={i}>
            {showMasks && <ellipse cx={inst.x + inst.w / 2} cy={inst.y + inst.h / 2}
              rx={inst.w / 2.2} ry={inst.h / 2.2} fill={inst.color} opacity={0.3} />}
            {showBoxes && <rect x={inst.x} y={inst.y} width={inst.w} height={inst.h}
              fill="none" stroke={inst.color} strokeWidth={2} />}
            <text x={inst.x} y={inst.y - 4} fontSize={10} fill={inst.color} fontWeight="bold">{inst.label}</text>
          </g>
        ))}
      </svg>
    </div>
  )
}

export default function InstanceSeg() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Instance segmentation combines object detection with pixel-level segmentation,
        producing individual masks for each object instance. Mask R-CNN is the foundational approach.
      </p>

      <DefinitionBlock title="Instance vs Semantic Segmentation">
        <p>
          Semantic segmentation labels pixels by class. Instance segmentation additionally
          distinguishes individual objects of the same class:
        </p>
        <BlockMath math="\text{Output} = \{(c_k, m_k, s_k)\}_{k=1}^{K}" />
        <p className="mt-2">
          where <InlineMath math="c_k" /> is the class, <InlineMath math="m_k \in \{0,1\}^{H \times W}" /> is
          the binary mask, and <InlineMath math="s_k" /> is the confidence score for the <InlineMath math="k" />-th instance.
        </p>
      </DefinitionBlock>

      <MaskDemo />

      <TheoremBlock title="Mask R-CNN Architecture" id="mask-rcnn">
        <p>Mask R-CNN extends Faster R-CNN with a parallel mask prediction branch:</p>
        <BlockMath math="\mathcal{L} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{box}} + \mathcal{L}_{\text{mask}}" />
        <p className="mt-1">
          The mask loss is per-pixel binary cross-entropy applied only to the ground truth class:
        </p>
        <BlockMath math="\mathcal{L}_{\text{mask}} = -\frac{1}{m^2}\sum_{ij}\left[y_{ij}\log\hat{m}_{ij}^c + (1-y_{ij})\log(1-\hat{m}_{ij}^c)\right]" />
        <p className="mt-1">
          where <InlineMath math="m = 28" /> is the mask resolution and <InlineMath math="c" /> is the predicted class.
        </p>
      </TheoremBlock>

      <ExampleBlock title="RoIAlign vs RoIPool">
        <p>
          RoIPool introduces quantization errors by snapping to grid cells. RoIAlign uses
          bilinear interpolation at exact floating-point coordinates:
        </p>
        <BlockMath math="\text{RoIAlign}(x, y) = \sum_{ij} \max(0, 1 - |x - x_i|) \cdot \max(0, 1 - |y - y_j|) \cdot f_{ij}" />
        <p className="mt-1">
          This eliminates misalignment artifacts and improves mask AP by 1-3 points.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Mask R-CNN with Torchvision"
        code={`import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

# Load pretrained Mask R-CNN
model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')
model.eval()

# Inference
images = [torch.rand(3, 640, 480)]
with torch.no_grad():
    predictions = model(images)

# Each prediction contains:
boxes = predictions[0]['boxes']    # (N, 4) bounding boxes
labels = predictions[0]['labels']  # (N,) class labels
scores = predictions[0]['scores']  # (N,) confidence scores
masks = predictions[0]['masks']    # (N, 1, H, W) instance masks

# Filter by confidence
keep = scores > 0.7
final_masks = masks[keep] > 0.5  # Binary masks at threshold 0.5

# Panoptic segmentation: combine instance + semantic
def merge_to_panoptic(instance_masks, semantic_pred):
    """Merge instance and semantic for panoptic output."""
    panoptic = semantic_pred.clone()
    for i, mask in enumerate(instance_masks):
        panoptic[mask.squeeze()] = 1000 + i  # Unique instance ID
    return panoptic

print(f"Detected {keep.sum()} instances")
print(f"Mask shape: {final_masks.shape}")`}
      />

      <NoteBlock type="note" title="Panoptic Segmentation">
        <p>
          Panoptic segmentation unifies instance and semantic segmentation into a single task.
          Every pixel gets both a class label and an instance ID. "Things" (countable objects)
          get instance IDs while "stuff" (amorphous regions like sky, road) share a single ID
          per class. Modern approaches like Panoptic FPN and MaskFormer handle both in one model.
        </p>
      </NoteBlock>
    </div>
  )
}

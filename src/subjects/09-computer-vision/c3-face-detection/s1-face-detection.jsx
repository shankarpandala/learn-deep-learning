import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function CascadeDemo() {
  const [stage, setStage] = useState(0)
  const stages = [
    { name: 'Input', boxes: 12, color: '#9ca3af' },
    { name: 'P-Net (12x12)', boxes: 8, color: '#8b5cf6' },
    { name: 'R-Net (24x24)', boxes: 4, color: '#7c3aed' },
    { name: 'O-Net (48x48)', boxes: 2, color: '#6d28d9' },
  ]
  const W = 300, H = 140

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">MTCNN Cascade Stages</h3>
      <div className="flex gap-2 mb-4">
        {stages.map((s, i) => (
          <button key={i} onClick={() => setStage(i)}
            className={`px-3 py-1 rounded text-sm ${stage === i ? 'bg-violet-500 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300'}`}>
            {s.name}
          </button>
        ))}
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <rect x={10} y={10} width={120} height={120} fill="#f3f4f6" stroke="#d1d5db" rx={4} />
        <text x={70} y={75} textAnchor="middle" fontSize={11} fill="#6b7280">Image</text>
        {Array.from({ length: stages[stage].boxes }).map((_, i) => {
          const bx = 150 + (i % 4) * 35
          const by = 20 + Math.floor(i / 4) * 55
          return <rect key={i} x={bx} y={by} width={28} height={28} fill="none" stroke={stages[stage].color} strokeWidth={2} rx={2} />
        })}
        <text x={220} y={H - 8} fontSize={11} fill="#6b7280">{stages[stage].boxes} candidates</text>
      </svg>
    </div>
  )
}

export default function FaceDetection() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Face detection localizes all faces in an image with bounding boxes and optional landmark points.
        Modern detectors achieve robust performance across scales, poses, and occlusions.
      </p>

      <DefinitionBlock title="Multi-Task Face Detection">
        <p>Face detectors jointly optimize multiple objectives:</p>
        <BlockMath math="\mathcal{L} = \lambda_1 \mathcal{L}_{\text{cls}} + \lambda_2 \mathcal{L}_{\text{box}} + \lambda_3 \mathcal{L}_{\text{landmark}}" />
        <p className="mt-2">
          where classification loss determines face/non-face, box regression refines location,
          and landmark loss localizes facial keypoints (eyes, nose, mouth corners).
        </p>
      </DefinitionBlock>

      <CascadeDemo />

      <TheoremBlock title="RetinaFace Multi-Task Loss" id="retinaface">
        <p>RetinaFace adds a dense regression branch for 3D face vertices:</p>
        <BlockMath math="\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda_1 \mathcal{L}_{\text{box}} + \lambda_2 \mathcal{L}_{\text{pts}} + \lambda_3 \mathcal{L}_{\text{mesh}}" />
        <p className="mt-1">
          The mesh loss leverages a graph convolution decoder that predicts a 3D face
          shape <InlineMath math="\mathbf{S} \in \mathbb{R}^{N \times 3}" />, providing self-supervision
          that improves 2D detection accuracy.
        </p>
      </TheoremBlock>

      <ExampleBlock title="MTCNN Pipeline">
        <ol className="list-decimal ml-5 space-y-1">
          <li><strong>P-Net</strong>: Shallow CNN on image pyramid, produces candidate boxes at 12x12</li>
          <li><strong>R-Net</strong>: Refines candidates at 24x24, rejects false positives</li>
          <li><strong>O-Net</strong>: Final stage at 48x48, outputs boxes + 5 landmarks</li>
        </ol>
        <p className="mt-2">Each stage reduces candidates by roughly 50-80%.</p>
      </ExampleBlock>

      <PythonCode
        title="RetinaFace with InsightFace"
        code={`import torch
import torch.nn as nn

class RetinaFaceHead(nn.Module):
    """Simplified RetinaFace detection head."""
    def __init__(self, in_channels=256, num_anchors=2):
        super().__init__()
        self.cls = nn.Conv2d(in_channels, num_anchors * 2, 1)
        self.box = nn.Conv2d(in_channels, num_anchors * 4, 1)
        self.landmark = nn.Conv2d(in_channels, num_anchors * 10, 1)

    def forward(self, x):
        return self.cls(x), self.box(x), self.landmark(x)

# Multi-scale feature pyramid for face detection
class FaceFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([
            RetinaFaceHead() for _ in range(3)  # 3 FPN levels
        ])

    def forward(self, fpn_features):
        results = []
        for feat, head in zip(fpn_features, self.heads):
            cls, box, lmk = head(feat)
            results.append({
                'cls': cls,     # (B, 2A, H, W)
                'box': box,     # (B, 4A, H, W)
                'lmk': lmk,    # (B, 10A, H, W) - 5 landmarks
            })
        return results

# Evaluation: compute AP at IoU=0.5
def compute_ap(pred_boxes, gt_boxes, iou_thresh=0.5):
    """Average precision for face detection."""
    from torchvision.ops import box_iou
    ious = box_iou(pred_boxes, gt_boxes)
    matches = ious.max(dim=1).values >= iou_thresh
    return matches.float().mean()`}
      />

      <NoteBlock type="note" title="Handling Tiny Faces">
        <p>
          Detecting small faces (under 20px) remains challenging. Key strategies include
          using high-resolution feature maps from FPN, training with image pyramids,
          and employing special anchor designs for small scales. RetinaFace achieves
          91.4% AP on WIDER FACE hard set by leveraging these techniques.
        </p>
      </NoteBlock>
    </div>
  )
}

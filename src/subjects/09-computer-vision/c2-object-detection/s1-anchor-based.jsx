import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function IoUDemo() {
  const [bx, setBx] = useState(60)
  const gt = { x: 40, y: 40, w: 80, h: 60 }
  const pred = { x: bx, y: 50, w: 70, h: 50 }
  const x1 = Math.max(gt.x, pred.x), y1 = Math.max(gt.y, pred.y)
  const x2 = Math.min(gt.x + gt.w, pred.x + pred.w), y2 = Math.min(gt.y + gt.h, pred.y + pred.h)
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  const union = gt.w * gt.h + pred.w * pred.h - inter
  const iou = union > 0 ? inter / union : 0

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">IoU Interactive Demo</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Prediction X offset:
        <input type="range" min={0} max={140} value={bx} onChange={e => setBx(parseInt(e.target.value))} className="w-40 accent-violet-500" />
        <span className="font-mono">IoU = {iou.toFixed(3)}</span>
      </label>
      <svg width={240} height={140} className="mx-auto block">
        <rect x={gt.x} y={gt.y} width={gt.w} height={gt.h} fill="none" stroke="#8b5cf6" strokeWidth={2} strokeDasharray="4,2" />
        <rect x={pred.x} y={pred.y} width={pred.w} height={pred.h} fill="none" stroke="#f97316" strokeWidth={2} />
        {inter > 0 && <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill="#8b5cf6" opacity={0.25} />}
        <text x={gt.x} y={gt.y - 4} fontSize={10} fill="#8b5cf6">GT</text>
        <text x={pred.x} y={pred.y - 4} fontSize={10} fill="#f97316">Pred</text>
      </svg>
    </div>
  )
}

export default function AnchorBased() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Anchor-based detectors place predefined bounding box priors across the image and refine
        them to match objects. This two-stage paradigm (e.g., Faster R-CNN) remains highly accurate.
      </p>

      <DefinitionBlock title="Intersection over Union (IoU)">
        <BlockMath math="\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}" />
        <p className="mt-2">
          IoU is the primary metric for matching predictions to ground truth boxes.
          A typical positive threshold is <InlineMath math="\text{IoU} \geq 0.5" />.
        </p>
      </DefinitionBlock>

      <IoUDemo />

      <TheoremBlock title="Anchor Box Regression" id="anchor-regression">
        <p>Given an anchor <InlineMath math="(x_a, y_a, w_a, h_a)" />, the network predicts offsets:</p>
        <BlockMath math="\hat{x} = x_a + t_x w_a, \quad \hat{y} = y_a + t_y h_a" />
        <BlockMath math="\hat{w} = w_a e^{t_w}, \quad \hat{h} = h_a e^{t_h}" />
        <p className="mt-1">
          The smooth L1 loss penalizes box regression errors robustly.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Non-Maximum Suppression (NMS)">
        <p>NMS removes duplicate detections by iteratively:</p>
        <ol className="list-decimal ml-5 mt-2 space-y-1">
          <li>Select the box with highest confidence score</li>
          <li>Remove all boxes with <InlineMath math="\text{IoU} > \tau_{\text{nms}}" /> against it</li>
          <li>Repeat until no boxes remain</li>
        </ol>
        <p className="mt-1">Typical <InlineMath math="\tau_{\text{nms}} = 0.5" />.</p>
      </ExampleBlock>

      <PythonCode
        title="Faster R-CNN with Torchvision"
        code={`import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.ops import nms, box_iou

# Load pretrained Faster R-CNN
model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
model.eval()

# Inference
images = [torch.rand(3, 640, 480)]
with torch.no_grad():
    predictions = model(images)

boxes = predictions[0]['boxes']    # (N, 4) xyxy format
scores = predictions[0]['scores']  # (N,)
labels = predictions[0]['labels']  # (N,)

# Apply NMS manually
keep = nms(boxes, scores, iou_threshold=0.5)
filtered_boxes = boxes[keep]

# Compute IoU matrix between predictions and GT
gt_boxes = torch.tensor([[50, 50, 200, 200]], dtype=torch.float)
ious = box_iou(filtered_boxes, gt_boxes)
print(f"IoU with GT: {ious.squeeze()}")`}
      />

      <NoteBlock type="note" title="Feature Pyramid Networks">
        <p>
          FPN builds a multi-scale feature pyramid by combining top-down and lateral connections.
          This enables detecting objects at different scales: large objects from deep (low-res)
          features and small objects from shallow (high-res) features. Most modern anchor-based
          detectors use FPN as the backbone neck.
        </p>
      </NoteBlock>
    </div>
  )
}

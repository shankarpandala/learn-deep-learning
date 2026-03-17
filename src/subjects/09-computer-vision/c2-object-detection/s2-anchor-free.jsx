import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function CenterNetDemo() {
  const [cx, setCx] = useState(120)
  const [cy, setCy] = useState(80)
  const W = 260, H = 180
  const sigma = 20
  const heatmapPoints = []
  for (let x = 0; x < W; x += 4) {
    for (let y = 0; y < H; y += 4) {
      const d2 = (x - cx) ** 2 + (y - cy) ** 2
      const val = Math.exp(-d2 / (2 * sigma * sigma))
      if (val > 0.05) heatmapPoints.push({ x, y, val })
    }
  }

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">CenterNet Heatmap Demo</h3>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Click to move the object center</p>
      <svg width={W} height={H} className="mx-auto block border border-gray-200 dark:border-gray-700 rounded cursor-crosshair"
        onClick={e => { const r = e.currentTarget.getBoundingClientRect(); setCx(e.clientX - r.left); setCy(e.clientY - r.top) }}>
        {heatmapPoints.map((p, i) => (
          <rect key={i} x={p.x} y={p.y} width={4} height={4} fill="#8b5cf6" opacity={p.val * 0.8} />
        ))}
        <circle cx={cx} cy={cy} r={3} fill="#f97316" />
        <rect x={cx - 40} y={cy - 30} width={80} height={60} fill="none" stroke="#f97316" strokeWidth={1.5} />
        <text x={cx + 5} y={cy - 33} fontSize={10} fill="#f97316">center</text>
      </svg>
    </div>
  )
}

export default function AnchorFree() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Anchor-free detectors eliminate handcrafted anchor boxes by directly predicting object
        locations as keypoints or center points, simplifying the detection pipeline.
      </p>

      <DefinitionBlock title="CenterNet Formulation">
        <p>
          CenterNet predicts a heatmap <InlineMath math="\hat{Y} \in [0,1]^{H \times W \times C}" /> where
          peaks correspond to object centers. For each center, it regresses:
        </p>
        <BlockMath math="\hat{Y}_{xyc} = \exp\!\left(-\frac{(x - \tilde{p}_x)^2 + (y - \tilde{p}_y)^2}{2\sigma_p^2}\right)" />
        <p className="mt-2">
          Plus offset <InlineMath math="\hat{O} \in \mathbb{R}^{2}" /> and size <InlineMath math="\hat{S} \in \mathbb{R}^{2}" /> at each center.
        </p>
      </DefinitionBlock>

      <CenterNetDemo />

      <TheoremBlock title="FCOS: Fully Convolutional One-Stage" id="fcos">
        <p>
          FCOS predicts, for each spatial location <InlineMath math="(x,y)" /> on the feature map:
        </p>
        <BlockMath math="(l^*, t^*, r^*, b^*) = (x - x_0, y - y_0, x_1 - x, y_1 - y)" />
        <p className="mt-1">
          These are distances from the location to the four sides of the bounding box.
          A centerness score <InlineMath math="\sqrt{\frac{\min(l,r)}{\max(l,r)} \cdot \frac{\min(t,b)}{\max(t,b)}}" /> suppresses
          low-quality predictions far from object centers.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Anchor-Free vs Anchor-Based">
        <p>Key advantages of anchor-free methods:</p>
        <ul className="list-disc ml-5 mt-2 space-y-1">
          <li>No hyperparameters for anchor sizes, ratios, or aspect ratios</li>
          <li>Simpler training with fewer positive/negative sampling heuristics</li>
          <li>Naturally handle objects of arbitrary shape</li>
          <li>CenterNet achieves 45.1 AP on COCO at real-time speeds</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="CenterNet-Style Detection Head"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterNetHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Heatmap head (object centers)
        self.heatmap = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )
        # Box size head (width, height)
        self.size = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
        )
        # Offset head (sub-pixel refinement)
        self.offset = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
        )

    def forward(self, x):
        hm = torch.sigmoid(self.heatmap(x))
        sz = self.size(x)
        off = self.offset(x)
        return hm, sz, off

# Focal loss for heatmap training
def focal_loss(pred, gt, alpha=2, beta=4):
    pos = gt.eq(1).float()
    neg = gt.lt(1).float()
    pos_loss = -((1 - pred)**alpha * torch.log(pred + 1e-6)) * pos
    neg_loss = -((1 - gt)**beta * pred**alpha * torch.log(1 - pred + 1e-6)) * neg
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1)`}
      />

      <NoteBlock type="note" title="Keypoint Detection">
        <p>
          Anchor-free methods naturally extend to keypoint detection (e.g., CornerNet detects
          top-left and bottom-right corners). This paradigm unifies object detection, pose
          estimation, and instance segmentation under a single keypoint-based framework.
        </p>
      </NoteBlock>
    </div>
  )
}

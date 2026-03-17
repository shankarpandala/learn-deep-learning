import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function PoseDemo() {
  const [poseOffset, setPoseOffset] = useState(0)
  const W = 260, H = 240
  const rad = poseOffset * Math.PI / 180

  const joints = {
    head: [130, 35],
    neck: [130, 60],
    lShoulder: [100, 70], rShoulder: [160, 70],
    lElbow: [80 + Math.sin(rad) * 15, 105], rElbow: [180 - Math.sin(rad) * 15, 105],
    lWrist: [65 + Math.sin(rad) * 25, 140], rWrist: [195 - Math.sin(rad) * 25, 140],
    lHip: [110, 140], rHip: [150, 140],
    lKnee: [105, 180], rKnee: [155, 180],
    lAnkle: [100, 215], rAnkle: [160, 215],
  }

  const bones = [
    ['head', 'neck'], ['neck', 'lShoulder'], ['neck', 'rShoulder'],
    ['lShoulder', 'lElbow'], ['rShoulder', 'rElbow'],
    ['lElbow', 'lWrist'], ['rElbow', 'rWrist'],
    ['neck', 'lHip'], ['neck', 'rHip'],
    ['lHip', 'lKnee'], ['rHip', 'rKnee'],
    ['lKnee', 'lAnkle'], ['rKnee', 'rAnkle'],
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">2D Pose Skeleton</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Arm angle:
        <input type="range" min={-60} max={60} value={poseOffset} onChange={e => setPoseOffset(parseInt(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        {bones.map(([a, b], i) => (
          <line key={i} x1={joints[a][0]} y1={joints[a][1]} x2={joints[b][0]} y2={joints[b][1]}
            stroke="#8b5cf6" strokeWidth={2.5} strokeLinecap="round" />
        ))}
        {Object.values(joints).map((pos, i) => (
          <circle key={i} cx={pos[0]} cy={pos[1]} r={4} fill="#7c3aed" stroke="white" strokeWidth={1.5} />
        ))}
      </svg>
    </div>
  )
}

export default function Pose2D() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        2D pose estimation localizes body keypoints (joints) in image coordinates. The two main
        approaches are heatmap-based regression and direct coordinate regression.
      </p>

      <DefinitionBlock title="Heatmap-Based Pose Estimation">
        <p>
          For each keypoint <InlineMath math="k" />, predict a heatmap <InlineMath math="H_k \in \mathbb{R}^{h \times w}" /> where
          the target is a 2D Gaussian centered at the ground truth location:
        </p>
        <BlockMath math="H_k^*(x, y) = \exp\!\left(-\frac{(x - x_k)^2 + (y - y_k)^2}{2\sigma^2}\right)" />
        <p className="mt-2">
          The predicted keypoint is at <InlineMath math="\arg\max_{x,y} H_k(x, y)" />.
        </p>
      </DefinitionBlock>

      <PoseDemo />

      <TheoremBlock title="Top-Down vs Bottom-Up" id="pose-paradigm">
        <p><strong>Top-down</strong>: Detect persons first, then estimate pose per crop.</p>
        <BlockMath math="\text{Detector} \rightarrow \text{Crop} \rightarrow \text{Pose Net} \rightarrow K \text{ keypoints}" />
        <p className="mt-2"><strong>Bottom-up</strong>: Detect all keypoints first, then group into persons.</p>
        <BlockMath math="\text{All Keypoints} \xrightarrow{\text{association}} \text{Person Instances}" />
        <p className="mt-1">Top-down is more accurate; bottom-up is faster for multi-person scenes.</p>
      </TheoremBlock>

      <ExampleBlock title="COCO Keypoint Format">
        <p>The COCO dataset defines 17 body keypoints. The evaluation metric OKS (Object Keypoint Similarity):</p>
        <BlockMath math="\text{OKS} = \frac{\sum_k \exp\!\left(-d_k^2 / (2s^2\kappa_k^2)\right) \cdot \delta(v_k > 0)}{\sum_k \delta(v_k > 0)}" />
        <p className="mt-1">
          where <InlineMath math="d_k" /> is the Euclidean distance, <InlineMath math="s" /> is
          object scale, and <InlineMath math="\kappa_k" /> is a per-keypoint constant.
        </p>
      </ExampleBlock>

      <PythonCode
        title="HRNet Pose Estimation"
        code={`import torch
import torch.nn as nn

class SimpleHeatmapHead(nn.Module):
    """Heatmap prediction head for pose estimation."""
    def __init__(self, in_channels=256, num_keypoints=17):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(256, num_keypoints, 1)

    def forward(self, x):
        x = self.deconv(x)
        heatmaps = self.head(x)  # (B, K, H, W)
        return heatmaps

def decode_heatmaps(heatmaps):
    """Get keypoint coordinates from heatmaps."""
    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)
    max_idx = flat.argmax(dim=2)
    y = max_idx // W
    x = max_idx % W
    conf = flat.max(dim=2).values
    return torch.stack([x, y, conf], dim=2)  # (B, K, 3)

# Loss: MSE between predicted and target heatmaps
criterion = nn.MSELoss()
loss = criterion(pred_heatmaps, target_heatmaps)`}
      />

      <NoteBlock type="note" title="High-Resolution Networks (HRNet)">
        <p>
          HRNet maintains high-resolution representations throughout the network by running
          parallel multi-resolution branches with repeated feature exchange. This avoids the
          information loss from downsampling-then-upsampling in encoder-decoder designs,
          achieving state-of-the-art results: 77.0 AP on COCO keypoint detection.
        </p>
      </NoteBlock>
    </div>
  )
}

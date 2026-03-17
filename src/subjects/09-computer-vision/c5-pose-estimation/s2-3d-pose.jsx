import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function Lifting3DDemo() {
  const [rotY, setRotY] = useState(0)
  const W = 280, H = 220
  const rad = rotY * Math.PI / 180

  const joints3D = [
    [0, -0.8, 0], [0, -0.5, 0], [-0.25, -0.4, 0], [0.25, -0.4, 0],
    [-0.45, -0.1, 0.1], [0.45, -0.1, -0.1],
    [0, 0, 0], [-0.15, 0.3, 0], [0.15, 0.3, 0],
    [-0.15, 0.7, 0], [0.15, 0.7, 0],
  ]

  const project = ([x, y, z]) => {
    const rx = x * Math.cos(rad) - z * Math.sin(rad)
    return [W / 2 + rx * 100, H / 2 + y * 110]
  }

  const bones = [[0, 1], [1, 2], [1, 3], [2, 4], [3, 5], [1, 6], [6, 7], [6, 8], [7, 9], [8, 10]]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">3D Pose Rotation</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Y-axis rotation: {rotY}&deg;
        <input type="range" min={-90} max={90} value={rotY} onChange={e => setRotY(parseInt(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        {bones.map(([a, b], i) => {
          const pa = project(joints3D[a]), pb = project(joints3D[b])
          return <line key={i} x1={pa[0]} y1={pa[1]} x2={pb[0]} y2={pb[1]} stroke="#8b5cf6" strokeWidth={2.5} strokeLinecap="round" />
        })}
        {joints3D.map((j, i) => {
          const p = project(j)
          return <circle key={i} cx={p[0]} cy={p[1]} r={4} fill="#7c3aed" stroke="white" strokeWidth={1.5} />
        })}
      </svg>
    </div>
  )
}

export default function Pose3D() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        3D pose estimation recovers the three-dimensional positions of body joints from images or
        video. The key challenge is the inherent depth ambiguity in 2D projections.
      </p>

      <DefinitionBlock title="2D-to-3D Lifting">
        <p>
          Given 2D keypoints <InlineMath math="p \in \mathbb{R}^{K \times 2}" />, predict 3D
          positions <InlineMath math="P \in \mathbb{R}^{K \times 3}" /> via a lifting network:
        </p>
        <BlockMath math="P = g_\phi(p) \quad \text{where} \quad g: \mathbb{R}^{K \times 2} \rightarrow \mathbb{R}^{K \times 3}" />
        <p className="mt-2">
          The 2D-to-3D projection relationship under weak perspective:
        </p>
        <BlockMath math="p = \Pi P = \begin{bmatrix} f & 0 \\ 0 & f \end{bmatrix} \begin{bmatrix} X/Z \\ Y/Z \end{bmatrix}" />
      </DefinitionBlock>

      <Lifting3DDemo />

      <TheoremBlock title="Depth Ambiguity" id="depth-ambiguity">
        <p>
          A single 2D projection has infinitely many 3D interpretations. The reprojection loss alone
          is insufficient:
        </p>
        <BlockMath math="\mathcal{L}_{\text{reproj}} = \sum_k \|\Pi(P_k) - p_k\|^2" />
        <p className="mt-1">
          Structural priors (bone lengths, joint angle limits) and temporal consistency
          constrain the solution space:
        </p>
        <BlockMath math="\mathcal{L} = \mathcal{L}_{\text{3D}} + \lambda_1 \mathcal{L}_{\text{bone}} + \lambda_2 \mathcal{L}_{\text{smooth}}" />
      </TheoremBlock>

      <ExampleBlock title="3D Pose Approaches">
        <ul className="list-disc ml-5 space-y-1">
          <li><strong>Lifting</strong>: SimpleBL lifts 2D detections with a residual MLP (MPJPE: 36.5mm)</li>
          <li><strong>Volumetric</strong>: Predict 3D heatmaps <InlineMath math="H \in \mathbb{R}^{D \times H \times W}" /></li>
          <li><strong>Temporal</strong>: VideoPose3D uses dilated convolutions over 2D pose sequences</li>
          <li><strong>Multi-view</strong>: Triangulate from calibrated cameras for ground truth</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Simple 2D-to-3D Lifting Network"
        code={`import torch
import torch.nn as nn

class LiftingNet(nn.Module):
    """2D-to-3D lifting with residual blocks (SimpleBL)."""
    def __init__(self, num_joints=17, hidden=1024, num_blocks=2):
        super().__init__()
        self.input_proj = nn.Linear(num_joints * 2, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
            ) for _ in range(num_blocks)
        ])
        self.output = nn.Linear(hidden, num_joints * 3)

    def forward(self, x_2d):
        # x_2d: (B, 17, 2) -> flatten
        x = self.input_proj(x_2d.view(x_2d.shape[0], -1))
        for block in self.blocks:
            x = x + block(x)  # Residual connection
        out = self.output(x)
        return out.view(-1, 17, 3)  # (B, 17, 3)

# MPJPE loss (mean per-joint position error)
def mpjpe(pred, target):
    return torch.norm(pred - target, dim=-1).mean()

# Procrustes-aligned MPJPE (P-MPJPE)
def p_mpjpe(pred, target):
    """Align pred to target via Procrustes then compute MPJPE."""
    # Center both
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    tgt_c = target - target.mean(dim=1, keepdim=True)
    # SVD for optimal rotation
    U, S, V = torch.svd(tgt_c.transpose(1, 2) @ pred_c)
    R = V @ U.transpose(1, 2)
    return torch.norm(pred_c @ R.transpose(1, 2) - tgt_c, dim=-1).mean()`}
      />

      <NoteBlock type="note" title="Temporal Models">
        <p>
          Processing 2D poses over time dramatically improves 3D accuracy. VideoPose3D uses
          temporal convolutions with receptive fields of 243 frames, reducing MPJPE from 52mm
          (single-frame) to 37mm. The temporal smoothness naturally resolves depth ambiguities
          and reduces jitter in predictions.
        </p>
      </NoteBlock>
    </div>
  )
}

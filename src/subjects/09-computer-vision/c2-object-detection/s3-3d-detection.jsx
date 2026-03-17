import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function PointCloudDemo() {
  const [viewAngle, setViewAngle] = useState(0)
  const W = 280, H = 200
  const points = Array.from({ length: 80 }, (_, i) => ({
    x: Math.cos(i * 0.45) * 40 + Math.random() * 20,
    y: Math.sin(i * 0.45) * 30 + Math.random() * 15,
    z: (i % 20) * 2 - 20 + Math.random() * 5,
  }))
  const rad = viewAngle * Math.PI / 180

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">3D Point Cloud View</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Rotation: {viewAngle}&deg;
        <input type="range" min={0} max={360} value={viewAngle} onChange={e => setViewAngle(parseInt(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        {points.map((p, i) => {
          const rx = p.x * Math.cos(rad) - p.z * Math.sin(rad)
          const rz = p.x * Math.sin(rad) + p.z * Math.cos(rad)
          const scale = 1 + rz / 100
          return <circle key={i} cx={W / 2 + rx * scale} cy={H / 2 - p.y * scale} r={2 * scale}
            fill="#8b5cf6" opacity={0.4 + scale * 0.3} />
        })}
        <rect x={W / 2 - 45} y={H / 2 - 35} width={90} height={50} fill="none" stroke="#f97316" strokeWidth={1.5} strokeDasharray="4,2" />
        <text x={W / 2 - 44} y={H / 2 - 38} fontSize={10} fill="#f97316">3D BBox</text>
      </svg>
    </div>
  )
}

export default function Detection3D() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        3D object detection estimates object positions, dimensions, and orientations in three-dimensional
        space. Key representations include point clouds from LiDAR and bird's-eye view projections.
      </p>

      <DefinitionBlock title="3D Bounding Box">
        <p>A 3D box is parameterized by 7 degrees of freedom:</p>
        <BlockMath math="\mathbf{b} = (x, y, z, w, h, l, \theta)" />
        <p className="mt-2">
          where <InlineMath math="(x, y, z)" /> is the center, <InlineMath math="(w, h, l)" /> are
          dimensions, and <InlineMath math="\theta" /> is yaw rotation. The 3D IoU considers
          volumetric overlap.
        </p>
      </DefinitionBlock>

      <PointCloudDemo />

      <TheoremBlock title="PointNet Feature Extraction" id="pointnet">
        <p>
          PointNet processes unordered point sets directly. For <InlineMath math="N" /> points
          with features <InlineMath math="x_i" />:
        </p>
        <BlockMath math="f = \gamma\!\left(\max_{i=1,\ldots,N} h(x_i)\right)" />
        <p className="mt-1">
          where <InlineMath math="h" /> is a shared MLP and <InlineMath math="\gamma" /> is another MLP.
          The max-pooling ensures permutation invariance over input points.
        </p>
      </TheoremBlock>

      <ExampleBlock title="3D Detection Methods Comparison">
        <ul className="list-disc ml-5 space-y-1">
          <li><strong>Point-based</strong>: PointRCNN processes raw points directly (accurate, slow)</li>
          <li><strong>Voxel-based</strong>: VoxelNet/SECOND discretize into voxel grid (fast, scalable)</li>
          <li><strong>Pillar-based</strong>: PointPillars uses vertical columns (real-time, autonomous driving)</li>
          <li><strong>BEV-based</strong>: BEVFusion projects camera + LiDAR to bird's-eye view</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Voxelization and 3D Detection"
        code={`import torch
import torch.nn as nn

class VoxelEncoder(nn.Module):
    """Simplified voxel feature encoder (VFE)."""
    def __init__(self, in_dim=4, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )

    def forward(self, voxel_features, voxel_count):
        # voxel_features: (num_voxels, max_points, C)
        # Aggregate points within each voxel
        x = self.mlp(voxel_features.view(-1, 4))
        x = x.view(*voxel_features.shape[:2], -1)
        # Max pooling over points in each voxel
        x = x.max(dim=1).values  # (num_voxels, hidden)
        return x

# Bird's Eye View projection
def scatter_to_bev(voxel_feats, coords, grid_size=(512, 512)):
    """Scatter voxel features to BEV grid."""
    bev = torch.zeros(1, voxel_feats.shape[1],
                      *grid_size, device=voxel_feats.device)
    bev[0, :, coords[:, 2], coords[:, 3]] = voxel_feats.T
    return bev  # Apply 2D conv backbone on BEV map

# Loss: smooth L1 for box regression + focal for class
def detection_loss(pred_boxes, gt_boxes, pred_cls, gt_cls):
    reg_loss = nn.SmoothL1Loss()(pred_boxes, gt_boxes)
    cls_loss = sigmoid_focal_loss(pred_cls, gt_cls)
    return reg_loss + cls_loss`}
      />

      <NoteBlock type="note" title="Multi-Modal Fusion">
        <p>
          Modern autonomous driving systems fuse camera images with LiDAR point clouds.
          BEVFusion projects both modalities into a shared bird's-eye view space using
          camera-to-BEV transformations like LSS (Lift, Splat, Shoot), enabling unified
          feature extraction and detection.
        </p>
      </NoteBlock>
    </div>
  )
}

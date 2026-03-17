import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function SkeletonViewer() {
  const [frame, setFrame] = useState(0)
  const W = 200, H = 250

  const keypoints = [
    [100, 30],   // 0: head
    [100, 70],   // 1: neck
    [70, 70],    // 2: l_shoulder
    [130, 70],   // 3: r_shoulder
    [50, 120],   // 4: l_elbow
    [150, 120],  // 5: r_elbow
    [40, 160],   // 6: l_wrist
    [160, 160],  // 7: r_wrist
    [85, 140],   // 8: l_hip
    [115, 140],  // 9: r_hip
    [80, 190],   // 10: l_knee
    [120, 190],  // 11: r_knee
    [75, 230],   // 12: l_ankle
    [125, 230],  // 13: r_ankle
  ]

  const bones = [[0,1],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7],[1,8],[1,9],[8,10],[9,11],[10,12],[11,13],[8,9]]

  const dx = Math.sin(frame * 0.3) * 15
  const animated = keypoints.map(([x, y], i) => {
    if (i === 6) return [x + dx * 2, y - Math.abs(dx)]
    if (i === 7) return [x - dx * 2, y - Math.abs(dx)]
    if (i === 4) return [x + dx, y]
    if (i === 5) return [x - dx, y]
    return [x, y]
  })

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Skeleton Joint Visualization</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Frame: {frame}
        <input type="range" min={0} max={20} step={1} value={frame} onChange={e => setFrame(Number(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        {bones.map(([a, b], i) => (
          <line key={i} x1={animated[a][0]} y1={animated[a][1]} x2={animated[b][0]} y2={animated[b][1]} stroke="#8b5cf6" strokeWidth={2} />
        ))}
        {animated.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r={4} fill="#7c3aed" stroke="white" strokeWidth={1.5} />
        ))}
      </svg>
      <p className="text-xs text-gray-500 text-center mt-1">
        14 joints, {bones.length} bones &mdash; Input tensor: [N, C, T, V, M] = [batch, channels, frames, joints, persons]
      </p>
    </div>
  )
}

export default function SkeletonBasedRecognition() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Skeleton-based action recognition uses body joint coordinates as input, offering
        robustness to appearance variations, viewpoint changes, and background clutter.
        Graph Convolutional Networks model the body as a graph for powerful spatial-temporal reasoning.
      </p>

      <DefinitionBlock title="Skeleton Graph Representation">
        <p>The human skeleton is represented as a spatial-temporal graph:</p>
        <BlockMath math="G = (V, E_s \cup E_t), \quad V = \{v_{t,j} | t \in T, j \in J\}" />
        <p className="mt-2">
          Vertices are joint positions over time. Spatial edges <InlineMath math="E_s" /> connect
          physically linked joints (bones), while temporal edges <InlineMath math="E_t" /> connect
          the same joint across consecutive frames. Input features are typically
          <InlineMath math="(x, y, z)" /> coordinates or <InlineMath math="(x, y, \text{confidence})" />.
        </p>
      </DefinitionBlock>

      <SkeletonViewer />

      <TheoremBlock title="Spatial-Temporal Graph Convolution (ST-GCN)" id="stgcn">
        <p>ST-GCN applies graph convolution on the skeleton graph with a learnable adjacency matrix:</p>
        <BlockMath math="f_\text{out} = \sum_{k=0}^{K-1} \hat{A}_k X W_k, \quad \hat{A}_k = D_k^{-1/2}(A_k + I)D_k^{-1/2}" />
        <p className="mt-1">
          The adjacency matrix <InlineMath math="A" /> is partitioned into <InlineMath math="K" /> subsets
          based on distance from the root joint (centripetal, root, centrifugal). This spatial
          convolution is combined with temporal convolution for full spatiotemporal modeling:
        </p>
        <BlockMath math="f = \text{TemporalConv}(\text{GraphConv}(X, A))" />
      </TheoremBlock>

      <ExampleBlock title="Adaptive Graph Structures">
        <p>
          Modern approaches learn the graph topology rather than using the physical skeleton:
        </p>
        <BlockMath math="A_\text{adaptive} = A_\text{physical} + B_\text{learnable} + C(X)" />
        <p className="mt-1">
          where <InlineMath math="B" /> is a learnable residual graph and <InlineMath math="C(X)" /> is a
          data-dependent graph computed via dot-product attention between joint features.
          This allows discovering non-physical but semantically meaningful connections
          (e.g., left hand to right hand for clapping).
        </p>
      </ExampleBlock>

      <PythonCode
        title="Skeleton-Based Action Recognition with GCN"
        code={`import torch
import torch.nn as nn

class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_joints=25, A=None, t_kernel=9):
        super().__init__()
        self.num_joints = num_joints
        # Learnable adjacency matrix
        self.A = nn.Parameter(torch.eye(num_joints).unsqueeze(0))
        # Spatial graph convolution
        self.gcn = nn.Conv2d(in_ch, out_ch, 1)
        self.bn_s = nn.BatchNorm2d(out_ch)
        # Temporal convolution
        self.tcn = nn.Conv2d(out_ch, out_ch, (t_kernel, 1), padding=(t_kernel//2, 0))
        self.bn_t = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        # x: [B, C, T, V] where V=num_joints
        B, C, T, V = x.shape
        # Graph convolution: multiply by adjacency
        A = self.A.softmax(dim=-1)
        x_g = torch.einsum('bctv,kvw->bctw', x, A)
        x_g = torch.relu(self.bn_s(self.gcn(x_g)))
        # Temporal convolution
        x_t = torch.relu(self.bn_t(self.tcn(x_g)))
        return x_t

class SkeletonClassifier(nn.Module):
    def __init__(self, num_joints=25, num_classes=60):
        super().__init__()
        self.blocks = nn.Sequential(
            STGCNBlock(3, 64, num_joints),
            STGCNBlock(64, 128, num_joints),
            STGCNBlock(128, 256, num_joints),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

# NTU-RGBD skeleton input: [B, C, T, V]
B, C, T, V = 4, 3, 64, 25  # 3D coords, 64 frames, 25 joints
skeleton = torch.randn(B, C, T, V)
model = SkeletonClassifier(num_joints=V, num_classes=60)
logits = model(skeleton)
print(f"Skeleton input: {skeleton.shape}")
print(f"Action logits: {logits.shape}")  # [4, 60]`}
      />

      <NoteBlock type="note" title="Multi-Modal Fusion">
        <p>
          Combining skeleton data with RGB features improves robustness: skeletons provide
          structural motion information while RGB captures appearance and context. Recent methods
          like PoseConv3D create 3D heatmap volumes from skeletons, enabling processing with
          standard 3D CNNs and easy fusion with RGB features at the backbone level.
        </p>
      </NoteBlock>
    </div>
  )
}

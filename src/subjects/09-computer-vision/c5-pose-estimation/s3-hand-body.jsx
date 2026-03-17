import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function HandDemo() {
  const [spread, setSpread] = useState(0)
  const W = 240, H = 220
  const palm = [120, 160]
  const fingers = [
    { base: [85, 140], mid: [75, 110], tip: [70 - spread * 0.6, 75] },
    { base: [100, 125], mid: [95, 85], tip: [92 - spread * 0.3, 50] },
    { base: [118, 120], mid: [118, 78], tip: [118, 42] },
    { base: [136, 125], mid: [140, 85], tip: [143 + spread * 0.3, 50] },
    { base: [152, 135], mid: [162, 105], tip: [170 + spread * 0.6, 75] },
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Hand Keypoint Tracking</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Finger spread:
        <input type="range" min={0} max={30} value={spread} onChange={e => setSpread(parseInt(e.target.value))} className="w-36 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <circle cx={palm[0]} cy={palm[1]} r={22} fill="#ede9fe" stroke="#8b5cf6" strokeWidth={1.5} />
        {fingers.map((f, i) => (
          <g key={i}>
            <line x1={palm[0]} y1={palm[1] - 15} x2={f.base[0]} y2={f.base[1]} stroke="#c4b5fd" strokeWidth={2} />
            <line x1={f.base[0]} y1={f.base[1]} x2={f.mid[0]} y2={f.mid[1]} stroke="#8b5cf6" strokeWidth={2} />
            <line x1={f.mid[0]} y1={f.mid[1]} x2={f.tip[0]} y2={f.tip[1]} stroke="#7c3aed" strokeWidth={2} />
            {[f.base, f.mid, f.tip].map((pt, j) => (
              <circle key={j} cx={pt[0]} cy={pt[1]} r={3} fill="#7c3aed" stroke="white" strokeWidth={1} />
            ))}
          </g>
        ))}
      </svg>
    </div>
  )
}

export default function HandBody() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Dense pose estimation, hand tracking, and body mesh recovery go beyond sparse keypoints
        to produce detailed surface representations of the human body and hands.
      </p>

      <DefinitionBlock title="SMPL Body Model">
        <p>
          SMPL is a parametric body model controlled by pose <InlineMath math="\theta \in \mathbb{R}^{72}" /> and
          shape <InlineMath math="\beta \in \mathbb{R}^{10}" />:
        </p>
        <BlockMath math="M(\beta, \theta) = W(T_P(\beta, \theta), J(\beta), \theta, \mathbf{W})" />
        <p className="mt-2">
          where <InlineMath math="T_P" /> is the posed template, <InlineMath math="J" /> are joint
          locations, and <InlineMath math="\mathbf{W}" /> are blend skinning weights.
          Output: 6890 vertices forming a triangulated mesh.
        </p>
      </DefinitionBlock>

      <HandDemo />

      <TheoremBlock title="MANO Hand Model" id="mano">
        <p>
          MANO parameterizes hand meshes with pose <InlineMath math="\theta \in \mathbb{R}^{48}" /> (16 joints x 3)
          and shape <InlineMath math="\beta \in \mathbb{R}^{10}" />:
        </p>
        <BlockMath math="V = \bar{V} + B_S(\beta) + B_P(\theta)" />
        <p className="mt-1">
          where <InlineMath math="\bar{V}" /> is the mean hand, <InlineMath math="B_S" /> are shape
          blend shapes, and <InlineMath math="B_P" /> are pose-dependent correctives. The final
          mesh has 778 vertices and 21 joints.
        </p>
      </TheoremBlock>

      <ExampleBlock title="DensePose: Dense Correspondence">
        <p>DensePose maps every visible pixel to a UV coordinate on the body surface:</p>
        <BlockMath math="f: (x, y) \rightarrow (I, U, V)" />
        <ul className="list-disc ml-5 mt-2 space-y-1">
          <li><InlineMath math="I" />: body part index (1 of 24 parts)</li>
          <li><InlineMath math="(U, V) \in [0, 1]^2" />: surface coordinates within that part</li>
          <li>Enables pixel-level body surface correspondence across images</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Body Mesh Recovery (HMR)"
        code={`import torch
import torch.nn as nn

class HMRHead(nn.Module):
    """Human Mesh Recovery regression head."""
    def __init__(self, feat_dim=2048):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )
        # SMPL parameters
        self.pose = nn.Linear(1024, 72)     # 24 joints * 3
        self.shape = nn.Linear(1024, 10)    # Shape coefficients
        self.cam = nn.Linear(1024, 3)       # Weak-perspective camera

    def forward(self, features):
        x = self.fc(features)
        return {
            'pose': self.pose(x),
            'shape': self.shape(x),
            'camera': self.cam(x),
        }

# Hand landmark detection (MediaPipe-style)
class HandLandmarkNet(nn.Module):
    def __init__(self, num_landmarks=21):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.regressor = nn.Linear(64, num_landmarks * 3)

    def forward(self, hand_crop):
        feat = self.backbone(hand_crop).flatten(1)
        landmarks = self.regressor(feat)
        return landmarks.view(-1, 21, 3)  # (B, 21, xyz)`}
      />

      <NoteBlock type="note" title="Whole-Body Estimation">
        <p>
          Modern whole-body models like SMPL-X jointly model the body (22 joints), hands (30
          joints), and face (3 jaw joints + expression). This enables capturing complete
          human behavior including gestures and facial expressions in a single forward pass.
          Applications span AR/VR, sign language recognition, and motion capture.
        </p>
      </NoteBlock>
    </div>
  )
}

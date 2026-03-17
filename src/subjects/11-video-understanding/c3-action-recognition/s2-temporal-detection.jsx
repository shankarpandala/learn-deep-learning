import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function TemporalDetectionVisualization() {
  const [iouThreshold, setIouThreshold] = useState(0.5)

  const predictions = [
    { start: 10, end: 35, label: 'Running', conf: 0.9 },
    { start: 45, end: 70, label: 'Jumping', conf: 0.85 },
    { start: 80, end: 110, label: 'Throwing', conf: 0.7 },
  ]
  const groundTruth = [
    { start: 12, end: 38, label: 'Running' },
    { start: 50, end: 68, label: 'Jumping' },
    { start: 85, end: 120, label: 'Throwing' },
  ]

  const computeIoU = (p, g) => {
    const inter = Math.max(0, Math.min(p.end, g.end) - Math.max(p.start, g.start))
    const union = (p.end - p.start) + (g.end - g.start) - inter
    return inter / union
  }

  const totalLen = 130
  const W = 360

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Temporal Action Detection</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        IoU threshold: {iouThreshold.toFixed(2)}
        <input type="range" min={0.1} max={0.9} step={0.05} value={iouThreshold} onChange={e => setIouThreshold(Number(e.target.value))} className="w-32 accent-violet-500" />
      </label>
      <svg width={W} height={100} className="block">
        <text x={0} y={15} fontSize={10} fill="#8b5cf6">Ground Truth</text>
        {groundTruth.map((g, i) => (
          <rect key={`gt-${i}`} x={g.start / totalLen * W} y={20} width={(g.end - g.start) / totalLen * W} height={14} fill="#8b5cf6" opacity={0.4} rx={3} />
        ))}
        <text x={0} y={55} fontSize={10} fill="#f97316">Predictions</text>
        {predictions.map((p, i) => {
          const iou = computeIoU(p, groundTruth[i])
          const match = iou >= iouThreshold
          return (
            <g key={`pred-${i}`}>
              <rect x={p.start / totalLen * W} y={60} width={(p.end - p.start) / totalLen * W} height={14} fill={match ? '#22c55e' : '#ef4444'} opacity={0.6} rx={3} />
              <text x={(p.start + p.end) / 2 / totalLen * W} y={88} fontSize={8} fill="#6b7280" textAnchor="middle">
                IoU={iou.toFixed(2)}
              </text>
            </g>
          )
        })}
      </svg>
      <p className="text-xs text-gray-500 mt-1">
        Green = correct detection (IoU &ge; {iouThreshold.toFixed(2)}), Red = missed
      </p>
    </div>
  )
}

export default function TemporalActionDetection() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Temporal action detection localizes action instances in untrimmed videos, predicting
        both the temporal boundaries (start/end times) and the action class for each instance.
        This is the video equivalent of object detection in images.
      </p>

      <DefinitionBlock title="Temporal Action Detection Task">
        <p>Given an untrimmed video, predict a set of action instances:</p>
        <BlockMath math="\{(t_s^i, t_e^i, c^i, p^i)\}_{i=1}^{N}" />
        <p className="mt-2">
          where <InlineMath math="t_s^i, t_e^i" /> are start/end times, <InlineMath math="c^i" /> is
          the action class, and <InlineMath math="p^i" /> is a confidence score. Evaluation uses
          mAP at various temporal IoU thresholds (0.3, 0.5, 0.7).
        </p>
      </DefinitionBlock>

      <TemporalDetectionVisualization />

      <TheoremBlock title="Anchor-Based Detection" id="anchor-based-detection">
        <p>
          BMN (Boundary Matching Network) generates proposals by predicting start/end
          probability and proposal confidence for all candidate pairs:
        </p>
        <BlockMath math="\text{BM}(t_s, t_e) = \text{MLP}(\text{Pool}(f[t_s : t_e]))" />
        <p className="mt-1">
          Start and end boundaries are predicted independently, then combined.
          The boundary-matching confidence map <InlineMath math="\text{BM} \in \mathbb{R}^{T \times T}" />
          scores all <InlineMath math="(t_s, t_e)" /> pairs, filtered by NMS.
        </p>
      </TheoremBlock>

      <ExampleBlock title="ActionFormer: Anchor-Free Detection">
        <p>
          ActionFormer uses a Transformer encoder with multi-scale temporal feature pyramids:
        </p>
        <BlockMath math="(d_s^t, d_e^t, c_t) = \text{Head}(f_l^t)" />
        <p className="mt-1">
          At each temporal position <InlineMath math="t" /> and scale <InlineMath math="l" />, it predicts
          distances to boundaries <InlineMath math="d_s, d_e" /> and class probabilities. This
          anchor-free approach achieved state-of-the-art results on ActivityNet and THUMOS.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Temporal Action Detection Pipeline"
        code={`import torch
import torch.nn as nn

class TemporalActionDetector(nn.Module):
    def __init__(self, feat_dim=2048, num_classes=20, num_scales=5):
        super().__init__()
        # Multi-scale temporal feature pyramid
        self.pyramids = nn.ModuleList()
        for s in range(num_scales):
            self.pyramids.append(nn.Sequential(
                nn.Conv1d(feat_dim if s == 0 else feat_dim // 2,
                          feat_dim // 2, 3, stride=2, padding=1),
                nn.ReLU(),
            ))
        # Detection heads per scale
        self.cls_head = nn.Conv1d(feat_dim // 2, num_classes, 1)
        self.reg_head = nn.Conv1d(feat_dim // 2, 2, 1)  # start/end offsets

    def forward(self, features):  # [B, D, T]
        detections = []
        x = features
        for i, pyramid in enumerate(self.pyramids):
            x = pyramid(x)
            cls = self.cls_head(x)       # [B, C, T_i]
            reg = self.reg_head(x).relu()  # [B, 2, T_i]
            detections.append((cls, reg, x.shape[-1]))
        return detections

# Feature extraction (from pre-trained backbone)
feat_dim = 2048
T = 256  # temporal positions
features = torch.randn(2, feat_dim, T)

detector = TemporalActionDetector(feat_dim=feat_dim, num_classes=20)
outputs = detector(features)

for i, (cls, reg, t_len) in enumerate(outputs):
    print(f"Scale {i}: cls={cls.shape}, reg={reg.shape}, T={t_len}")

# Decode predictions from scale 0
cls_probs = outputs[0][0].softmax(dim=1)  # [B, C, T]
offsets = outputs[0][1]  # [B, 2, T]
print(f"\\nPredicted classes: {cls_probs.shape}")
print(f"Boundary offsets: {offsets.shape}")`}
      />

      <NoteBlock type="note" title="From Detection to Dense Captioning">
        <p>
          Dense video captioning extends temporal detection by generating natural language
          descriptions for each detected segment. Models like Vid2Seq unify temporal localization
          and captioning in a single sequence-to-sequence framework, predicting special time
          tokens interleaved with text tokens. This connects action detection with video-language understanding.
        </p>
      </NoteBlock>
    </div>
  )
}

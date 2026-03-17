import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function DatasetBenchmarks() {
  const [dataset, setDataset] = useState('kinetics400')

  const datasets = {
    kinetics400: { name: 'Kinetics-400', classes: 400, clips: '306K', duration: '~10s', source: 'YouTube', topMethod: 'VideoMAEv2 (90.0%)', challenge: 'Web noise, class imbalance' },
    kinetics700: { name: 'Kinetics-700', classes: 700, clips: '650K', duration: '~10s', source: 'YouTube', topMethod: 'InternVideo2 (83.7%)', challenge: 'Fine-grained + long-tail' },
    ssv2: { name: 'Something-Something v2', classes: 174, clips: '221K', duration: '2-6s', source: 'Crowdsourced', topMethod: 'VideoMAEv2 (77.0%)', challenge: 'Temporal reasoning required' },
    moments: { name: 'Moments in Time', classes: 339, clips: '1M', duration: '3s', source: 'Mixed', topMethod: 'SlowFast + NL (34.4%)', challenge: 'Multi-label, ambiguous actions' },
  }

  const d = datasets[dataset]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Action Classification Benchmarks</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.entries(datasets).map(([key, val]) => (
          <button key={key} onClick={() => setDataset(key)}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${dataset === key ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-3 gap-3 text-sm">
        {[['Classes', d.classes], ['Clips', d.clips], ['Duration', d.duration],
          ['Source', d.source], ['Top method', d.topMethod], ['Challenge', d.challenge]
        ].map(([label, val]) => (
          <div key={label} className="rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3">
            <p className="text-xs text-violet-600 dark:text-violet-400 font-semibold">{label}</p>
            <p className="text-gray-700 dark:text-gray-300">{val}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function ActionClassification() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Action classification assigns a single activity label to a trimmed video clip. While
        conceptually simple, it drives architecture innovation and serves as the primary
        benchmark for video understanding systems.
      </p>

      <DefinitionBlock title="Video Classification Pipeline">
        <p>A standard video classification system predicts:</p>
        <BlockMath math="P(y | V) = \text{softmax}(W \cdot \text{Pool}(f_\theta(V)) + b)" />
        <p className="mt-2">
          where <InlineMath math="V = \{x_1, \ldots, x_T\}" /> is a clip of <InlineMath math="T" /> frames,
          <InlineMath math="f_\theta" /> is the spatiotemporal backbone (3D CNN or Video Transformer),
          and Pool is global average pooling over space and time.
        </p>
      </DefinitionBlock>

      <DatasetBenchmarks />

      <TheoremBlock title="Appearance vs Temporal Reasoning" id="appearance-vs-temporal">
        <p>
          Datasets reveal different requirements. On Kinetics, a single frame achieves ~65%
          accuracy (appearance-biased). On Something-Something v2, temporal order matters:
        </p>
        <BlockMath math="\text{SSv2: } \text{Acc}_\text{1-frame} \approx 20\% \ll \text{Acc}_\text{video} \approx 77\%" />
        <p className="mt-1">
          Actions like "pushing something left to right" vs "pushing something right to left"
          require genuine temporal understanding, not just scene recognition. This makes SSv2
          the standard test for temporal modeling.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Multi-Crop Testing">
        <p>
          Standard practice for evaluation uses multiple temporal and spatial crops:
        </p>
        <BlockMath math="\hat{y} = \frac{1}{K}\sum_{k=1}^{K} f_\theta(\text{crop}_k(V))" />
        <p className="mt-1">
          Typically 4 temporal crops (uniformly spaced) x 3 spatial crops (left, center, right)
          = 12 views. Scores are averaged for final prediction. This adds 12x compute at
          inference but consistently improves accuracy by 1-3%.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Video Action Classification"
        code={`import torch
import torch.nn as nn

class VideoClassifier(nn.Module):
    def __init__(self, num_classes=400, backbone='r3d'):
        super().__init__()
        # 3D ResNet backbone (simplified)
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 128, (3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(128), nn.ReLU(),
            nn.Conv3d(128, 256, (3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(256), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, video):  # [B, C, T, H, W]
        features = self.features(video).flatten(1)
        return self.classifier(features)

# Training loop sketch
model = VideoClassifier(num_classes=400)
video = torch.randn(4, 3, 16, 224, 224)  # 4 clips, 16 frames
logits = model(video)
print(f"Logits: {logits.shape}")  # [4, 400]

# Multi-crop evaluation
def multi_crop_eval(model, video, num_temporal=4, num_spatial=3):
    """Average predictions over multiple crops."""
    T = video.shape[2]
    scores = []
    for t in range(num_temporal):
        start = t * (T - 16) // (num_temporal - 1) if num_temporal > 1 else 0
        clip = video[:, :, start:start+16]
        for s in range(num_spatial):
            # Spatial crops: left, center, right
            offset = s * (224 - 224) // max(num_spatial - 1, 1)
            crop = clip[:, :, :, :, offset:offset+224]
            with torch.no_grad():
                scores.append(model(crop).softmax(dim=-1))
    return torch.stack(scores).mean(dim=0)

avg_pred = multi_crop_eval(model, video)
print(f"Multi-crop prediction: {avg_pred.shape}")  # [4, 400]`}
      />

      <NoteBlock type="note" title="Foundation Models for Action Recognition">
        <p>
          Large pre-trained video models (InternVideo, VideoMAEv2) now dominate benchmarks by
          leveraging massive pre-training followed by fine-tuning. Zero-shot classification
          using video-language models (VideoCLIP, X-CLIP) is also competitive, classifying
          actions from text descriptions without any video-specific training labels.
        </p>
      </NoteBlock>
    </div>
  )
}

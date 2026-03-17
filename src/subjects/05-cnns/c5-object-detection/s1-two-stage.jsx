import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function DetectionPipeline() {
  const [stage, setStage] = useState(0)
  const stages = [
    { name: 'Backbone', desc: 'Extract feature maps from input image using a CNN (e.g., ResNet-50 FPN)', color: '#ddd6fe' },
    { name: 'Region Proposal Network', desc: 'Generate ~2000 candidate bounding boxes (proposals) with objectness scores', color: '#c4b5fd' },
    { name: 'RoI Pooling/Align', desc: 'Extract fixed-size features from each proposal region on the feature map', color: '#a78bfa' },
    { name: 'Classification + Regression', desc: 'Classify each proposal and refine bounding box coordinates', color: '#8b5cf6' },
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Two-Stage Detection Pipeline</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Stage: {stage + 1}/4
        <input type="range" min={0} max={3} value={stage} onChange={e => setStage(parseInt(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <div className="flex gap-2 mb-3">
        {stages.map((s, i) => (
          <div key={i} className="flex-1 p-2 rounded-lg text-center text-xs font-medium transition-all"
            style={{ backgroundColor: i <= stage ? s.color : '#f3f4f6', color: i <= stage ? '#4c1d95' : '#9ca3af', border: i === stage ? '2px solid #7c3aed' : '2px solid transparent' }}>
            {s.name}
          </div>
        ))}
      </div>
      <p className="text-sm text-gray-700 dark:text-gray-300 p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20">
        <strong>{stages[stage].name}:</strong> {stages[stage].desc}
      </p>
    </div>
  )
}

export default function TwoStageDetectors() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Two-stage detectors first generate region proposals, then classify and refine them.
        The R-CNN family (R-CNN, Fast R-CNN, Faster R-CNN) progressively improved this paradigm,
        culminating in end-to-end trainable systems with high accuracy.
      </p>

      <DefinitionBlock title="Faster R-CNN">
        <p>
          Faster R-CNN replaces selective search with a learned <strong>Region Proposal Network (RPN)</strong> that
          shares convolutional features with the detection network. At each spatial location, the RPN
          predicts <InlineMath math="k" /> anchor boxes with objectness scores:
        </p>
        <BlockMath math="\mathcal{L}_{\text{RPN}} = \frac{1}{N_{\text{cls}}} \sum_i \mathcal{L}_{\text{cls}}(p_i, p_i^*) + \lambda \frac{1}{N_{\text{reg}}} \sum_i p_i^* \mathcal{L}_{\text{reg}}(t_i, t_i^*)" />
      </DefinitionBlock>

      <DetectionPipeline />

      <TheoremBlock title="RoI Align" id="roi-align">
        <p>
          RoI Align (from Mask R-CNN) avoids quantization errors by using bilinear interpolation
          instead of rounding coordinates. For each bin in the output grid:
        </p>
        <BlockMath math="y = \sum_{i,j} \max(0, 1 - |x - x_i|) \cdot \max(0, 1 - |y - y_j|) \cdot f_{ij}" />
        <p className="mt-2">This preserves spatial precision crucial for pixel-level tasks like segmentation.</p>
      </TheoremBlock>

      <ExampleBlock title="R-CNN Family Evolution">
        <p>
          R-CNN (2014): CNN features + SVM, ~50s/image. Fast R-CNN (2015): shared features + RoI
          pooling, ~2s/image. Faster R-CNN (2015): learned RPN, ~0.2s/image (5 FPS). Each iteration
          moved more computation into the shared CNN backbone.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Faster R-CNN with torchvision"
        code={`import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

# Load pretrained Faster R-CNN
model = fasterrcnn_resnet50_fpn_v2(
    weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
model.eval()

# Inference on a dummy image
x = torch.randn(1, 3, 800, 800)
with torch.no_grad():
    predictions = model(x)

# predictions[0] contains boxes, labels, scores
print(f"Detected boxes: {predictions[0]['boxes'].shape}")
print(f"Labels: {predictions[0]['labels'][:5]}")
print(f"Scores: {predictions[0]['scores'][:5].numpy().round(3)}")

# Filter by confidence threshold
keep = predictions[0]['scores'] > 0.5
print(f"High-confidence detections: {keep.sum().item()}")`}
      />

      <NoteBlock type="note" title="Feature Pyramid Networks (FPN)">
        <p>
          FPN builds a multi-scale feature pyramid by combining top-down semantically strong features
          with bottom-up spatially precise features via lateral connections. This enables detecting
          objects at multiple scales efficiently and is now standard in modern detectors.
        </p>
      </NoteBlock>
    </div>
  )
}

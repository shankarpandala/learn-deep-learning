import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function DetectorComparison() {
  const [metric, setMetric] = useState('speed')
  const detectors = [
    { name: 'Faster R-CNN', speed: 15, map: 42.0, type: 'Two-stage' },
    { name: 'SSD-512', speed: 22, map: 28.8, type: 'One-stage' },
    { name: 'YOLOv3', speed: 30, map: 33.0, type: 'One-stage' },
    { name: 'RetinaNet', speed: 12, map: 40.4, type: 'One-stage' },
    { name: 'YOLOv8-L', speed: 80, map: 52.9, type: 'One-stage' },
    { name: 'FCOS', speed: 25, map: 41.5, type: 'Anchor-free' },
  ]
  const maxVal = metric === 'speed' ? 100 : 60

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Detector Comparison</h3>
      <div className="flex gap-3 mb-3">
        {['speed', 'accuracy'].map(m => (
          <label key={m} className="flex items-center gap-1.5 text-sm text-gray-600 dark:text-gray-400">
            <input type="radio" name="metric" checked={metric === m} onChange={() => setMetric(m)} className="accent-violet-500" />
            {m === 'speed' ? 'Speed (FPS)' : 'Accuracy (mAP)'}
          </label>
        ))}
      </div>
      <div className="space-y-2">
        {detectors.map((d, i) => {
          const val = metric === 'speed' ? d.speed : d.map
          return (
            <div key={i} className="flex items-center gap-2">
              <span className="text-xs w-24 text-gray-600 dark:text-gray-400">{d.name}</span>
              <div className="flex-1 bg-gray-100 dark:bg-gray-800 rounded-full h-4">
                <div className="h-4 rounded-full bg-violet-500 transition-all flex items-center justify-end pr-1"
                  style={{ width: `${(val / maxVal) * 100}%` }}>
                  <span className="text-[10px] text-white font-bold">{val}</span>
                </div>
              </div>
              <span className="text-[10px] text-gray-400 w-16">{d.type}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default function OneStageDetectors() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        One-stage detectors skip the region proposal step and predict bounding boxes and class
        probabilities directly from feature maps in a single pass. This makes them significantly
        faster while modern versions approach two-stage accuracy.
      </p>

      <DefinitionBlock title="YOLO (You Only Look Once)">
        <p>
          YOLO divides the image into an <InlineMath math="S \times S" /> grid. Each cell predicts{' '}
          <InlineMath math="B" /> bounding boxes with confidence and <InlineMath math="C" /> class
          probabilities. The output tensor has shape:
        </p>
        <BlockMath math="S \times S \times (B \times 5 + C)" />
        <p className="mt-2">Where each box predicts <InlineMath math="(x, y, w, h, \text{conf})" />.</p>
      </DefinitionBlock>

      <TheoremBlock title="Focal Loss (RetinaNet)" id="focal-loss">
        <p>
          One-stage detectors suffer from extreme foreground-background class imbalance. Focal loss
          down-weights easy negatives:
        </p>
        <BlockMath math="\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)" />
        <p className="mt-2">
          With <InlineMath math="\gamma = 2" />, a well-classified example with{' '}
          <InlineMath math="p_t = 0.9" /> gets <InlineMath math="100\times" /> less loss than a
          hard example with <InlineMath math="p_t = 0.1" />.
        </p>
      </TheoremBlock>

      <DetectorComparison />

      <ExampleBlock title="Anchor-Free Detection (FCOS)">
        <p>
          FCOS predicts, for each pixel in the feature map, its distance to the four sides of the
          bounding box: <InlineMath math="(l^*, t^*, r^*, b^*)" />. This eliminates anchor box
          hyperparameters and simplifies the detection pipeline. A centerness branch suppresses
          low-quality predictions far from object centers.
        </p>
      </ExampleBlock>

      <PythonCode
        title="YOLOv8 Inference with Ultralytics"
        code={`# pip install ultralytics
from ultralytics import YOLO
import torch

# Load pretrained YOLOv8
model = YOLO("yolov8n.pt")  # nano model for speed

# Inference
results = model("image.jpg")  # or pass a tensor/numpy array

# Access results
for result in results:
    boxes = result.boxes
    print(f"Detected {len(boxes)} objects")
    for box in boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        xyxy = box.xyxy[0].tolist()
        print(f"  Class {cls}, conf {conf:.2f}, box {xyxy}")

# Export to different formats
# model.export(format="onnx")  # ONNX
# model.export(format="tflite")  # TensorFlow Lite`}
      />

      <NoteBlock type="note" title="SSD: Multi-Scale Detection">
        <p>
          SSD (Single Shot Detector) predicts from multiple feature map scales simultaneously,
          using earlier (higher resolution) maps for small objects and later maps for large objects.
          This multi-scale approach was adopted by many subsequent architectures including
          RetinaNet and YOLO variants.
        </p>
      </NoteBlock>
    </div>
  )
}

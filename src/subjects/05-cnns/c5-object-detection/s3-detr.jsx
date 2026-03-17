import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function DETRArchitecture() {
  const [activeStep, setActiveStep] = useState(0)
  const steps = [
    { name: 'CNN Backbone', desc: 'ResNet-50 extracts features, producing a feature map of shape C x H x W', icon: 'B' },
    { name: 'Transformer Encoder', desc: '6 encoder layers with multi-head self-attention over flattened spatial features + positional encoding', icon: 'E' },
    { name: 'Transformer Decoder', desc: '6 decoder layers with N learned object queries attending to encoder output via cross-attention', icon: 'D' },
    { name: 'FFN Heads', desc: 'Parallel prediction heads output class label and bounding box (cx, cy, w, h) for each query', icon: 'H' },
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">DETR Architecture</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Component: {activeStep + 1}/4
        <input type="range" min={0} max={3} value={activeStep} onChange={e => setActiveStep(parseInt(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <div className="flex gap-2 mb-3 justify-center">
        {steps.map((s, i) => (
          <div key={i} className="flex flex-col items-center gap-1">
            <div className={`w-12 h-12 rounded-lg flex items-center justify-center text-lg font-bold transition-all ${i <= activeStep ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-400 dark:bg-gray-800'}`}>
              {s.icon}
            </div>
            {i < steps.length - 1 && <span className="text-gray-300">&darr;</span>}
          </div>
        ))}
      </div>
      <p className="text-sm text-gray-700 dark:text-gray-300 p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20">
        <strong>{steps[activeStep].name}:</strong> {steps[activeStep].desc}
      </p>
    </div>
  )
}

export default function DETR() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        DETR (Detection Transformer, Carion et al. 2020) reformulates object detection as a
        direct set prediction problem. It eliminates hand-designed components like anchors,
        NMS, and proposal generation, using a Transformer encoder-decoder with learned object queries.
      </p>

      <DefinitionBlock title="DETR Set Prediction">
        <p>
          DETR uses <InlineMath math="N" /> learned object queries (typically <InlineMath math="N = 100" />),
          each producing one prediction. Hungarian matching finds the optimal one-to-one assignment
          between predictions and ground truth:
        </p>
        <BlockMath math="\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^{N} \mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)})" />
      </DefinitionBlock>

      <DETRArchitecture />

      <TheoremBlock title="Hungarian Matching Loss" id="hungarian-loss">
        <p>The matching cost combines classification and box regression:</p>
        <BlockMath math="\mathcal{L}_{\text{match}} = -\mathbb{1}_{c_i \neq \varnothing} \hat{p}_{\sigma(i)}(c_i) + \mathbb{1}_{c_i \neq \varnothing} \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\sigma(i)})" />
        <p className="mt-2">
          Box loss uses a combination of L1 loss and generalized IoU:
        </p>
        <BlockMath math="\mathcal{L}_{\text{box}} = \lambda_{\text{iou}} \mathcal{L}_{\text{GIoU}} + \lambda_{\text{L1}} \| b - \hat{b} \|_1" />
      </TheoremBlock>

      <ExampleBlock title="DETR Advantages">
        <p>
          DETR eliminates: anchor box design, NMS post-processing, and complex multi-stage pipelines.
          It achieves 42.0 mAP on COCO (comparable to Faster R-CNN) with a dramatically simpler
          architecture. It excels at detecting large objects due to global attention.
        </p>
      </ExampleBlock>

      <PythonCode
        title="DETR Inference with Hugging Face"
        code={`import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

# Load pretrained DETR
processor = DetrImageProcessor.from_pretrained(
    "facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50")
model.eval()

# Dummy input (normally use a real image)
dummy_image = torch.randn(3, 800, 1200)
inputs = processor(images=dummy_image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Post-process: convert to boxes and labels
target_sizes = torch.tensor([[800, 1200]])
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.7)

print(f"Detections: {len(results[0]['boxes'])}")
for score, label, box in zip(
    results[0]["scores"], results[0]["labels"], results[0]["boxes"]):
    print(f"  {model.config.id2label[label.item()]}: "
          f"{score:.2f}, box={box.tolist()}")`}
      />

      <NoteBlock type="note" title="DETR Variants">
        <p>
          Deformable DETR addresses slow convergence by replacing global attention with deformable
          attention, attending to a small set of key sampling points. DINO-DETR adds contrastive
          denoising and anchor denoising for faster convergence and improved accuracy, achieving
          63.3 mAP on COCO.
        </p>
      </NoteBlock>
    </div>
  )
}

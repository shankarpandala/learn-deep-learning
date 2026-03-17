import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function SAMPromptDemo() {
  const [promptType, setPromptType] = useState('point')
  const [clickPos, setClickPos] = useState({ x: 140, y: 90 })
  const W = 280, H = 180
  const segCenter = { x: 140, y: 90 }

  const handleClick = (e) => {
    const r = e.currentTarget.getBoundingClientRect()
    setClickPos({ x: e.clientX - r.left, y: e.clientY - r.top })
  }

  const dist = Math.sqrt((clickPos.x - segCenter.x) ** 2 + (clickPos.y - segCenter.y) ** 2)
  const maskVisible = promptType === 'point' ? dist < 80 : true

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">SAM Prompt Types</h3>
      <div className="flex gap-2 mb-3">
        {['point', 'box', 'auto'].map(t => (
          <button key={t} onClick={() => setPromptType(t)}
            className={`px-3 py-1 rounded text-sm capitalize ${promptType === t ? 'bg-violet-500 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300'}`}>
            {t}
          </button>
        ))}
      </div>
      <svg width={W} height={H} className="mx-auto block bg-gray-50 dark:bg-gray-800 rounded cursor-crosshair" onClick={handleClick}>
        {maskVisible && <ellipse cx={segCenter.x} cy={segCenter.y} rx={60} ry={45} fill="#8b5cf6" opacity={0.25} stroke="#8b5cf6" strokeWidth={1.5} />}
        {promptType === 'point' && (
          <>
            <circle cx={clickPos.x} cy={clickPos.y} r={5} fill="#22c55e" stroke="white" strokeWidth={1.5} />
            <text x={clickPos.x + 8} y={clickPos.y + 4} fontSize={10} fill="#22c55e">click</text>
          </>
        )}
        {promptType === 'box' && (
          <rect x={80} y={45} width={120} height={90} fill="none" stroke="#f97316" strokeWidth={2} strokeDasharray="4,2" />
        )}
        {promptType === 'auto' && (
          <>
            <ellipse cx={80} cy={70} rx={30} ry={25} fill="#8b5cf6" opacity={0.2} stroke="#8b5cf6" strokeWidth={1} />
            <ellipse cx={200} cy={110} rx={35} ry={28} fill="#f97316" opacity={0.2} stroke="#f97316" strokeWidth={1} />
            <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={10} fill="#6b7280">Automatic everything mode</text>
          </>
        )}
      </svg>
    </div>
  )
}

export default function SAM() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The Segment Anything Model (SAM) is a foundation model for image segmentation that can
        segment any object given a prompt. Trained on 1B+ masks, it generalizes to unseen domains.
      </p>

      <DefinitionBlock title="SAM Architecture">
        <p>SAM consists of three components:</p>
        <BlockMath math="\text{Image} \xrightarrow{\text{ViT Encoder}} \mathbf{E} \xrightarrow[\text{Prompt}]{\text{Decoder}} \text{Masks}" />
        <p className="mt-2">
          The image encoder runs once, then the lightweight mask decoder produces masks for
          any number of prompts (points, boxes, or text) in real-time.
        </p>
      </DefinitionBlock>

      <SAMPromptDemo />

      <TheoremBlock title="Promptable Segmentation" id="promptable-seg">
        <p>SAM's mask decoder uses cross-attention between prompt tokens and image embeddings:</p>
        <BlockMath math="\mathbf{Q} = \text{Prompt Tokens}, \quad \mathbf{K} = \mathbf{V} = \mathbf{E}_{\text{img}}" />
        <BlockMath math="\text{Mask} = \text{MLP}\!\left(\text{CrossAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V})\right) \cdot \mathbf{E}_{\text{img}}" />
        <p className="mt-1">
          The output is a dot product between updated prompt tokens and image embeddings,
          producing <InlineMath math="256 \times 256" /> mask logits per prompt.
        </p>
      </TheoremBlock>

      <ExampleBlock title="SAM Training Data (SA-1B)">
        <ul className="list-disc ml-5 space-y-1">
          <li>11 million images with 1.1 billion masks</li>
          <li>Three-stage annotation: assisted-manual, semi-automatic, fully automatic</li>
          <li>99.1% mask quality measured by human evaluation</li>
          <li>400x more masks than any previous segmentation dataset</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Using SAM for Segmentation"
        code={`import torch
from segment_anything import sam_model_registry, SamPredictor

# Load SAM model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

# Set image (encodes once with ViT)
image = load_image("photo.jpg")  # (H, W, 3) numpy
predictor.set_image(image)

# Point prompt: positive click at (x=500, y=300)
masks, scores, logits = predictor.predict(
    point_coords=torch.tensor([[500, 300]]),
    point_labels=torch.tensor([1]),  # 1=foreground
    multimask_output=True,  # Returns 3 mask candidates
)
best_mask = masks[scores.argmax()]  # Pick highest IoU

# Box prompt
masks_box, _, _ = predictor.predict(
    box=torch.tensor([100, 100, 400, 350]),  # xyxy
    multimask_output=False,
)

# Automatic mask generation (segment everything)
from segment_anything import SamAutomaticMaskGenerator
generator = SamAutomaticMaskGenerator(sam)
all_masks = generator.generate(image)
print(f"Found {len(all_masks)} segments")`}
      />

      <NoteBlock type="note" title="SAM 2 and Video Segmentation">
        <p>
          SAM 2 extends the foundation model paradigm to video, tracking segments across frames
          with memory-based attention. It handles occlusions, appearance changes, and new
          objects appearing mid-video. The streaming architecture processes frames sequentially
          while maintaining a memory bank of past predictions and image features.
        </p>
      </NoteBlock>
    </div>
  )
}

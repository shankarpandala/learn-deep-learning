import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function HungarianDemo() {
  const [step, setStep] = useState(0)
  const W = 300, H = 160
  const predictions = [
    { x: 50, y: 50, label: 'q1' }, { x: 150, y: 80, label: 'q2' },
    { x: 250, y: 50, label: 'q3' }, { x: 100, y: 120, label: 'q4' },
  ]
  const gts = [
    { x: 60, y: 60, label: 'gt1' }, { x: 240, y: 55, label: 'gt2' },
  ]
  const matches = step >= 1 ? [[0, 0], [2, 1]] : []

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Hungarian Matching</h3>
      <div className="flex gap-2 mb-3">
        {['Queries + GT', 'Bipartite Match'].map((label, i) => (
          <button key={i} onClick={() => setStep(i)}
            className={`px-3 py-1 rounded text-sm ${step === i ? 'bg-violet-500 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300'}`}>
            {label}
          </button>
        ))}
      </div>
      <svg width={W} height={H} className="mx-auto block">
        {matches.map(([pi, gi], i) => (
          <line key={i} x1={predictions[pi].x} y1={predictions[pi].y} x2={gts[gi].x} y2={gts[gi].y}
            stroke="#22c55e" strokeWidth={2} strokeDasharray="4,2" />
        ))}
        {predictions.map((p, i) => {
          const matched = matches.some(([pi]) => pi === i)
          return (
            <g key={`p${i}`}>
              <circle cx={p.x} cy={p.y} r={8} fill={matched ? '#8b5cf6' : '#d1d5db'} opacity={0.8} />
              <text x={p.x} y={p.y + 3} textAnchor="middle" fontSize={8} fill="white">{p.label}</text>
            </g>
          )
        })}
        {gts.map((g, i) => (
          <g key={`g${i}`}>
            <rect x={g.x - 10} y={g.y - 10} width={20} height={20} fill="none" stroke="#f97316" strokeWidth={2} />
            <text x={g.x} y={g.y + 25} textAnchor="middle" fontSize={9} fill="#f97316">{g.label}</text>
          </g>
        ))}
        <text x={10} y={H - 5} fontSize={9} fill="#6b7280">
          {step === 0 ? 'N queries, M ground truths' : 'Optimal 1-to-1 assignment (unmatched = no-object)'}
        </text>
      </svg>
    </div>
  )
}

export default function DetectionTransformers() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        DETR (DEtection TRansformer) reformulates object detection as a direct set prediction
        problem, eliminating anchor boxes, NMS, and hand-designed components entirely.
      </p>

      <DefinitionBlock title="DETR Architecture">
        <p>DETR processes images through a CNN backbone, transformer encoder-decoder, and prediction heads:</p>
        <BlockMath math="\text{Image} \xrightarrow{\text{CNN}} \mathbf{F} \xrightarrow{\text{Encoder}} \mathbf{Z} \xrightarrow[\text{Object Queries}]{\text{Decoder}} \{(\hat{c}_i, \hat{b}_i)\}_{i=1}^{N}" />
        <p className="mt-2">
          <InlineMath math="N" /> learnable object queries attend to image features via cross-attention,
          each predicting a class <InlineMath math="\hat{c}" /> and box <InlineMath math="\hat{b}" /> (or "no object").
        </p>
      </DefinitionBlock>

      <HungarianDemo />

      <TheoremBlock title="Bipartite Matching Loss" id="hungarian">
        <p>DETR finds the optimal one-to-one assignment between predictions and ground truth:</p>
        <BlockMath math="\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^{N} \mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)})" />
        <p className="mt-1">where the matching cost combines classification and box terms:</p>
        <BlockMath math="\mathcal{L}_{\text{match}} = -\mathbb{1}_{c_i \neq \varnothing}\hat{p}_{\sigma(i)}(c_i) + \mathbb{1}_{c_i \neq \varnothing}\mathcal{L}_{\text{box}}(b_i, \hat{b}_{\sigma(i)})" />
        <p className="mt-1">
          Solved efficiently using the Hungarian algorithm in <InlineMath math="\mathcal{O}(N^3)" />.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Deformable DETR Improvements">
        <ul className="list-disc ml-5 space-y-1">
          <li><strong>Deformable attention</strong>: attends to a small set of sampling points instead of all tokens</li>
          <li><strong>Multi-scale features</strong>: processes FPN features at multiple resolutions</li>
          <li><strong>10x faster convergence</strong>: 50 epochs vs 500 for vanilla DETR</li>
          <li><strong>Better small object detection</strong>: multi-scale attention at high-res features</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="DETR-Style Detection"
        code={`import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

class DETRHead(nn.Module):
    def __init__(self, d_model=256, num_classes=91, num_queries=100):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=1024,
            batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.class_head = nn.Linear(d_model, num_classes + 1)  # +1 for no-obj
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 4),  # (cx, cy, w, h) normalized
        )

    def forward(self, encoder_output):
        B = encoder_output.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        hs = self.decoder(queries, encoder_output)
        return self.class_head(hs), self.box_head(hs).sigmoid()

def hungarian_loss(pred_cls, pred_box, gt_cls, gt_box):
    """Compute loss with Hungarian matching."""
    # Cost matrix: classification + L1 + GIoU
    cost_cls = -pred_cls.softmax(-1)[..., gt_cls]  # (N, M)
    cost_box = torch.cdist(pred_box, gt_box, p=1)
    cost = cost_cls + 5 * cost_box
    # Hungarian matching
    row_idx, col_idx = linear_sum_assignment(cost.detach().cpu())
    return row_idx, col_idx  # Matched indices`}
      />

      <NoteBlock type="note" title="End-to-End Detection">
        <p>
          DETR's key insight is replacing hand-designed components (anchors, NMS, proposal
          generation) with learned set prediction. This simplifies the pipeline but
          originally required very long training (500 epochs). Deformable DETR, DAB-DETR,
          and DINO have progressively addressed convergence speed, achieving state-of-the-art
          results (63.3 AP on COCO) with practical training schedules.
        </p>
      </NoteBlock>
    </div>
  )
}

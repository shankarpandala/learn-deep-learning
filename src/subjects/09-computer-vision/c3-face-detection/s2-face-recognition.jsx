import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function EmbeddingDemo() {
  const [margin, setMargin] = useState(0.3)
  const W = 280, H = 200
  const anchor = { x: 100, y: 100 }
  const positive = { x: 130, y: 80 }
  const negative = { x: 200, y: 140 }
  const dAP = Math.sqrt((anchor.x - positive.x) ** 2 + (anchor.y - positive.y) ** 2)
  const dAN = Math.sqrt((anchor.x - negative.x) ** 2 + (anchor.y - negative.y) ** 2)
  const marginPixels = margin * 200

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Triplet Loss Embedding Space</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Margin: {margin.toFixed(2)}
        <input type="range" min={0.05} max={0.8} step={0.05} value={margin}
          onChange={e => setMargin(parseFloat(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <circle cx={anchor.x} cy={anchor.y} r={marginPixels} fill="none" stroke="#8b5cf6" strokeWidth={1} strokeDasharray="4,3" opacity={0.5} />
        <line x1={anchor.x} y1={anchor.y} x2={positive.x} y2={positive.y} stroke="#22c55e" strokeWidth={1.5} />
        <line x1={anchor.x} y1={anchor.y} x2={negative.x} y2={negative.y} stroke="#ef4444" strokeWidth={1.5} />
        <circle cx={anchor.x} cy={anchor.y} r={6} fill="#8b5cf6" />
        <circle cx={positive.x} cy={positive.y} r={6} fill="#22c55e" />
        <circle cx={negative.x} cy={negative.y} r={6} fill="#ef4444" />
        <text x={anchor.x - 20} y={anchor.y + 18} fontSize={10} fill="#8b5cf6">Anchor</text>
        <text x={positive.x - 5} y={positive.y - 10} fontSize={10} fill="#22c55e">Positive</text>
        <text x={negative.x - 5} y={negative.y + 16} fontSize={10} fill="#ef4444">Negative</text>
      </svg>
      <p className="mt-1 text-center text-xs text-gray-500">
        d(A,P) = {(dAP / 200).toFixed(2)} | d(A,N) = {(dAN / 200).toFixed(2)} | loss = {Math.max(0, dAP / 200 - dAN / 200 + margin).toFixed(3)}
      </p>
    </div>
  )
}

export default function FaceRecognition() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Face recognition maps face images to compact embedding vectors where distance reflects
        identity similarity. Modern systems use metric learning losses to achieve superhuman accuracy.
      </p>

      <DefinitionBlock title="Triplet Loss (FaceNet)">
        <p>
          Given an anchor <InlineMath math="a" />, positive <InlineMath math="p" /> (same identity),
          and negative <InlineMath math="n" /> (different identity):
        </p>
        <BlockMath math="\mathcal{L}_{\text{triplet}} = \max\!\left(0,\; \|f(a) - f(p)\|^2 - \|f(a) - f(n)\|^2 + m\right)" />
        <p className="mt-2">
          The margin <InlineMath math="m" /> enforces a minimum gap between positive and negative pairs
          in the embedding space.
        </p>
      </DefinitionBlock>

      <EmbeddingDemo />

      <TheoremBlock title="ArcFace Angular Margin" id="arcface">
        <p>ArcFace adds an angular margin to the softmax classification loss:</p>
        <BlockMath math="\mathcal{L}_{\text{arc}} = -\log \frac{e^{s \cos(\theta_{y_i} + m)}}{e^{s \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cos \theta_j}}" />
        <p className="mt-1">
          where <InlineMath math="\theta_j = \arccos(W_j^T f(x))" /> is the angle between the
          feature and class center, <InlineMath math="s" /> is a scale factor, and <InlineMath math="m" /> is the additive angular margin.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Face Verification Pipeline">
        <ol className="list-decimal ml-5 space-y-1">
          <li>Detect and align faces using landmarks (5-point alignment)</li>
          <li>Extract 512-d embedding: <InlineMath math="v = f_\theta(\text{face})" /></li>
          <li>L2-normalize: <InlineMath math="\hat{v} = v / \|v\|" /></li>
          <li>Compare cosine similarity: <InlineMath math="\text{sim} = \hat{v}_1 \cdot \hat{v}_2" /></li>
          <li>Threshold at <InlineMath math="\tau \approx 0.4" /> for same/different identity</li>
        </ol>
      </ExampleBlock>

      <PythonCode
        title="ArcFace Loss Implementation"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    def __init__(self, embed_dim=512, num_classes=10000,
                 s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, embeddings, labels):
        # Normalize weights and features
        W = F.normalize(self.W, dim=1)
        x = F.normalize(embeddings, dim=1)

        # Compute cos(theta)
        cosine = x @ W.T  # (B, num_classes)

        # Add angular margin to target class
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        target_logits = torch.cos(theta[range(len(labels)), labels] + self.m)
        cosine[range(len(labels)), labels] = target_logits

        # Scale and compute cross-entropy
        logits = cosine * self.s
        return F.cross_entropy(logits, labels)

# Usage
loss_fn = ArcFaceLoss(embed_dim=512, num_classes=85742)
embeddings = backbone(face_images)  # (B, 512)
loss = loss_fn(embeddings, identity_labels)`}
      />

      <NoteBlock type="note" title="Hard Mining Strategies">
        <p>
          Training efficiency depends heavily on selecting informative triplets. Online hard mining
          selects the hardest positive (farthest same-identity) and hardest negative (closest
          different-identity) within each mini-batch. Semi-hard mining selects negatives that
          are farther than the positive but still within the margin, providing more stable gradients.
        </p>
      </NoteBlock>
    </div>
  )
}

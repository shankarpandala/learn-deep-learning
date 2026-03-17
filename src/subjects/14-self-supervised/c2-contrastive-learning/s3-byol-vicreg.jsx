import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function MethodComparisonViz() {
  const [selected, setSelected] = useState('byol')
  const methods = {
    byol: { name: 'BYOL', negatives: 'No', momentum: 'Yes', predictor: 'Yes', key: 'Asymmetric architecture + momentum prevents collapse' },
    simsiam: { name: 'SimSiam', negatives: 'No', momentum: 'No', predictor: 'Yes', key: 'Stop-gradient alone prevents collapse (surprisingly)' },
    vicreg: { name: 'VICReg', negatives: 'No', momentum: 'No', predictor: 'No', key: 'Variance + covariance regularization prevents collapse' },
    barlow: { name: 'Barlow Twins', negatives: 'No', momentum: 'No', predictor: 'No', key: 'Cross-correlation matrix should equal identity' },
  }
  const m = methods[selected]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-2 text-base font-bold text-gray-800 dark:text-gray-200">Non-Contrastive Methods</h3>
      <div className="flex gap-2 mb-3 flex-wrap">
        {Object.entries(methods).map(([key, v]) => (
          <button key={key} onClick={() => setSelected(key)}
            className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${selected === key ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400'}`}>
            {v.name}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-3 gap-2 text-xs bg-gray-50 dark:bg-gray-800 rounded-lg p-3">
        <div><span className="text-gray-500">Negatives:</span> <span className={`font-medium ${m.negatives === 'No' ? 'text-violet-600' : ''}`}>{m.negatives}</span></div>
        <div><span className="text-gray-500">Momentum:</span> <span className="font-medium">{m.momentum}</span></div>
        <div><span className="text-gray-500">Predictor:</span> <span className="font-medium">{m.predictor}</span></div>
      </div>
      <p className="text-xs text-violet-600 mt-2 text-center font-medium">{m.key}</p>
    </div>
  )
}

export default function BYOLVICReg() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Non-contrastive methods learn representations without negative pairs. BYOL showed that a
        momentum encoder with an asymmetric predictor suffices, while VICReg uses explicit
        regularization of the embedding statistics.
      </p>

      <DefinitionBlock title="BYOL (Bootstrap Your Own Latent)">
        <p>
          BYOL uses an online network <InlineMath math="(\theta)" /> with a predictor and a target network
          <InlineMath math="(\xi)" /> updated via momentum:
        </p>
        <BlockMath math="\mathcal{L}_{\text{BYOL}} = \|\bar{q}_\theta(\mathbf{z}_\theta) - \bar{\mathbf{z}}_\xi\|^2" />
        <p className="mt-2">
          where <InlineMath math="\bar{\cdot}" /> denotes L2 normalization, <InlineMath math="q_\theta" /> is the predictor,
          and the loss is symmetrized over both views. No negatives needed.
        </p>
      </DefinitionBlock>

      <MethodComparisonViz />

      <ExampleBlock title="SimSiam: Simplicity Is All You Need">
        <p>
          SimSiam removes even the momentum encoder. The key insight is that stop-gradient on the
          target branch creates an implicit moving average:
        </p>
        <BlockMath math="\mathcal{L} = -\frac{1}{2}\left[\text{sim}(p_1, \text{sg}(z_2)) + \text{sim}(p_2, \text{sg}(z_1))\right]" />
        <p className="mt-1">
          where <InlineMath math="\text{sg}" /> is stop-gradient. Without it, instant collapse occurs.
        </p>
      </ExampleBlock>

      <PythonCode
        title="BYOL Implementation"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class BYOL(nn.Module):
    def __init__(self, encoder, proj_dim=256, pred_dim=128, momentum=0.996):
        super().__init__()
        self.momentum = momentum

        # Online network
        self.encoder = encoder
        feat_dim = 512  # encoder output dim
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim), nn.BatchNorm1d(pred_dim), nn.ReLU(),
            nn.Linear(pred_dim, proj_dim),
        )

        # Target network (no gradients)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_projector = copy.deepcopy(self.projector)
        for p in list(self.target_encoder.parameters()) + list(self.target_projector.parameters()):
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self):
        for po, pt in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            pt.data = self.momentum * pt.data + (1 - self.momentum) * po.data
        for po, pt in zip(self.projector.parameters(), self.target_projector.parameters()):
            pt.data = self.momentum * pt.data + (1 - self.momentum) * po.data

    def forward(self, x1, x2):
        # Online predictions
        p1 = self.predictor(self.projector(self.encoder(x1)))
        p2 = self.predictor(self.projector(self.encoder(x2)))

        with torch.no_grad():
            self.update_target()
            t1 = self.target_projector(self.target_encoder(x1))
            t2 = self.target_projector(self.target_encoder(x2))

        loss = (F.cosine_similarity(p1, t2.detach(), dim=-1).mean()
              + F.cosine_similarity(p2, t1.detach(), dim=-1).mean())
        return -loss  # maximize cosine similarity

print("BYOL: No negatives, no large batches needed")
print("Key: predictor + momentum encoder = implicit regularization")`}
      />

      <WarningBlock title="Batch Normalization in BYOL">
        <p>
          Early analysis suggested BYOL's success depended critically on batch normalization in the
          projector, which implicitly provides negative-like information across the batch. Later
          work showed that BYOL can work without BN if the predictor is properly initialized and
          the learning rate is carefully tuned.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Which Method to Choose?">
        <p>
          For Vision Transformers: DINO/DINOv2 (momentum + self-distillation) dominates.
          For CNNs with limited compute: VICReg or Barlow Twins (simple, no momentum).
          For large-scale with TPUs: SimCLR v2 remains competitive with large batches.
          All methods achieve similar linear probe accuracy on ImageNet (~75% with ResNet-50).
        </p>
      </NoteBlock>
    </div>
  )
}

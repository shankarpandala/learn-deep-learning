import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function FineTuneVisualizer() {
  const [strategy, setStrategy] = useState('full')
  const layers = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'FC']
  const frozen = {
    full: [],
    'feature-extract': ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5'],
    'gradual-unfreeze': ['Conv1', 'Conv2', 'Conv3'],
  }

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Fine-Tuning Strategy</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {[['full', 'Full Fine-Tune'], ['feature-extract', 'Feature Extract'], ['gradual-unfreeze', 'Gradual Unfreeze']].map(([k, l]) => (
          <button key={k} onClick={() => setStrategy(k)}
            className={`px-3 py-1 rounded text-sm ${strategy === k ? 'bg-violet-500 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300'}`}>
            {l}
          </button>
        ))}
      </div>
      <div className="flex justify-center gap-2">
        {layers.map(layer => {
          const isFrozen = frozen[strategy].includes(layer)
          return (
            <div key={layer} className="flex flex-col items-center gap-1">
              <div className={`w-14 h-10 rounded flex items-center justify-center text-xs font-mono text-white ${isFrozen ? 'bg-gray-400' : 'bg-violet-500'}`}>
                {layer}
              </div>
              <span className="text-xs text-gray-500">{isFrozen ? 'frozen' : 'train'}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default function TransferLearning() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Transfer learning leverages features learned on large-scale datasets (like ImageNet) and
        adapts them to new tasks with limited data, dramatically reducing training time and data requirements.
      </p>

      <DefinitionBlock title="Transfer Learning">
        <p>
          Given a source model <InlineMath math="f_\theta" /> trained on task <InlineMath math="\mathcal{T}_s" />,
          transfer learning adapts parameters to a target task <InlineMath math="\mathcal{T}_t" />:
        </p>
        <BlockMath math="\theta^* = \arg\min_\theta \mathcal{L}_t(f_\theta) \quad \text{initialized from } \theta_s" />
        <p className="mt-2">Lower layers learn generic features (edges, textures) that transfer well across domains.</p>
      </DefinitionBlock>

      <FineTuneVisualizer />

      <TheoremBlock title="Domain Shift Bound" id="domain-shift">
        <p>
          The target risk is bounded by the source risk plus domain divergence:
        </p>
        <BlockMath math="\epsilon_t(h) \leq \epsilon_s(h) + d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_s, \mathcal{D}_t) + \lambda" />
        <p className="mt-1">
          where <InlineMath math="d_{\mathcal{H}\Delta\mathcal{H}}" /> measures the divergence between
          source and target distributions, and <InlineMath math="\lambda" /> is the ideal joint error.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Fine-Tuning Learning Rates">
        <p>Discriminative learning rates assign different rates per layer group:</p>
        <BlockMath math="\eta_l = \eta_{\text{base}} \cdot \gamma^{L - l}" />
        <p className="mt-1">
          With <InlineMath math="\gamma = 0.1" />, early layers train at 100x smaller learning rate
          than the classification head, preserving pretrained features.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Transfer Learning with PyTorch"
        code={`import torch
import torch.nn as nn
from torchvision import models

# Load pretrained ResNet-50
model = models.resnet50(weights='IMAGENET1K_V2')

# Strategy 1: Feature extraction (freeze backbone)
for param in model.parameters():
    param.requires_grad = False

# Replace classifier for new task (10 classes)
model.fc = nn.Sequential(
    nn.Linear(2048, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10),
)

# Strategy 2: Discriminative learning rates
param_groups = [
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3},
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

# Gradual unfreezing after warmup
def unfreeze_layer(model, layer_name):
    for name, param in model.named_parameters():
        if layer_name in name:
            param.requires_grad = True`}
      />

      <NoteBlock type="note" title="When to Fine-Tune vs Feature Extract">
        <p>
          <strong>Feature extraction</strong> works well when the target dataset is small and similar
          to the source. <strong>Full fine-tuning</strong> is preferred when you have sufficient target
          data or the domains differ significantly. Gradual unfreezing offers a middle ground
          that often achieves the best results on medium-sized datasets.
        </p>
      </NoteBlock>
    </div>
  )
}

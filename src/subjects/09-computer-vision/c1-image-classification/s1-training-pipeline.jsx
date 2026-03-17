import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function AugmentationDemo() {
  const [augType, setAugType] = useState('original')
  const transforms = {
    original: { label: 'Original', tx: 0, ty: 0, scale: 1, rotate: 0, opacity: 1 },
    flip: { label: 'Horizontal Flip', tx: 0, ty: 0, scale: -1, rotate: 0, opacity: 1 },
    rotate: { label: 'Rotation (+15)', tx: 0, ty: 0, scale: 1, rotate: 15, opacity: 1 },
    crop: { label: 'Random Crop', tx: 10, ty: 10, scale: 1.3, rotate: 0, opacity: 1 },
    color: { label: 'Color Jitter', tx: 0, ty: 0, scale: 1, rotate: 0, opacity: 0.7 },
  }
  const t = transforms[augType]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Data Augmentation Preview</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.entries(transforms).map(([key, val]) => (
          <button key={key} onClick={() => setAugType(key)}
            className={`px-3 py-1 rounded text-sm ${augType === key ? 'bg-violet-500 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300'}`}>
            {val.label}
          </button>
        ))}
      </div>
      <svg width={200} height={200} className="mx-auto block border border-gray-200 dark:border-gray-700 rounded">
        <g transform={`translate(100,100) rotate(${t.rotate}) scale(${t.scale}) translate(${t.tx},${t.ty})`} opacity={t.opacity}>
          <rect x={-40} y={-40} width={80} height={80} fill="#8b5cf6" rx={4} />
          <circle cx={-10} cy={-10} r={8} fill="#fbbf24" />
          <polygon points="0,10 20,-15 40,10" fill="#34d399" />
        </g>
      </svg>
      <p className="mt-2 text-center text-sm text-gray-500 dark:text-gray-400">Transform: {t.label}</p>
    </div>
  )
}

export default function TrainingPipeline() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        A robust image classification pipeline covers data loading, augmentation, model training,
        and evaluation. Each stage has a significant impact on final accuracy and generalization.
      </p>

      <DefinitionBlock title="Standard Training Pipeline">
        <p>The pipeline consists of sequential stages:</p>
        <BlockMath math="\text{Data} \xrightarrow{\text{augment}} \text{Batch} \xrightarrow{\text{forward}} \hat{y} \xrightarrow{\mathcal{L}} \text{loss} \xrightarrow{\nabla} \text{update}" />
        <p className="mt-2">
          The cross-entropy loss for <InlineMath math="C" /> classes is:
        </p>
        <BlockMath math="\mathcal{L} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)" />
      </DefinitionBlock>

      <AugmentationDemo />

      <ExampleBlock title="Common Augmentation Strategy">
        <p>For ImageNet-scale training, a typical augmentation stack includes:</p>
        <ul className="list-disc ml-5 mt-2 space-y-1">
          <li>Random resized crop to <InlineMath math="224 \times 224" /></li>
          <li>Horizontal flip with <InlineMath math="p = 0.5" /></li>
          <li>Color jitter (brightness, contrast, saturation)</li>
          <li>RandAugment or AutoAugment policies</li>
          <li>Mixup / CutMix regularization</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="PyTorch Training Pipeline"
        code={`import torch
import torch.nn as nn
from torchvision import transforms, datasets, models

# Data augmentation pipeline
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_set = datasets.ImageFolder('data/train', train_transform)
loader = torch.utils.data.DataLoader(
    train_set, batch_size=64, shuffle=True, num_workers=4)

model = models.resnet50(pretrained=False, num_classes=10)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Training loop
for epoch in range(100):
    model.train()
    for images, labels in loader:
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: loss={loss.item():.4f}")`}
      />

      <WarningBlock title="Label Smoothing">
        <p>
          Hard one-hot labels can cause overconfident predictions. Label smoothing
          replaces the target <InlineMath math="y_c" /> with:
        </p>
        <BlockMath math="y_c' = (1 - \epsilon) \cdot y_c + \frac{\epsilon}{C}" />
        <p className="mt-1">
          Typical <InlineMath math="\epsilon = 0.1" />. This improves calibration and generalization.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Learning Rate Scheduling">
        <p>
          Cosine annealing is the most popular schedule for image classification. It
          decays the learning rate from <InlineMath math="\eta_{\max}" /> to <InlineMath math="\eta_{\min}" /> following:
        </p>
        <BlockMath math="\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t\pi}{T}\right)\right)" />
        <p className="mt-1">Warm-up for the first 5-10 epochs stabilizes early training.</p>
      </NoteBlock>
    </div>
  )
}

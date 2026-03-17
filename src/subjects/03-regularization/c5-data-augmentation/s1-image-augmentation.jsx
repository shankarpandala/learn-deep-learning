import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function AugmentationDemo() {
  const [augType, setAugType] = useState('none')
  const W = 200, H = 200

  const basePixels = Array.from({ length: 8 }, (_, r) =>
    Array.from({ length: 8 }, (_, c) => {
      const dist = Math.sqrt((r - 3.5) ** 2 + (c - 3.5) ** 2)
      return dist < 2.5 ? '#8b5cf6' : dist < 3.5 ? '#c4b5fd' : '#ede9fe'
    })
  )

  const applyAug = (r, c, color) => {
    const cellSize = W / 8
    let x = c * cellSize, y = r * cellSize
    let w = cellSize, h = cellSize
    let finalColor = color
    let opacity = 1

    if (augType === 'hflip') { x = W - (c + 1) * cellSize }
    if (augType === 'vflip') { y = H - (r + 1) * cellSize }
    if (augType === 'crop') {
      if (r < 1 || r > 6 || c < 1 || c > 6) opacity = 0.15
    }
    if (augType === 'jitter') {
      const shift = Math.sin(r * 3 + c * 7) * 30
      const base = parseInt(color.slice(1), 16)
      const rr = Math.min(255, Math.max(0, ((base >> 16) & 0xFF) + shift))
      const gg = Math.min(255, Math.max(0, ((base >> 8) & 0xFF) + shift))
      const bb = Math.min(255, Math.max(0, (base & 0xFF) + shift))
      finalColor = `rgb(${Math.round(rr)},${Math.round(gg)},${Math.round(bb)})`
    }
    if (augType === 'erase' && r >= 2 && r <= 4 && c >= 3 && c <= 5) {
      finalColor = '#9ca3af'
    }

    return { x, y, w, h, fill: finalColor, opacity }
  }

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Image Augmentation Demo</h3>
      <div className="flex flex-wrap gap-2 mb-3">
        {['none', 'hflip', 'vflip', 'crop', 'jitter', 'erase'].map(t => (
          <button key={t} onClick={() => setAugType(t)}
            className={`rounded px-3 py-1 text-xs ${augType === t ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300'}`}>
            {t === 'none' ? 'Original' : t === 'hflip' ? 'H-Flip' : t === 'vflip' ? 'V-Flip' : t === 'crop' ? 'Center Crop' : t === 'jitter' ? 'Color Jitter' : 'Random Erase'}
          </button>
        ))}
      </div>
      <svg width={W} height={H} className="mx-auto block border border-gray-200 rounded dark:border-gray-700">
        {basePixels.map((row, r) => row.map((color, c) => {
          const { x, y, w, h, fill, opacity } = applyAug(r, c, color)
          return <rect key={`${r}${c}`} x={x} y={y} width={w} height={h} fill={fill} opacity={opacity} />
        }))}
      </svg>
    </div>
  )
}

export default function ImageAugmentation() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Data augmentation creates new training examples by applying label-preserving
        transformations, effectively expanding the dataset and encoding known invariances
        into the training process.
      </p>

      <DefinitionBlock title="Data Augmentation as Regularization">
        <p>
          Augmentation regularizes by encouraging invariance. If transformation
          <InlineMath math="T" /> preserves labels, training
          on <InlineMath math="T(x)" /> encourages <InlineMath math="f(T(x)) = f(x)" />, equivalent to
          minimizing:
        </p>
        <BlockMath math="\mathcal{L}_{\text{aug}} = \mathbb{E}_{T \sim \mathcal{T}} \left[\mathcal{L}(f(T(x)), y)\right]" />
      </DefinitionBlock>

      <AugmentationDemo />

      <ExampleBlock title="Standard Image Augmentations">
        <ul className="list-disc ml-4 space-y-1">
          <li><strong>Horizontal flip</strong>: invariant for most objects, not text or asymmetric objects</li>
          <li><strong>Random crop</strong>: encourages translation invariance</li>
          <li><strong>Color jitter</strong>: robustness to lighting (brightness, contrast, saturation, hue)</li>
          <li><strong>Random rotation</strong>: <InlineMath math="\pm 15°" /> typically safe</li>
          <li><strong>Random erasing</strong>: occlusion robustness, similar to cutout</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Image Augmentation with torchvision"
        code={`import torch
from torchvision import transforms

# Standard training augmentation pipeline
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.33)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Validation: no augmentation, only resize and normalize
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Test-time augmentation (TTA)
def predict_tta(model, image, n_aug=5):
    model.eval()
    preds = []
    for _ in range(n_aug):
        aug_img = train_transform(image).unsqueeze(0)
        with torch.no_grad():
            preds.append(model(aug_img))
    return torch.stack(preds).mean(0)  # average predictions

print("Train augmentations:", len(train_transform.transforms))`}
      />

      <WarningBlock title="Augmentation Pitfalls">
        <p>
          Never apply training augmentations to validation or test data (except for TTA).
          Be careful with augmentations that can change semantics: vertical flips make
          "6" look like "9", aggressive color jitter can break color-dependent tasks, and
          random erasing can remove the entire object in small-object detection.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Augmentation Strength Schedule">
        <p>
          Some modern recipes increase augmentation strength over training (progressive augmentation).
          Light augmentation early helps the model learn basic features, while strong augmentation
          later prevents overfitting as the model memorizes easy patterns.
        </p>
      </NoteBlock>
    </div>
  )
}

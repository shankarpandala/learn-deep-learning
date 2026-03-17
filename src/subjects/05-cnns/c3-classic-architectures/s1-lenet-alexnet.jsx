import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ArchitectureComparison() {
  const [selected, setSelected] = useState('lenet')
  const archs = {
    lenet: {
      name: 'LeNet-5 (1998)', layers: ['Conv 5x5, 6', 'Pool 2x2', 'Conv 5x5, 16', 'Pool 2x2', 'FC 120', 'FC 84', 'FC 10'],
      params: '60K', input: '32x32x1', innovation: 'First successful CNN for digit recognition'
    },
    alexnet: {
      name: 'AlexNet (2012)', layers: ['Conv 11x11, 96', 'Pool 3x3', 'Conv 5x5, 256', 'Pool 3x3', 'Conv 3x3, 384', 'Conv 3x3, 384', 'Conv 3x3, 256', 'Pool 3x3', 'FC 4096', 'FC 4096', 'FC 1000'],
      params: '60M', input: '227x227x3', innovation: 'ImageNet breakthrough, ReLU, Dropout, GPU training'
    },
  }
  const arch = archs[selected]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Architecture Comparison</h3>
      <div className="flex gap-3 mb-4">
        {Object.entries(archs).map(([key, val]) => (
          <button key={key} onClick={() => setSelected(key)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${selected === key ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-xs text-gray-500 mb-1">Layers</p>
          <div className="space-y-1">
            {arch.layers.map((l, i) => (
              <div key={i} className="text-xs px-2 py-1 rounded bg-violet-50 dark:bg-violet-900/20 text-violet-800 dark:text-violet-200">{l}</div>
            ))}
          </div>
        </div>
        <div className="space-y-3">
          <div><p className="text-xs text-gray-500">Input</p><p className="font-medium text-gray-800 dark:text-gray-200">{arch.input}</p></div>
          <div><p className="text-xs text-gray-500">Parameters</p><p className="font-medium text-gray-800 dark:text-gray-200">{arch.params}</p></div>
          <div><p className="text-xs text-gray-500">Key Innovation</p><p className="text-sm text-gray-700 dark:text-gray-300">{arch.innovation}</p></div>
        </div>
      </div>
    </div>
  )
}

export default function LeNetAlexNet() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        LeNet-5 and AlexNet are foundational CNN architectures. LeNet pioneered the conv-pool-fc
        pattern for digit recognition, while AlexNet's 2012 ImageNet victory launched the deep
        learning revolution in computer vision.
      </p>

      <DefinitionBlock title="LeNet-5 Architecture">
        <p>Yann LeCun's 1998 architecture for handwritten digit recognition on MNIST:</p>
        <BlockMath math="\text{Input}_{32 \times 32} \to \text{C5}_{6} \to \text{Pool} \to \text{C5}_{16} \to \text{Pool} \to \text{FC}_{120} \to \text{FC}_{84} \to \text{FC}_{10}" />
        <p className="mt-2">Used sigmoid/tanh activations and average pooling.</p>
      </DefinitionBlock>

      <DefinitionBlock title="AlexNet Architecture">
        <p>Krizhevsky et al. (2012) won ImageNet with a top-5 error of 15.3% (vs 26.2% for non-DL):</p>
        <BlockMath math="227 \times 227 \to [96, 256, 384, 384, 256] \to \text{FC}_{4096} \to \text{FC}_{4096} \to 1000" />
        <p className="mt-2">Key innovations: ReLU activation, dropout regularization, data augmentation, and dual-GPU training.</p>
      </DefinitionBlock>

      <ArchitectureComparison />

      <ExampleBlock title="ImageNet Scale">
        <p>ImageNet (ILSVRC) contains 1.2 million training images across 1,000 classes.
          AlexNet reduced top-5 error by nearly 11 percentage points, demonstrating that
          deep CNNs trained on GPUs could vastly outperform hand-engineered features.</p>
      </ExampleBlock>

      <PythonCode
        title="AlexNet in PyTorch"
        code={`import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, 5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(256, 384, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

model = AlexNet()
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
x = torch.randn(1, 3, 227, 227)
print(f"Output: {model(x).shape}")  # [1, 1000]`}
      />

      <WarningBlock title="Historical Context">
        <p>
          AlexNet's original implementation used two GPUs (GTX 580, 3GB each) to fit the model.
          Today it runs easily on any GPU. The architecture itself is rarely used in practice now,
          but its ideas (ReLU, dropout, data augmentation) remain foundational.
        </p>
      </WarningBlock>
    </div>
  )
}

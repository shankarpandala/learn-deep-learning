import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ArchitectureComparison() {
  const [selected, setSelected] = useState('lenet')

  const archs = {
    lenet: {
      name: 'LeNet-5 (1998)',
      layers: ['Conv 5x5, 6 filters', 'AvgPool 2x2', 'Conv 5x5, 16 filters', 'AvgPool 2x2', 'FC 120', 'FC 84', 'FC 10'],
      params: '~60K',
      input: '32x32x1',
      activation: 'Tanh',
    },
    alexnet: {
      name: 'AlexNet (2012)',
      layers: ['Conv 11x11, 96, stride 4', 'MaxPool 3x3', 'Conv 5x5, 256, pad 2', 'MaxPool 3x3', 'Conv 3x3, 384', 'Conv 3x3, 384', 'Conv 3x3, 256', 'MaxPool 3x3', 'FC 4096', 'FC 4096', 'FC 1000'],
      params: '~61M',
      input: '227x227x3',
      activation: 'ReLU',
    },
  }
  const arch = archs[selected]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Architecture Comparison</h3>
      <div className="flex gap-3 mb-4">
        {Object.keys(archs).map(k => (
          <button key={k} onClick={() => setSelected(k)} className={`px-3 py-1 rounded text-sm ${selected === k ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {archs[k].name}
          </button>
        ))}
      </div>
      <div className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
        <p><strong>Input:</strong> {arch.input} | <strong>Params:</strong> {arch.params} | <strong>Activation:</strong> {arch.activation}</p>
        <div className="flex flex-wrap gap-1 mt-2">
          {arch.layers.map((l, i) => (
            <span key={i} className="px-2 py-1 rounded bg-violet-50 border border-violet-200 text-xs font-mono dark:bg-violet-900/30 dark:border-violet-700">
              {l}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}

export default function LeNetAlexNet() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        LeNet-5 pioneered CNNs for digit recognition, while AlexNet's ImageNet victory in 2012
        ignited the deep learning revolution. These architectures established the conv-pool-fc
        template used for years.
      </p>

      <DefinitionBlock title="LeNet-5 (LeCun et al., 1998)">
        <p>
          Designed for <InlineMath math="32 \times 32" /> handwritten digit classification. Used
          <InlineMath math="5 \times 5" /> convolutions, average pooling, and tanh activations.
          Demonstrated that learned features outperform hand-engineered ones.
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="AlexNet (Krizhevsky et al., 2012)">
        <p>Key innovations that enabled the ImageNet breakthrough:</p>
        <p className="mt-1">1. <strong>ReLU activation</strong> instead of tanh (6x faster training)</p>
        <p>2. <strong>Dropout</strong> (p=0.5) for regularization</p>
        <p>3. <strong>Data augmentation</strong> (crops, flips, color jittering)</p>
        <p>4. <strong>GPU training</strong> on two GTX 580s with model parallelism</p>
      </DefinitionBlock>

      <ArchitectureComparison />

      <ExampleBlock title="ImageNet Breakthrough">
        <p>
          AlexNet achieved <strong>15.3% top-5 error</strong> on ImageNet (ILSVRC 2012), dramatically
          beating the runner-up at 26.2%. This <InlineMath math="10.9\%" /> gap proved deep learning's
          superiority over traditional computer vision.
        </p>
      </ExampleBlock>

      <PythonCode
        title="AlexNet in PyTorch"
        code={`import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4, padding=2), nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(256, 384, 3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = AlexNet()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Output: {model(torch.randn(1, 3, 227, 227)).shape}")`}
      />

      <NoteBlock type="note" title="Legacy and Impact">
        <p>
          While LeNet and AlexNet are no longer state-of-the-art, their design principles persist:
          hierarchical feature extraction, increasing channel depth, and spatial downsampling. Modern
          architectures refine but do not abandon these ideas.
        </p>
      </NoteBlock>
    </div>
  )
}

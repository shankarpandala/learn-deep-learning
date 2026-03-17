import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function InceptionModuleViz() {
  const [showDetails, setShowDetails] = useState(false)
  const branches = [
    { name: '1x1 conv', color: '#8b5cf6', filters: 64, desc: 'Channel mixing' },
    { name: '1x1 -> 3x3', color: '#a78bfa', filters: 128, desc: 'Local features' },
    { name: '1x1 -> 5x5', color: '#c4b5fd', filters: 32, desc: 'Larger patterns' },
    { name: '3x3 pool -> 1x1', color: '#ddd6fe', filters: 32, desc: 'Pool features' },
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Inception Module</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        <input type="checkbox" checked={showDetails} onChange={e => setShowDetails(e.target.checked)} className="accent-violet-500" />
        Show filter counts
      </label>
      <div className="flex justify-center gap-3">
        {branches.map((b, i) => (
          <div key={i} className="text-center">
            <div className="rounded-lg p-3 text-xs font-medium" style={{ backgroundColor: b.color, color: '#1e1b4b', minWidth: 80 }}>
              {b.name}
              {showDetails && <div className="mt-1 font-bold">{b.filters} filters</div>}
            </div>
            <p className="text-xs text-gray-500 mt-1">{b.desc}</p>
          </div>
        ))}
      </div>
      <div className="text-center mt-3">
        <div className="inline-block rounded-lg bg-violet-100 dark:bg-violet-900/30 px-4 py-2 text-sm font-medium text-violet-800 dark:text-violet-200">
          Filter Concatenation &rarr; {showDetails ? `${branches.reduce((s, b) => s + b.filters, 0)} channels` : 'Combined output'}
        </div>
      </div>
    </div>
  )
}

export default function VGGInception() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        VGG and GoogLeNet (Inception) represented two contrasting philosophies in CNN design.
        VGG proved that depth with simple 3x3 convolutions works well, while Inception showed
        that multi-scale feature extraction within a single layer is highly efficient.
      </p>

      <DefinitionBlock title="VGG Architecture (2014)">
        <p>
          VGG uses exclusively <InlineMath math="3 \times 3" /> convolutions stacked deeply (16-19 layers).
          Two stacked <InlineMath math="3 \times 3" /> convs have the same receptive field as one{' '}
          <InlineMath math="5 \times 5" /> but fewer parameters:
        </p>
        <BlockMath math="2 \times (3^2 C^2) = 18C^2 < 25C^2 = 5^2 C^2" />
      </DefinitionBlock>

      <TheoremBlock title="VGG Design Principle" id="vgg-principle">
        <p>VGG follows a simple rule: after each max-pooling, double the number of channels:</p>
        <BlockMath math="64 \xrightarrow{\text{pool}} 128 \xrightarrow{\text{pool}} 256 \xrightarrow{\text{pool}} 512 \xrightarrow{\text{pool}} 512" />
        <p className="mt-2">
          This maintains roughly constant computational cost per layer since spatial dimensions
          halve while channels double: <InlineMath math="(H/2)^2 \cdot 2C \approx H^2 \cdot C / 2" />.
        </p>
      </TheoremBlock>

      <DefinitionBlock title="Inception Module (GoogLeNet, 2014)">
        <p>
          Processes input through parallel branches of different kernel sizes, then concatenates
          outputs along the channel dimension. Bottleneck <InlineMath math="1 \times 1" /> convolutions
          reduce computation before expensive <InlineMath math="3 \times 3" /> and{' '}
          <InlineMath math="5 \times 5" /> convolutions.
        </p>
      </DefinitionBlock>

      <InceptionModuleViz />

      <ExampleBlock title="Parameter Efficiency">
        <p>GoogLeNet achieved better accuracy than VGG-16 with 12x fewer parameters:</p>
        <BlockMath math="\text{VGG-16: 138M params} \quad \text{vs} \quad \text{GoogLeNet: 6.8M params}" />
        <p>The 1x1 bottleneck convolutions were key to this efficiency.</p>
      </ExampleBlock>

      <PythonCode
        title="Inception Module in PyTorch"
        code={`import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, in_ch, ch1, ch3_red, ch3, ch5_red, ch5, pool_proj):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, ch1, 1), nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, ch3_red, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch3_red, ch3, 3, padding=1), nn.ReLU(inplace=True))
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_ch, ch5_red, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch5_red, ch5, 5, padding=2), nn.ReLU(inplace=True))
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, pool_proj, 1), nn.ReLU(inplace=True))

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch3(x),
                          self.branch5(x), self.branch_pool(x)], dim=1)

module = InceptionModule(192, 64, 96, 128, 16, 32, 32)
x = torch.randn(1, 192, 28, 28)
out = module(x)
print(f"Output: {out.shape}")  # [1, 256, 28, 28] (64+128+32+32)
print(f"Params: {sum(p.numel() for p in module.parameters()):,}")`}
      />

      <NoteBlock type="note" title="Evolution of Inception">
        <p>
          Inception v2 replaced <InlineMath math="5 \times 5" /> with two stacked{' '}
          <InlineMath math="3 \times 3" /> convolutions. Inception v3 factorized{' '}
          <InlineMath math="n \times n" /> into <InlineMath math="1 \times n" /> and{' '}
          <InlineMath math="n \times 1" />. Inception v4 combined Inception modules with residual connections.
        </p>
      </NoteBlock>
    </div>
  )
}

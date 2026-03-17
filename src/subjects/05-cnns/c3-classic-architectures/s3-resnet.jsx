import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ResidualBlockViz() {
  const [showBottleneck, setShowBottleneck] = useState(false)
  const basicLayers = ['3x3 Conv, BN, ReLU', '3x3 Conv, BN']
  const bottleneckLayers = ['1x1 Conv (reduce), BN, ReLU', '3x3 Conv, BN, ReLU', '1x1 Conv (expand), BN']
  const layers = showBottleneck ? bottleneckLayers : basicLayers
  const blockW = 200, layerH = 36, gap = 8, padY = 40
  const totalH = layers.length * (layerH + gap) + 2 * padY

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Residual Block</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        <input type="checkbox" checked={showBottleneck} onChange={e => setShowBottleneck(e.target.checked)} className="accent-violet-500" />
        Show bottleneck variant
      </label>
      <svg width={340} height={totalH} className="mx-auto block">
        <text x={blockW / 2 + 40} y={20} textAnchor="middle" fontSize={12} fill="#6b7280">x (input)</text>
        <line x1={blockW / 2 + 40} y1={25} x2={blockW / 2 + 40} y2={padY} stroke="#9ca3af" strokeWidth={1.5} />
        {layers.map((l, i) => {
          const y = padY + i * (layerH + gap)
          return (
            <g key={i}>
              <rect x={40} y={y} width={blockW} height={layerH} rx={6} fill="#ede9fe" stroke="#8b5cf6" strokeWidth={1.5} />
              <text x={40 + blockW / 2} y={y + layerH / 2 + 5} textAnchor="middle" fontSize={11} fill="#4c1d95">{l}</text>
            </g>
          )
        })}
        {/* Skip connection */}
        <line x1={blockW + 50} y1={padY} x2={blockW + 80} y2={padY} stroke="#7c3aed" strokeWidth={2} />
        <line x1={blockW + 80} y1={padY} x2={blockW + 80} y2={padY + layers.length * (layerH + gap) - gap + layerH / 2} stroke="#7c3aed" strokeWidth={2} />
        <line x1={blockW + 80} y1={padY + layers.length * (layerH + gap) - gap + layerH / 2} x2={blockW + 50} y2={padY + layers.length * (layerH + gap) - gap + layerH / 2} stroke="#7c3aed" strokeWidth={2} markerEnd="url(#arrow)" />
        <text x={blockW + 90} y={padY + (layers.length * (layerH + gap)) / 2} fontSize={11} fill="#7c3aed" fontWeight="bold">identity</text>
        <defs><marker id="arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#7c3aed" /></marker></defs>
        {/* Add + ReLU */}
        <text x={blockW / 2 + 40} y={totalH - 10} textAnchor="middle" fontSize={12} fill="#6b7280">F(x) + x &rarr; ReLU</text>
      </svg>
    </div>
  )
}

export default function ResNet() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        ResNet (He et al., 2015) introduced skip connections that allow gradients to flow directly
        through the network. This breakthrough enabled training of networks with 100+ layers,
        winning ILSVRC 2015 with a 3.57% top-5 error rate, surpassing human performance.
      </p>

      <DefinitionBlock title="Residual Learning">
        <p>Instead of learning a mapping <InlineMath math="H(x)" /> directly, the network learns the residual:</p>
        <BlockMath math="H(x) = F(x) + x" />
        <p className="mt-2">
          Where <InlineMath math="F(x)" /> is the residual function learned by stacked layers.
          If the identity mapping is optimal, it is easier to push <InlineMath math="F(x) \to 0" /> than
          to learn <InlineMath math="H(x) = x" /> through nonlinear layers.
        </p>
      </DefinitionBlock>

      <ResidualBlockViz />

      <TheoremBlock title="Gradient Flow Through Skip Connections" id="resnet-gradient">
        <p>With skip connections, the gradient decomposes as:</p>
        <BlockMath math="\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \prod_{i=l}^{L-1}\left(1 + \frac{\partial F_i}{\partial x_i}\right)" />
        <p className="mt-2">
          The additive <InlineMath math="1" /> ensures gradients never vanish completely, regardless
          of depth. This is the key insight enabling very deep networks.
        </p>
      </TheoremBlock>

      <ExampleBlock title="ResNet Variants">
        <p>
          ResNet-18: 11.7M params | ResNet-50: 25.6M params | ResNet-152: 60.2M params.
          ResNet-50 uses bottleneck blocks (<InlineMath math="1 \times 1 \to 3 \times 3 \to 1 \times 1" />)
          which are more parameter-efficient than the basic blocks in ResNet-18/34.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Residual Block in PyTorch"
        code={`import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # skip connection
        return self.relu(out)

block = BasicBlock(64, 128, stride=2)
x = torch.randn(1, 64, 56, 56)
print(f"Output: {block(x).shape}")  # [1, 128, 28, 28]
print(f"Params: {sum(p.numel() for p in block.parameters()):,}")`}
      />

      <NoteBlock type="note" title="Pre-activation ResNet">
        <p>
          He et al. (2016) later proposed placing BN and ReLU before convolutions
          (BN-ReLU-Conv instead of Conv-BN-ReLU). This "pre-activation" variant improves
          gradient flow and is easier to optimize, especially in very deep networks (1000+ layers).
        </p>
      </NoteBlock>
    </div>
  )
}

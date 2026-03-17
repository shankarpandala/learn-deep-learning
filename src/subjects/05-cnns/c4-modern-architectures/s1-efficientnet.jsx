import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ScalingExplorer() {
  const [phi, setPhi] = useState(1)
  const alpha = 1.2, beta = 1.1, gamma = 1.15
  const depth = Math.pow(alpha, phi).toFixed(2)
  const width = Math.pow(beta, phi).toFixed(2)
  const resolution = Math.pow(gamma, phi).toFixed(2)
  const flops = Math.pow(alpha * beta * beta * gamma * gamma, phi).toFixed(2)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Compound Scaling Explorer</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-4">
        Compound coefficient <InlineMath math={`\\phi = ${phi}`} />
        <input type="range" min={0} max={7} step={1} value={phi} onChange={e => setPhi(parseInt(e.target.value))} className="w-48 accent-violet-500" />
      </label>
      <div className="grid grid-cols-4 gap-3 text-center">
        {[
          ['Depth', depth, `\\alpha^{\\phi} = ${depth}`],
          ['Width', width, `\\beta^{\\phi} = ${width}`],
          ['Resolution', resolution, `\\gamma^{\\phi} = ${resolution}`],
          ['FLOPs', flops, `\\approx 2^{\\phi} = ${flops}`],
        ].map(([label, val, formula]) => (
          <div key={label} className="p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20">
            <p className="text-xs text-gray-500">{label} scale</p>
            <p className="text-lg font-bold text-violet-700 dark:text-violet-300">{val}x</p>
          </div>
        ))}
      </div>
      <p className="text-xs text-center text-gray-500 mt-2">
        EfficientNet B{phi}: depth {depth}x, width {width}x, resolution {resolution}x
      </p>
    </div>
  )
}

export default function EfficientNet() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        EfficientNet (Tan & Le, 2019) introduced compound scaling, a principled method to scale
        CNNs by jointly increasing depth, width, and resolution. The base model EfficientNet-B0
        was found via neural architecture search and then scaled to B1-B7.
      </p>

      <DefinitionBlock title="Compound Scaling">
        <p>Given a compound coefficient <InlineMath math="\phi" />, scale the network dimensions:</p>
        <BlockMath math="\text{depth: } d = \alpha^\phi, \quad \text{width: } w = \beta^\phi, \quad \text{resolution: } r = \gamma^\phi" />
        <p className="mt-2">Subject to the constraint <InlineMath math="\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2" />,
          so FLOPs roughly double with each unit increase in <InlineMath math="\phi" />.</p>
      </DefinitionBlock>

      <TheoremBlock title="Scaling Constraint" id="compound-constraint">
        <BlockMath math="\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2" />
        <p className="mt-2">
          Since FLOPS scale as <InlineMath math="d \cdot w^2 \cdot r^2" />, this constraint ensures
          total FLOPs increase by <InlineMath math="2^\phi" />. EfficientNet uses{' '}
          <InlineMath math="\alpha = 1.2, \beta = 1.1, \gamma = 1.15" />.
        </p>
      </TheoremBlock>

      <ScalingExplorer />

      <ExampleBlock title="EfficientNet Performance">
        <p>
          EfficientNet-B7 achieved 84.3% top-1 accuracy on ImageNet with 66M parameters and 37B FLOPs.
          For comparison, GPipe (the previous SOTA) needed 556M parameters and 4x more FLOPs
          for similar accuracy.
        </p>
      </ExampleBlock>

      <PythonCode
        title="EfficientNet with torchvision"
        code={`import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Load pretrained EfficientNet-B0
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model.eval()

# EfficientNet-B0: 5.3M params, 224x224 input
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = model(x)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Output: {out.shape}")  # [1, 1000]

# MBConv block is the key building block
# It uses depthwise separable convs + squeeze-and-excitation
print(f"\\nArchitecture overview:")
for name, module in model.features.named_children():
    print(f"  Stage {name}: {module.__class__.__name__}")`}
      />

      <NoteBlock type="note" title="Squeeze-and-Excitation (SE) Blocks">
        <p>
          Each MBConv block in EfficientNet includes a SE module that recalibrates channel-wise
          features. It squeezes spatial information via global average pooling, then excites
          channels through two FC layers with a sigmoid gate:{' '}
          <InlineMath math="s = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot z))" />. This adaptive
          channel weighting improves representational power with minimal overhead.
        </p>
      </NoteBlock>

      <NoteBlock type="note" title="EfficientNet v2">
        <p>
          EfficientNet v2 (2021) improved training speed by using Fused-MBConv blocks in early
          stages (replacing depthwise separable with standard convolutions), progressive learning
          (increasing image size during training), and adaptive regularization. It achieves
          similar accuracy to EfficientNet v1 while training up to 11x faster.
        </p>
      </NoteBlock>
    </div>
  )
}

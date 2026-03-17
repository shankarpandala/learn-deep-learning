import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ModernizationSteps() {
  const [step, setStep] = useState(0)
  const steps = [
    { name: 'ResNet-50 baseline', acc: 78.8, change: 'Starting point' },
    { name: 'Training recipe update', acc: 80.6, change: 'AdamW, augmentation, longer training (300 epochs)' },
    { name: 'Macro design changes', acc: 80.9, change: 'Stage ratio 3:3:9:3, patchify stem (4x4 stride-4 conv)' },
    { name: 'Depthwise conv 7x7', acc: 81.3, change: 'Replace 3x3 with 7x7 depthwise (like self-attention spatial mixing)' },
    { name: 'Inverted bottleneck', acc: 81.5, change: 'Expand channels 4x with 1x1 conv (like Transformer FFN)' },
    { name: 'Fewer activations & norms', acc: 82.0, change: 'Single GELU, LayerNorm instead of BatchNorm' },
    { name: 'Final: ConvNeXt-T', acc: 82.1, change: 'Micro design tweaks (separate downsampling layers)' },
  ]
  const s = steps[step]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Modernization Roadmap</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Step {step + 1}/{steps.length}
        <input type="range" min={0} max={steps.length - 1} value={step} onChange={e => setStep(parseInt(e.target.value))} className="w-48 accent-violet-500" />
      </label>
      <div className="p-4 rounded-lg bg-violet-50 dark:bg-violet-900/20">
        <div className="flex justify-between items-center mb-2">
          <p className="font-bold text-violet-800 dark:text-violet-200">{s.name}</p>
          <p className="text-lg font-bold text-violet-700 dark:text-violet-300">{s.acc}%</p>
        </div>
        <p className="text-sm text-gray-600 dark:text-gray-400">{s.change}</p>
        <div className="mt-3 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
          <div className="bg-violet-500 h-2 rounded-full transition-all" style={{ width: `${(s.acc - 78) / 5 * 100}%` }} />
        </div>
      </div>
    </div>
  )
}

export default function ConvNeXt() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        ConvNeXt (Liu et al., 2022) demonstrated that a pure convolutional architecture, when
        modernized with training techniques and design choices from Vision Transformers, can match
        or exceed Swin Transformer performance. It provides a strong CNN baseline for the ViT era.
      </p>

      <DefinitionBlock title="ConvNeXt Block">
        <p>Each ConvNeXt block consists of:</p>
        <BlockMath math="\text{DWConv}_{7 \times 7} \to \text{LayerNorm} \to \text{Linear}_{4C} \to \text{GELU} \to \text{Linear}_{C}" />
        <p className="mt-2">
          This mirrors the structure of a Transformer block: spatial mixing (depthwise conv
          analogous to self-attention) followed by channel mixing (inverted bottleneck MLP).
        </p>
      </DefinitionBlock>

      <ModernizationSteps />

      <ExampleBlock title="ConvNeXt Performance">
        <p>ConvNeXt matches Swin Transformer at all model sizes:</p>
        <BlockMath math="\text{ConvNeXt-T: 82.1\%} \approx \text{Swin-T: 81.3\%}" />
        <BlockMath math="\text{ConvNeXt-B: 83.8\%} \approx \text{Swin-B: 83.5\%}" />
        <p>With comparable parameters, FLOPs, and throughput at each scale.</p>
      </ExampleBlock>

      <PythonCode
        title="ConvNeXt Block in PyTorch"
        code={`import torch
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, expansion=4, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))  # Layer Scale

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC for LayerNorm
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return shortcut + x

block = ConvNeXtBlock(dim=96)
x = torch.randn(1, 96, 56, 56)
print(f"Output: {block(x).shape}")  # [1, 96, 56, 56]
print(f"Params: {sum(p.numel() for p in block.parameters()):,}")`}
      />

      <WarningBlock title="Key Takeaway">
        <p>
          ConvNeXt shows that the Vision Transformer's superiority was largely due to improved
          training recipes (stronger augmentation, AdamW, longer schedules) rather than the
          self-attention mechanism itself. Architecture and training recipe improvements are
          complementary and often conflated.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="ConvNeXt v2">
        <p>
          ConvNeXt v2 (2023) introduces a Global Response Normalization (GRN) layer and is designed
          to work well with masked autoencoder (MAE) self-supervised pretraining, further closing
          the gap with Vision Transformers in the self-supervised regime.
        </p>
      </NoteBlock>
    </div>
  )
}

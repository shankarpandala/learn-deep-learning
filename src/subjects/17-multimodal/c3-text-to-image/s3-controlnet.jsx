import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ControlTypeExplorer() {
  const [controlType, setControlType] = useState('canny')
  const controls = {
    canny: { name: 'Canny Edges', desc: 'Edge maps extracted from images to preserve structure and boundaries.', useCase: 'Architectural drawings, product design' },
    depth: { name: 'Depth Maps', desc: 'Estimated depth to control 3D layout and perspective of generated scene.', useCase: 'Scene composition, room layouts' },
    pose: { name: 'OpenPose', desc: 'Human body keypoints to control pose and position of people.', useCase: 'Character art, fashion design' },
    segmentation: { name: 'Semantic Segmentation', desc: 'Pixel-level class labels define what goes where in the image.', useCase: 'Landscape design, scene manipulation' },
  }
  const c = controls[controlType]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">ControlNet Conditioning Types</h3>
      <div className="flex gap-2 mb-4 flex-wrap">
        {Object.entries(controls).map(([key, val]) => (
          <button key={key} onClick={() => setControlType(key)}
            className={`px-3 py-1 rounded-lg text-sm transition ${controlType === key ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20 text-sm space-y-1">
        <p><strong>{c.name}</strong></p>
        <p className="text-gray-600 dark:text-gray-400">{c.desc}</p>
        <p className="text-xs text-gray-500">Use case: {c.useCase}</p>
      </div>
    </div>
  )
}

export default function ControlNetGuidedGeneration() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        ControlNet adds spatial conditioning to pretrained diffusion models by creating a trainable
        copy of the encoder blocks. This enables precise control over generated images using
        edge maps, depth maps, poses, and other structural guides.
      </p>

      <DefinitionBlock title="ControlNet Architecture">
        <p>ControlNet creates a trainable copy of the locked U-Net encoder and connects it via zero convolutions:</p>
        <BlockMath math="y_c = \mathcal{F}(x; \Theta) + \mathcal{Z}(\mathcal{F}(x + \mathcal{Z}(c; \Theta_{z1}); \Theta_c); \Theta_{z2})" />
        <p className="mt-2">where <InlineMath math="\mathcal{F}" /> is the frozen U-Net block, <InlineMath math="\Theta_c" /> is its trainable copy, <InlineMath math="c" /> is the control signal, and <InlineMath math="\mathcal{Z}" /> is a zero-initialized convolution (output starts at zero, preserving the original model).</p>
      </DefinitionBlock>

      <ControlTypeExplorer />

      <ExampleBlock title="Zero Convolution: Why It Works">
        <p>The zero convolution <InlineMath math="\mathcal{Z}(x) = W \cdot x + b" /> is initialized with <InlineMath math="W=0, b=0" />:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>At initialization: <InlineMath math="\mathcal{Z}(x) = 0" />, so the pretrained model is unchanged</li>
          <li>Gradients are non-zero: <InlineMath math="\frac{\partial \mathcal{Z}}{\partial W} = x \neq 0" /></li>
          <li>The network gradually learns to inject control signals without disrupting pretrained features</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="ControlNet Zero Convolution Block"
        code={`import torch
import torch.nn as nn

class ZeroConv(nn.Module):
    """Zero-initialized convolution for ControlNet."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)

class ControlNetBlock(nn.Module):
    """Simplified ControlNet block with zero convolutions."""
    def __init__(self, channels=320):
        super().__init__()
        self.zero_in = ZeroConv(channels, channels)
        self.trainable_copy = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.zero_out = ZeroConv(channels, channels)

    def forward(self, frozen_features, control_signal):
        # Inject control via zero conv -> process -> zero conv
        h = frozen_features + self.zero_in(control_signal)
        h = self.trainable_copy(h)
        return self.zero_out(h)  # Added to frozen U-Net output

block = ControlNetBlock(channels=320)
feat = torch.randn(1, 320, 64, 64)
ctrl = torch.randn(1, 320, 64, 64)
out = block(feat, ctrl)
print(f"Output shape: {out.shape}")
print(f"Initial output magnitude: {out.abs().mean():.6f}")  # ~0 at init`}
      />

      <NoteBlock type="note" title="IP-Adapter and Other Control Methods">
        <p>
          Beyond ControlNet, other methods add control to diffusion: IP-Adapter uses image embeddings
          as conditioning (image-prompted generation), T2I-Adapter adds lightweight control modules,
          and LoRA fine-tunes specific weight matrices for style transfer. These can be composed
          together for multi-condition generation — for example, combining a pose ControlNet
          with a depth ControlNet and a style LoRA to generate a specifically posed character
          in a particular style and spatial layout. This composability makes ControlNet
          a foundational tool in production image generation pipelines.
        </p>
      </NoteBlock>
    </div>
  )
}

import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function UNetDiagram() {
  const [showSkips, setShowSkips] = useState(true)
  const encoder = [
    { w: 80, h: 50, ch: 64, label: '64' },
    { w: 60, h: 40, ch: 128, label: '128' },
    { w: 44, h: 32, ch: 256, label: '256' },
    { w: 30, h: 24, ch: 512, label: '512' },
  ]
  const bottleneck = { w: 22, h: 18, ch: 1024, label: '1024' }
  const baseX = 20, baseY = 10, gap = 16

  let cx = baseX
  const encPositions = encoder.map((e) => {
    const pos = { x: cx, y: baseY + (50 - e.h) / 2, w: e.w, h: e.h, label: e.label }
    cx += e.w + gap
    return pos
  })
  const bnPos = { x: cx, y: baseY + (50 - bottleneck.h) / 2, w: bottleneck.w, h: bottleneck.h, label: bottleneck.label }
  cx += bottleneck.w + gap
  const decPositions = [...encoder].reverse().map((e) => {
    const pos = { x: cx, y: baseY + (50 - e.h) / 2, w: e.w, h: e.h, label: e.label }
    cx += e.w + gap
    return pos
  })

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">U-Net Architecture</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        <input type="checkbox" checked={showSkips} onChange={e => setShowSkips(e.target.checked)} className="accent-violet-500" />
        Show skip connections
      </label>
      <svg width={cx + 10} height={70} className="mx-auto block">
        {encPositions.map((p, i) => (
          <g key={`e-${i}`}>
            <rect x={p.x} y={p.y} width={p.w} height={p.h} rx={3} fill="#ddd6fe" stroke="#8b5cf6" strokeWidth={1.5} />
            <text x={p.x + p.w / 2} y={p.y + p.h / 2 + 4} textAnchor="middle" fontSize={9} fill="#4c1d95">{p.label}</text>
          </g>
        ))}
        <rect x={bnPos.x} y={bnPos.y} width={bnPos.w} height={bnPos.h} rx={3} fill="#8b5cf6" stroke="#6d28d9" strokeWidth={1.5} />
        <text x={bnPos.x + bnPos.w / 2} y={bnPos.y + bnPos.h / 2 + 3} textAnchor="middle" fontSize={7} fill="white">{bnPos.label}</text>
        {decPositions.map((p, i) => (
          <g key={`d-${i}`}>
            <rect x={p.x} y={p.y} width={p.w} height={p.h} rx={3} fill="#ede9fe" stroke="#a78bfa" strokeWidth={1.5} />
            <text x={p.x + p.w / 2} y={p.y + p.h / 2 + 4} textAnchor="middle" fontSize={9} fill="#5b21b6">{p.label}</text>
          </g>
        ))}
        {showSkips && encPositions.map((ep, i) => {
          const dp = decPositions[encoder.length - 1 - i]
          return <line key={`s-${i}`} x1={ep.x + ep.w / 2} y1={ep.y} x2={dp.x + dp.w / 2} y2={dp.y} stroke="#7c3aed" strokeWidth={1.5} strokeDasharray="4,3" opacity={0.6} />
        })}
      </svg>
      <div className="flex justify-between text-xs text-gray-500 mt-1 px-4">
        <span>Encoder (downsample)</span>
        <span>Bottleneck</span>
        <span>Decoder (upsample)</span>
      </div>
    </div>
  )
}

export default function FCNUNet() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Fully Convolutional Networks (FCN) and U-Net are foundational architectures for semantic
        segmentation. FCN replaced fully connected layers with convolutions for dense prediction,
        while U-Net added skip connections between encoder and decoder for precise localization.
      </p>

      <DefinitionBlock title="Fully Convolutional Network (FCN)">
        <p>
          FCN (Long et al., 2015) converts a classification CNN into a dense predictor by replacing
          FC layers with <InlineMath math="1 \times 1" /> convolutions and upsampling with
          transposed convolutions to produce pixel-wise predictions:
        </p>
        <BlockMath math="\hat{y}_{ij} = \arg\max_c \; f_c(x)_{ij}" />
      </DefinitionBlock>

      <DefinitionBlock title="U-Net Architecture">
        <p>
          U-Net (Ronneberger et al., 2015) uses an encoder-decoder structure with skip connections
          that concatenate encoder features with decoder features at each resolution:
        </p>
        <BlockMath math="\text{dec}_l = \text{Conv}\left(\text{cat}(\text{up}(\text{dec}_{l+1}), \; \text{enc}_l)\right)" />
        <p className="mt-2">This preserves fine spatial details lost during downsampling.</p>
      </DefinitionBlock>

      <UNetDiagram />

      <ExampleBlock title="Why Skip Connections Help">
        <p>
          Low-level encoder features contain precise spatial information (edges, textures) while
          high-level features contain semantic information (object identity). Skip connections
          combine both, enabling the network to produce segmentation masks that are both
          semantically correct and spatially precise.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Segmentation with segmentation_models_pytorch"
        code={`import segmentation_models_pytorch as smp
import torch

# U-Net with pretrained ResNet34 encoder (transfer learning)
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",   # pretrained backbone
    in_channels=3,
    classes=21,                   # e.g., Pascal VOC classes
    decoder_channels=(256, 128, 64, 32, 16),
)

x = torch.randn(1, 3, 256, 256)
output = model(x)
print(f"Input: {x.shape} -> Segmentation: {output.shape}")
# [1, 21, 256, 256] - per-pixel class logits

# Other architectures available out of the box:
unet_pp = smp.UnetPlusPlus(encoder_name="efficientnet-b4", classes=21)
deeplabv3 = smp.DeepLabV3Plus(encoder_name="resnet50", classes=21)
fpn = smp.FPN(encoder_name="resnet34", classes=21)

# Built-in losses and metrics
loss_fn = smp.losses.DiceLoss(mode="multiclass")
pred = model(x)
target = torch.randint(0, 21, (1, 256, 256))
loss = loss_fn(pred, target)
print(f"Dice loss: {loss.item():.4f}")

# 400+ encoder variants: ResNet, EfficientNet, ViT, MiT, etc.
print(f"Available encoders: {len(smp.encoders.get_encoder_names())}")
print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")`}
      />

      <NoteBlock type="note" title="U-Net in Medical Imaging">
        <p>
          U-Net was originally designed for biomedical image segmentation with very few training
          images. Its encoder-decoder design with skip connections became the standard architecture
          for medical imaging tasks and has been adapted into U-Net++, Attention U-Net, and
          nnU-Net (self-configuring U-Net).
        </p>
      </NoteBlock>
    </div>
  )
}

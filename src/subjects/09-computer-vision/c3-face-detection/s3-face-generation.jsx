import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function StyleMixingDemo() {
  const [coarseStyle, setCoarseStyle] = useState(50)
  const [fineStyle, setFineStyle] = useState(50)
  const W = 240, H = 160
  const hue1 = coarseStyle * 3.6
  const hue2 = fineStyle * 3.6

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Style Mixing Visualization</h3>
      <div className="flex flex-col gap-2 mb-4">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Coarse style (pose, shape):
          <input type="range" min={0} max={100} value={coarseStyle} onChange={e => setCoarseStyle(parseInt(e.target.value))} className="w-36 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Fine style (color, texture):
          <input type="range" min={0} max={100} value={fineStyle} onChange={e => setFineStyle(parseInt(e.target.value))} className="w-36 accent-violet-500" />
        </label>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <ellipse cx={W / 2} cy={H / 2 - 10} rx={35 + coarseStyle * 0.15} ry={45 + coarseStyle * 0.1}
          fill={`hsl(${hue2}, 60%, 75%)`} stroke="#8b5cf6" strokeWidth={1.5} />
        <circle cx={W / 2 - 12} cy={H / 2 - 20} r={4} fill={`hsl(${hue2}, 70%, 40%)`} />
        <circle cx={W / 2 + 12} cy={H / 2 - 20} r={4} fill={`hsl(${hue2}, 70%, 40%)`} />
        <path d={`M${W / 2 - 8},${H / 2 + 5} Q${W / 2},${H / 2 + 15} ${W / 2 + 8},${H / 2 + 5}`}
          fill="none" stroke={`hsl(${hue2}, 60%, 50%)`} strokeWidth={1.5} />
        <text x={W / 2} y={H - 5} textAnchor="middle" fontSize={10} fill="#6b7280">
          Coarse: {coarseStyle}% | Fine: {fineStyle}%
        </text>
      </svg>
    </div>
  )
}

export default function FaceGeneration() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Face generation using GANs (particularly StyleGAN) can create photorealistic synthetic faces.
        This technology enables face editing, but also raises concerns about deepfakes.
      </p>

      <DefinitionBlock title="StyleGAN Architecture">
        <p>StyleGAN maps a latent code through a mapping network to a style space:</p>
        <BlockMath math="z \in \mathcal{Z} \xrightarrow{f} w \in \mathcal{W} \xrightarrow{\text{AdaIN}} \text{synthesis}" />
        <p className="mt-2">
          Adaptive Instance Normalization (AdaIN) injects style at each layer:
        </p>
        <BlockMath math="\text{AdaIN}(x_i, y) = y_{s,i}\frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}" />
      </DefinitionBlock>

      <StyleMixingDemo />

      <TheoremBlock title="W+ Space for Editing" id="w-plus">
        <p>
          The extended <InlineMath math="\mathcal{W}^+" /> space uses different <InlineMath math="w" /> vectors
          per layer, enabling fine-grained control:
        </p>
        <BlockMath math="\mathcal{W}^+ = \{(w_1, w_2, \ldots, w_L) \mid w_i \in \mathcal{W}\}" />
        <p className="mt-1">
          Editing directions <InlineMath math="n" /> in <InlineMath math="\mathcal{W}" /> correspond to
          semantic attributes: <InlineMath math="w' = w + \alpha \cdot n" /> (e.g., age, smile, pose).
        </p>
      </TheoremBlock>

      <ExampleBlock title="StyleGAN Layers and Attributes">
        <ul className="list-disc ml-5 space-y-1">
          <li><strong>Coarse layers (4-8)</strong>: pose, face shape, hairstyle</li>
          <li><strong>Middle layers (16-32)</strong>: facial features, eye shape, nose</li>
          <li><strong>Fine layers (64-1024)</strong>: color scheme, skin texture, lighting</li>
        </ul>
        <p className="mt-2">
          Style mixing applies different <InlineMath math="w" /> codes at different resolution layers.
        </p>
      </ExampleBlock>

      <PythonCode
        title="StyleGAN2 Inference and Editing"
        code={`import torch
import torch.nn as nn

class MappingNetwork(nn.Module):
    """StyleGAN2 mapping network: Z -> W."""
    def __init__(self, z_dim=512, w_dim=512, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = z_dim if i == 0 else w_dim
            layers.extend([nn.Linear(in_d, w_dim), nn.LeakyReLU(0.2)])
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)  # w: (B, 512)

# Face editing in W space
def edit_face(w, direction, alpha=3.0):
    """Edit a face attribute by moving in latent space."""
    return w + alpha * direction

# Deepfake detection features
def extract_frequency_features(image):
    """DCT-based frequency analysis for deepfake detection."""
    # Real faces have consistent high-frequency patterns
    # GAN-generated faces often lack certain frequencies
    dct = torch.fft.fft2(image)
    magnitude = torch.abs(dct)
    # High-frequency energy ratio as detection feature
    h, w = magnitude.shape[-2:]
    center = magnitude[..., h//4:3*h//4, w//4:3*w//4].sum()
    total = magnitude.sum()
    return 1.0 - center / total  # Higher = more HF content`}
      />

      <WarningBlock title="Deepfake Detection">
        <p>
          Detecting AI-generated faces relies on artifacts invisible to humans:
          inconsistent specular reflections, frequency spectrum anomalies, and
          temporal flickering in videos. Binary classifiers trained on real vs generated
          faces achieve over 95% detection accuracy but struggle with unseen generators.
          Multi-spectral analysis using <InlineMath math="\mathcal{F}\{I\}" /> (frequency domain)
          provides more robust detection signals.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Ethical Considerations">
        <p>
          Face generation technology requires responsible use. Key concerns include
          non-consensual deepfakes, identity fraud, and misinformation. Research
          in provenance tracking, watermarking, and robust detection methods is critical
          for mitigating misuse while preserving beneficial applications.
        </p>
      </NoteBlock>
    </div>
  )
}

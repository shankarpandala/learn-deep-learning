import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function StyleMixingViz() {
  const [crossoverLayer, setCrossoverLayer] = useState(4)
  const layers = ['4x4', '8x8', '16x16', '32x32', '64x64', '128x128', '256x256', '512x512', '1024x1024']

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Style Mixing Crossover</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Crossover at layer: {crossoverLayer}
        <input type="range" min={1} max={8} step={1} value={crossoverLayer} onChange={e => setCrossoverLayer(parseInt(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <div className="flex gap-1 items-end justify-center">
        {layers.map((res, i) => {
          const fromA = i < crossoverLayer
          return (
            <div key={i} className="flex flex-col items-center">
              <div
                className={`rounded ${fromA ? 'bg-violet-500' : 'bg-orange-400'}`}
                style={{ width: 24, height: 8 + i * 6 }}
              />
              <span className="text-[8px] text-gray-400 mt-1">{res}</span>
            </div>
          )
        })}
      </div>
      <div className="flex justify-center gap-4 text-xs mt-2">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-2 bg-violet-500 rounded" /> Style A (coarse)</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-2 bg-orange-400 rounded" /> Style B (fine)</span>
      </div>
      <p className="text-xs text-gray-500 text-center mt-1">
        {crossoverLayer <= 3 ? 'Low crossover: A controls pose/shape, B controls colors/details' :
         crossoverLayer <= 6 ? 'Mid crossover: A controls structure, B controls fine features' :
         'High crossover: A controls almost everything, B only affects finest details'}
      </p>
    </div>
  )
}

export default function StyleGAN() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        StyleGAN revolutionized high-resolution image synthesis by introducing a style-based
        generator architecture that provides unprecedented control over the generation process
        at different levels of detail.
      </p>

      <DefinitionBlock title="StyleGAN Architecture">
        <p>Key innovations of the style-based generator:</p>
        <ul className="list-disc ml-5 mt-2 space-y-1">
          <li><strong>Mapping network</strong>: 8-layer MLP transforms <InlineMath math="\mathbf{z} \in \mathcal{Z}" /> to <InlineMath math="\mathbf{w} \in \mathcal{W}" /></li>
          <li><strong>AdaIN</strong>: Style <InlineMath math="\mathbf{w}" /> modulates features via adaptive instance normalization</li>
          <li><strong>Constant input</strong>: Synthesis starts from a learned constant, not from <InlineMath math="\mathbf{z}" /></li>
          <li><strong>Noise injection</strong>: Per-pixel noise adds stochastic variation (hair, freckles)</li>
        </ul>
        <BlockMath math="\text{AdaIN}(\mathbf{x}_i, \mathbf{y}) = y_{s,i}\frac{\mathbf{x}_i - \mu(\mathbf{x}_i)}{\sigma(\mathbf{x}_i)} + y_{b,i}" />
      </DefinitionBlock>

      <StyleMixingViz />

      <ExampleBlock title="Progressive Growing to StyleGAN3">
        <p>
          <strong>ProGAN</strong>: Progressively grows resolution during training (4x4 to 1024x1024).
          <strong>StyleGAN2</strong>: Removes progressive growing, fixes water droplet artifacts with
          weight demodulation. <strong>StyleGAN3</strong>: Achieves alias-free generation with
          continuous equivariance to translation and rotation.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Simplified StyleGAN Mapping Network + AdaIN"
        code={`import torch
import torch.nn as nn

class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.extend([nn.Linear(z_dim if i == 0 else w_dim, w_dim), nn.LeakyReLU(0.2)])
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class AdaIN(nn.Module):
    def __init__(self, channels, w_dim=512):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.style = nn.Linear(w_dim, channels * 2)

    def forward(self, x, w):
        style = self.style(w).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = style.chunk(2, dim=1)
        return gamma * self.norm(x) + beta

mapping = MappingNetwork()
adain = AdaIN(channels=256)

z = torch.randn(4, 512)
w = mapping(z)
feat = torch.randn(4, 256, 16, 16)
styled = adain(feat, w)
print(f"z: {z.shape} -> w: {w.shape}")
print(f"Styled features: {styled.shape}")`}
      />

      <NoteBlock type="note" title="The W and W+ Latent Spaces">
        <p>
          The intermediate latent space <InlineMath math="\mathcal{W}" /> is less entangled than
          <InlineMath math="\mathcal{Z}" />, enabling more meaningful interpolations. The extended
          <InlineMath math="\mathcal{W}^+" /> space uses different <InlineMath math="\mathbf{w}" /> vectors per layer,
          enabling GAN inversion: finding latents that reconstruct real images for editing.
        </p>
      </NoteBlock>
    </div>
  )
}

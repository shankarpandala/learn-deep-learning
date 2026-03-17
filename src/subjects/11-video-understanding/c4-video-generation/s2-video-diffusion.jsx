import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function VideoDiffusionModels() {
  const [selected, setSelected] = useState('sora')

  const models = {
    sora: { name: 'Sora', arch: 'DiT (Diffusion Transformer)', resolution: 'Up to 1080p', duration: 'Up to 60s', conditioning: 'Text + image', keyInnovation: 'Spacetime patches, variable resolution/duration' },
    runway: { name: 'Gen-3 Alpha', arch: 'DiT with temporal layers', resolution: '1080p', duration: '~10s', conditioning: 'Text + image + video', keyInnovation: 'Fine-grained temporal control and camera motion' },
    stablevideo: { name: 'Stable Video Diffusion', arch: 'UNet with temporal convs', resolution: '576x1024', duration: '~4s (25 frames)', conditioning: 'Image (img2vid)', keyInnovation: 'Curated pre-training, motion bucket conditioning' },
    cogvideo: { name: 'CogVideoX', arch: '3D-VAE + DiT', resolution: '720p', duration: '~6s', conditioning: 'Text', keyInnovation: 'Expert Transformer with adaptive LayerNorm' },
  }

  const m = models[selected]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Video Diffusion Models</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.entries(models).map(([key, val]) => (
          <button key={key} onClick={() => setSelected(key)}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${selected === key ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-3 gap-3 text-sm">
        {[['Architecture', m.arch], ['Resolution', m.resolution], ['Duration', m.duration],
          ['Conditioning', m.conditioning], ['Key innovation', m.keyInnovation]
        ].map(([label, val]) => (
          <div key={label} className={`rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3 ${label === 'Key innovation' ? 'col-span-2' : ''}`}>
            <p className="text-xs text-violet-600 dark:text-violet-400 font-semibold">{label}</p>
            <p className="text-gray-700 dark:text-gray-300">{val}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function VideoDiffusion() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Video diffusion models generate photorealistic video from text prompts by extending
        image diffusion to the temporal dimension. Systems like Sora demonstrate emergent
        understanding of 3D geometry, physics, and object permanence from pure video generation training.
      </p>

      <DefinitionBlock title="Video Latent Diffusion">
        <p>Video diffusion operates on spatiotemporal latent representations:</p>
        <BlockMath math="\mathcal{L} = \mathbb{E}_{z_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(z_t, t, c)\|^2\right], \quad z_0 = \text{Enc}_\text{3D}(V)" />
        <p className="mt-2">
          A 3D VAE encodes the video <InlineMath math="V \in \mathbb{R}^{T \times H \times W \times 3}" /> into
          a compressed latent <InlineMath math="z_0 \in \mathbb{R}^{T' \times H' \times W' \times C}" />.
          The denoising network <InlineMath math="\epsilon_\theta" /> is typically a DiT (Diffusion
          Transformer) processing spacetime patches.
        </p>
      </DefinitionBlock>

      <VideoDiffusionModels />

      <ExampleBlock title="Spacetime Patches (Sora)">
        <p>
          Sora tokenizes video into spacetime patches, treating video like a language of visual tokens:
        </p>
        <BlockMath math="z_\text{patch} \in \mathbb{R}^{(T/t_p) \times (H/h_p) \times (W/w_p) \times D}" />
        <p className="mt-1">
          With patch size <InlineMath math="(t_p, h_p, w_p) = (1, 2, 2)" /> in latent space, a 16-frame
          512x512 video becomes ~16K tokens. The DiT processes these with full attention, learning
          spatiotemporal relationships. Variable-resolution training enables generating at any
          aspect ratio and duration.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Simplified Video Diffusion Architecture"
        code={`import torch
import torch.nn as nn

class VideoDenoiser(nn.Module):
    """Simplified DiT-style video denoiser."""
    def __init__(self, latent_ch=4, dim=512, num_heads=8, depth=4):
        super().__init__()
        # Spacetime patch embedding
        self.patch_embed = nn.Conv3d(latent_ch, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # Time embedding
        self.time_mlp = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        # Text conditioning projection
        self.text_proj = nn.Linear(768, dim)
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
            for _ in range(depth)
        ])
        # Output projection
        self.output = nn.Linear(dim, latent_ch * 4)  # 2x2 patch

    def forward(self, z_t, t, text_emb):
        B = z_t.shape[0]
        # Patchify
        patches = self.patch_embed(z_t)  # [B, D, T, H/2, W/2]
        T_p, H_p, W_p = patches.shape[2:]
        x = patches.flatten(2).permute(0, 2, 1)  # [B, N, D]
        # Add conditioning
        t_emb = self.time_mlp(t.view(B, 1))  # [B, D]
        c_emb = self.text_proj(text_emb)      # [B, D]
        x = x + (t_emb + c_emb).unsqueeze(1)
        # Transformer
        for block in self.blocks:
            x = block(x)
        # Unpatchify
        noise_pred = self.output(x)
        noise_pred = noise_pred.view(B, T_p, H_p, W_p, 4, 2, 2)
        noise_pred = noise_pred.permute(0, 4, 1, 2, 5, 3, 6)
        return noise_pred.reshape(B, 4, T_p, H_p * 2, W_p * 2)

model = VideoDenoiser()
z_t = torch.randn(2, 4, 16, 32, 32)   # noisy latent video
t = torch.rand(2)                       # diffusion timestep
text_emb = torch.randn(2, 768)          # T5 text embedding
noise_pred = model(z_t, t, text_emb)
print(f"Noisy input: {z_t.shape}")
print(f"Noise prediction: {noise_pred.shape}")  # [2, 4, 16, 32, 32]`}
      />

      <WarningBlock title="Computational Requirements">
        <p>
          Training video diffusion models requires enormous compute: Sora is estimated at
          thousands of GPU-months. A single 60-second generation may take minutes on high-end
          hardware. Inference optimization through distillation, consistency models, and
          progressive generation is an active research area.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Video as World Simulation">
        <p>
          Sora's technical report describes the model as a "world simulator" that learns physical
          rules from video data. Emergent capabilities include consistent 3D geometry across
          camera movements, object persistence through occlusion, and realistic physical
          interactions. This suggests that scaling video generation may be a path toward
          understanding the physical world.
        </p>
      </NoteBlock>
    </div>
  )
}

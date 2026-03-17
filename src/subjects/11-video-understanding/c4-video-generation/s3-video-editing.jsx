import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function EditingTechniqueExplorer() {
  const [technique, setTechnique] = useState('instruct')

  const techniques = {
    instruct: { name: 'Text-Guided Editing', method: 'Instruction-following with diffusion', temporalConsistency: 'High (shared attention)', input: 'Video + edit instruction', examples: 'InstructVid2Vid, MagicEdit', useCase: '"Make the dog golden" on a video of a dog' },
    inpaint: { name: 'Video Inpainting', method: 'Masked diffusion generation', temporalConsistency: 'Medium (propagation needed)', input: 'Video + spatiotemporal mask', examples: 'ProPainter, E2FGVI', useCase: 'Remove an object from all frames seamlessly' },
    style: { name: 'Style Transfer', method: 'Latent space manipulation', temporalConsistency: 'Medium (flickering risk)', input: 'Video + style reference', examples: 'Rerender-A-Video, CoDeF', useCase: 'Apply Van Gogh style to a home video' },
    motion: { name: 'Motion Transfer', method: 'Pose/flow-guided generation', temporalConsistency: 'High (motion prior)', input: 'Source video + target motion', examples: 'DreamPose, MotionCtrl', useCase: 'Transfer dance moves to a different person' },
  }

  const t = techniques[technique]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Video Editing Techniques</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.entries(techniques).map(([key, val]) => (
          <button key={key} onClick={() => setTechnique(key)}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${technique === key ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3 text-sm">
        {[['Method', t.method], ['Temporal consistency', t.temporalConsistency],
          ['Input', t.input], ['Example models', t.examples], ['Use case', t.useCase]
        ].map(([label, val]) => (
          <div key={label} className={`rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3 ${label === 'Use case' ? 'col-span-2' : ''}`}>
            <p className="text-xs text-violet-600 dark:text-violet-400 font-semibold">{label}</p>
            <p className="text-gray-700 dark:text-gray-300">{val}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function VideoEditingManipulation() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Neural video editing enables modifying video content through text instructions, style
        transfer, and inpainting while maintaining temporal consistency. The key challenge is
        ensuring edits are coherent across frames without flickering or artifacts.
      </p>

      <DefinitionBlock title="DDIM Inversion for Video Editing">
        <p>Video editing via diffusion models typically uses DDIM inversion to obtain an editable latent:</p>
        <BlockMath math="z_T = \text{DDIM-Inv}(z_0, \{\epsilon_\theta(\cdot, t, c_\text{src})\}_{t=1}^{T})" />
        <p className="mt-2">
          The source video is inverted to noise <InlineMath math="z_T" />, then denoised with the
          edited prompt <InlineMath math="c_\text{edit}" />. Shared self-attention keys and values
          from the source denoising process preserve structure:
        </p>
        <BlockMath math="\text{Attn}(Q_\text{edit}, K_\text{src}, V_\text{src})" />
      </DefinitionBlock>

      <EditingTechniqueExplorer />

      <ExampleBlock title="Temporal Consistency via Cross-Frame Attention">
        <p>
          To prevent flickering, video editing models replace per-frame self-attention with
          cross-frame attention. For frame <InlineMath math="t" />, keys and values come from a
          reference frame (typically the first or previous):
        </p>
        <BlockMath math="\text{Attn}_t = \text{softmax}\!\left(\frac{Q_t K_\text{ref}^\top}{\sqrt{d}}\right) V_\text{ref}" />
        <p className="mt-1">
          This forces all frames to attend to the same reference, propagating the edit
          consistently. Extended attention (attending to multiple reference frames) further
          improves long-video consistency.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Video Editing with Cross-Frame Attention"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossFrameAttention(nn.Module):
    """Self-attention that uses keys/values from a reference frame."""
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, ref_frame=None):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        if ref_frame is not None:
            # Use keys/values from reference frame
            ref_qkv = self.qkv(ref_frame).reshape(B, N, 3, self.num_heads, self.head_dim)
            _, k, v = ref_qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)

class VideoEditor(nn.Module):
    def __init__(self, dim=512, num_frames=16):
        super().__init__()
        self.num_frames = num_frames
        self.attn = CrossFrameAttention(dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def edit_video(self, source_features, edit_features):
        """Apply edit while maintaining temporal consistency."""
        B_total, N, D = edit_features.shape
        B = B_total // self.num_frames

        # Use first frame as reference for cross-frame attention
        ref = source_features[:B * 1].repeat(self.num_frames, 1, 1)

        # Cross-frame attention (edit queries, source keys/values)
        x = self.norm(edit_features)
        x = edit_features + self.attn(x, ref_frame=ref)
        x = x + self.mlp(self.norm(x))
        return x

editor = VideoEditor(dim=512, num_frames=16)
source = torch.randn(32, 196, 512)  # 16 frames * 2 batch, 14x14 patches
edited = torch.randn(32, 196, 512)  # edited version (may flicker)
consistent = editor.edit_video(source, edited)
print(f"Source features: {source.shape}")
print(f"Consistent edit: {consistent.shape}")`}
      />

      <WarningBlock title="Ethical Implications">
        <p>
          Video editing and deepfake technology raise serious ethical concerns around
          misinformation, non-consensual content, and identity theft. Research in provenance
          tracking, watermarking (C2PA standard), and deepfake detection is critical.
          Responsible deployment requires robust detection tools alongside generation capabilities.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Content-Deformation Fields (CoDeF)">
        <p>
          CoDeF represents a video as a canonical content field plus a temporal deformation field,
          enabling editing the canonical image and propagating changes consistently across all
          frames. This neural representation approach avoids per-frame processing entirely,
          achieving superior temporal consistency for style transfer and object manipulation.
        </p>
      </NoteBlock>
    </div>
  )
}

import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function AttentionSchemeComparison() {
  const [scheme, setScheme] = useState('divided')

  const schemes = {
    joint: { name: 'Joint Space-Time', complexity: 'O((T*N)^2)', memory: 'Very high', quality: 'Best (small scale)', desc: 'Full attention over all patches across all frames simultaneously' },
    divided: { name: 'Divided Space-Time', complexity: 'O(T*N^2 + N*T^2)', memory: 'Moderate', quality: 'Best (scalable)', desc: 'Separate spatial attention within frames, then temporal attention across frames' },
    sparse: { name: 'Sparse (Local+Global)', complexity: 'O(T*N*k)', memory: 'Low', quality: 'Good', desc: 'Local spatial attention + sparse global temporal attention' },
    axial: { name: 'Axial', complexity: 'O(T*N*(sqrt(N)+T))', memory: 'Low', quality: 'Good', desc: 'Factorized along height, width, and time axes independently' },
  }

  const s = schemes[scheme]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Space-Time Attention Schemes</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.entries(schemes).map(([key, val]) => (
          <button key={key} onClick={() => setScheme(key)}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${scheme === key ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="rounded-lg bg-violet-50 dark:bg-violet-900/20 p-4">
        <p className="text-sm text-gray-700 dark:text-gray-300">{s.desc}</p>
        <div className="grid grid-cols-3 gap-3 mt-3 text-sm">
          <div><span className="text-xs text-violet-600 dark:text-violet-400 font-semibold block">Complexity</span>{s.complexity}</div>
          <div><span className="text-xs text-violet-600 dark:text-violet-400 font-semibold block">Memory</span>{s.memory}</div>
          <div><span className="text-xs text-violet-600 dark:text-violet-400 font-semibold block">Quality</span>{s.quality}</div>
        </div>
      </div>
    </div>
  )
}

export default function TimeSformer() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        TimeSformer adapts Vision Transformers to video by introducing divided space-time
        attention, where spatial and temporal attention are computed separately in each block.
        This factorization makes video Transformers scalable to long clips.
      </p>

      <DefinitionBlock title="Divided Space-Time Attention">
        <p>Each TimeSformer block applies two attention operations sequentially:</p>
        <BlockMath math="z'_t = \text{TemporalAttn}(z_t) + z_t, \quad z''_t = \text{SpatialAttn}(z'_t) + z'_t" />
        <p className="mt-2">
          <strong>Temporal attention:</strong> each patch attends to the same spatial position across
          all <InlineMath math="T" /> frames. <strong>Spatial attention:</strong> patches within the same
          frame attend to each other. This avoids the <InlineMath math="O((TN)^2)" /> cost of joint attention.
        </p>
      </DefinitionBlock>

      <AttentionSchemeComparison />

      <TheoremBlock title="Computational Savings" id="timesformer-savings">
        <p>
          For a video with <InlineMath math="T" /> frames and <InlineMath math="N" /> spatial patches per frame,
          divided attention reduces complexity from quadratic to:
        </p>
        <BlockMath math="\text{Joint: } O(T^2 N^2 d) \quad \to \quad \text{Divided: } O(TN^2 d + NT^2 d)" />
        <p className="mt-1">
          With <InlineMath math="T=8, N=196" /> (ViT-B/16 on 224px), this is a
          <InlineMath math="\sim 6\times" /> reduction. The savings grow linearly with clip length.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Patch Embedding for Video">
        <p>
          TimeSformer embeds video frames independently using the same ViT patch embedding,
          then adds learnable temporal position embeddings:
        </p>
        <BlockMath math="z_{t,p} = \text{PatchEmbed}(x_{t,p}) + e_p^\text{spatial} + e_t^\text{temporal}" />
        <p className="mt-1">
          A special <InlineMath math="\texttt{[CLS]}" /> token aggregates information across all
          frames for final classification.
        </p>
      </ExampleBlock>

      <PythonCode
        title="TimeSformer Divided Attention Block"
        code={`import torch
import torch.nn as nn

class DividedSpaceTimeAttention(nn.Module):
    def __init__(self, dim=768, num_heads=12, num_frames=8):
        super().__init__()
        self.num_frames = num_frames
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_t = nn.LayerNorm(dim)
        self.norm_s = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, T*N, D] where T=frames, N=spatial patches
        B, TN, D = x.shape
        T = self.num_frames
        N = TN // T

        # Temporal attention: group by spatial position
        xt = x.view(B, T, N, D).permute(0, 2, 1, 3).reshape(B * N, T, D)
        xt = self.norm_t(xt)
        temporal_out, _ = self.temporal_attn(xt, xt, xt)
        temporal_out = temporal_out.reshape(B, N, T, D).permute(0, 2, 1, 3).reshape(B, TN, D)
        x = x + temporal_out

        # Spatial attention: group by frame
        xs = x.view(B, T, N, D).reshape(B * T, N, D)
        xs = self.norm_s(xs)
        spatial_out, _ = self.spatial_attn(xs, xs, xs)
        spatial_out = spatial_out.reshape(B, T, N, D).reshape(B, TN, D)
        x = x + spatial_out

        return x

# Example usage
T, N, D = 8, 196, 768  # 8 frames, 14x14 patches, ViT-Base dim
block = DividedSpaceTimeAttention(dim=D, num_heads=12, num_frames=T)
x = torch.randn(2, T * N, D)
out = block(x)
print(f"Input: {x.shape}")   # [2, 1568, 768]
print(f"Output: {out.shape}")  # [2, 1568, 768]

# Compare parameter counts
joint_params = sum(p.numel() for p in nn.MultiheadAttention(D, 12).parameters())
divided_params = sum(p.numel() for p in block.parameters())
print(f"Joint attn params: {joint_params:,}")
print(f"Divided attn params: {divided_params:,} (2x due to dual attention)")`}
      />

      <NoteBlock type="note" title="From TimeSformer to Efficient Video Transformers">
        <p>
          TimeSformer demonstrated that divided attention matches or exceeds joint attention while
          being far more scalable. This insight led to many efficient video Transformer designs:
          <strong>ViViT</strong> (factorized encoder), <strong>MViT</strong> (pooling attention),
          and <strong>VideoSwin</strong> (shifted window attention in 3D). The common theme is
          exploiting spatiotemporal locality to avoid full attention over all tokens.
        </p>
      </NoteBlock>
    </div>
  )
}

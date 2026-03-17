import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ViViTVariantExplorer() {
  const [variant, setVariant] = useState('factorized_enc')

  const variants = {
    spatio_temporal: { name: 'Model 1: Spatio-temporal', tokens: 'T*N', encoders: '1 Transformer', tubelets: 'Yes', complexity: 'O((T*N)^2)', desc: 'All tokens attend to each other in a single Transformer' },
    factorized_enc: { name: 'Model 2: Factorized Encoder', tokens: 'N then T', encoders: 'Spatial + Temporal', tubelets: 'Yes', complexity: 'O(N^2 + T^2)', desc: 'Spatial encoder per frame, then temporal encoder over CLS tokens' },
    factorized_self: { name: 'Model 3: Factorized Self-Attn', tokens: 'T*N', encoders: '1 Transformer (factorized)', tubelets: 'Yes', complexity: 'O(T*N^2 + N*T^2)', desc: 'Like TimeSformer divided attention within a single model' },
    factorized_dot: { name: 'Model 4: Factorized Dot-Product', tokens: 'T*N', encoders: '1 Transformer (factorized KV)', tubelets: 'Yes', complexity: 'O(T*N*(N+T))', desc: 'Separate spatial/temporal heads with concatenated outputs' },
  }

  const v = variants[variant]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">ViViT Architecture Variants</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.entries(variants).map(([key, val]) => (
          <button key={key} onClick={() => setVariant(key)}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${variant === key ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="rounded-lg bg-violet-50 dark:bg-violet-900/20 p-4">
        <p className="text-sm text-gray-700 dark:text-gray-300">{v.desc}</p>
        <div className="grid grid-cols-2 gap-3 mt-3 text-sm">
          {[['Encoders', v.encoders], ['Complexity', v.complexity], ['Tubelet embedding', v.tubelets], ['Token count', v.tokens]].map(([label, val]) => (
            <div key={label}><span className="text-xs text-violet-600 dark:text-violet-400 font-semibold block">{label}</span><span className="text-gray-600 dark:text-gray-400">{val}</span></div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default function ViViTVideoMAE() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        ViViT explores multiple strategies for applying Vision Transformers to video, introducing
        tubelet embeddings for spatiotemporal tokenization. VideoMAE extends masked autoencoders
        to video, achieving strong results with minimal labeled data.
      </p>

      <DefinitionBlock title="Tubelet Embedding">
        <p>Instead of embedding 2D patches per frame, ViViT embeds 3D tubelets:</p>
        <BlockMath math="z_{t,p} = \text{Linear}(\text{flatten}(x[t:t{+}t_s, p_h:p_h{+}h, p_w:p_w{+}w]))" />
        <p className="mt-2">
          A tubelet of size <InlineMath math="t_s \times h \times w" /> (e.g., <InlineMath math="2 \times 16 \times 16" />)
          captures local spatiotemporal patterns in a single token, reducing the total token count
          by a factor of <InlineMath math="t_s" /> compared to frame-level patch embedding.
        </p>
      </DefinitionBlock>

      <ViViTVariantExplorer />

      <TheoremBlock title="VideoMAE: Masked Video Pre-training" id="videomae">
        <p>
          VideoMAE masks a very high ratio (90-95%) of video tokens and trains the encoder
          to reconstruct them:
        </p>
        <BlockMath math="\mathcal{L} = \frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} \|x_i - \hat{x}_i\|^2" />
        <p className="mt-1">
          The extreme masking ratio works because video is highly redundant temporally. Tube masking
          ensures consistent masks across frames, preventing trivial solutions from temporal interpolation.
          After pre-training, the encoder is fine-tuned for downstream tasks.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Why 90% Masking Works for Video">
        <p>
          Adjacent video frames differ by only a few pixels. If masking is applied independently
          per frame, the model can "cheat" by copying from unmasked patches in nearby frames.
          Tube masking forces the model to reason about motion and semantics:
        </p>
        <BlockMath math="\text{Tube mask: } \mathcal{M}_t = \mathcal{M}_0 \quad \forall t \in \{0, \ldots, T{-}1\}" />
        <p className="mt-1">
          The same spatial positions are masked across all frames, requiring genuine understanding
          to reconstruct.
        </p>
      </ExampleBlock>

      <PythonCode
        title="ViViT Factorized Encoder"
        code={`import torch
import torch.nn as nn

class ViViTFactorized(nn.Module):
    """ViViT Model 2: Factorized Encoder."""
    def __init__(self, num_frames=8, num_patches=196, dim=768, num_classes=400):
        super().__init__()
        self.num_frames = num_frames
        # Tubelet embedding: 2x16x16 tubelets
        self.patch_embed = nn.Conv3d(3, dim, kernel_size=(2, 16, 16), stride=(2, 16, 16))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # Spatial encoder (processes each frame independently)
        spatial_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=12, batch_first=True)
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, num_layers=6)

        # Temporal encoder (processes CLS tokens across frames)
        temporal_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=12, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=4)
        self.temporal_pos = nn.Parameter(torch.randn(1, num_frames // 2, dim))

        self.head = nn.Linear(dim, num_classes)

    def forward(self, video):  # video: [B, C, T, H, W]
        B = video.shape[0]
        # Tubelet embedding
        x = self.patch_embed(video)  # [B, D, T', H', W']
        T_out = x.shape[2]
        x = x.flatten(3).permute(0, 2, 3, 1)  # [B, T', N, D]

        # Spatial encoding per frame
        cls_tokens = []
        for t in range(T_out):
            frame_tokens = x[:, t]  # [B, N, D]
            cls = self.cls_token.expand(B, -1, -1)
            frame_tokens = torch.cat([cls, frame_tokens], dim=1) + self.pos_embed
            encoded = self.spatial_encoder(frame_tokens)
            cls_tokens.append(encoded[:, 0])  # CLS token

        # Temporal encoding over CLS tokens
        temporal_tokens = torch.stack(cls_tokens, dim=1) + self.temporal_pos
        temporal_out = self.temporal_encoder(temporal_tokens)

        # Classify from mean-pooled temporal CLS tokens
        return self.head(temporal_out.mean(dim=1))

model = ViViTFactorized(num_frames=8, num_patches=196, dim=768, num_classes=400)
video = torch.randn(2, 3, 8, 224, 224)
logits = model(video)
print(f"Classification logits: {logits.shape}")  # [2, 400]
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")`}
      />

      <NoteBlock type="note" title="Pre-training Data Efficiency">
        <p>
          VideoMAE demonstrates remarkable data efficiency: pre-training on just 3,000 videos from
          Kinetics-400 and fine-tuning on the full set achieves competitive accuracy. This suggests
          that video's temporal redundancy makes self-supervised learning particularly effective,
          requiring far less data than image-based MAE for comparable gains.
        </p>
      </NoteBlock>
    </div>
  )
}

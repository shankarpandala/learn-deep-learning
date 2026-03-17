import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ModalityRouter() {
  const [modality, setModality] = useState('text')
  const modalities = {
    text: { tokens: '~500 tokens', encoder: 'SentencePiece tokenizer', color: 'violet' },
    image: { tokens: '~256 tokens', encoder: 'ViT patches + resampler', color: 'violet' },
    audio: { tokens: '~128 tokens', encoder: 'Whisper encoder + projection', color: 'violet' },
    video: { tokens: '~1024 tokens', encoder: 'Per-frame ViT + temporal sampling', color: 'violet' },
  }
  const m = modalities[modality]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Unified Multimodal Tokenization</h3>
      <div className="flex gap-2 mb-4 flex-wrap">
        {Object.keys(modalities).map(mod => (
          <button key={mod} onClick={() => setModality(mod)}
            className={`px-3 py-1 rounded-lg text-sm capitalize transition ${modality === mod ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {mod}
          </button>
        ))}
      </div>
      <div className="p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20 space-y-1 text-sm">
        <p><strong>Modality:</strong> {modality}</p>
        <p><strong>Encoder:</strong> {m.encoder}</p>
        <p><strong>Typical sequence length:</strong> {m.tokens}</p>
        <p className="text-xs text-gray-500 mt-2">All modalities are projected to the same transformer embedding dimension</p>
      </div>
    </div>
  )
}

export default function UnifiedMultimodalModels() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Models like Gemini, GPT-4V, and Claude natively process multiple modalities within a single
        transformer architecture. Rather than bolting separate encoders together, these models
        are trained from the ground up on interleaved multimodal data.
      </p>

      <DefinitionBlock title="Native Multimodal Architecture">
        <p>A unified model processes all modalities as token sequences in a shared transformer:</p>
        <BlockMath math="h = \text{Transformer}(\text{concat}(E_{\text{text}}(x_t), E_{\text{image}}(x_i), E_{\text{audio}}(x_a), \ldots))" />
        <p className="mt-2">Each modality has its own encoder <InlineMath math="E_m" /> that produces tokens in a shared embedding space of dimension <InlineMath math="d" />. The transformer attends across all tokens regardless of modality.</p>
      </DefinitionBlock>

      <ModalityRouter />

      <ExampleBlock title="Gemini Architecture Insights">
        <p>Gemini (Google DeepMind) is natively multimodal from pre-training:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>Trained on interleaved text, image, audio, and video from the start</li>
          <li>Uses SentencePiece for text, ViT-style patches for images</li>
          <li>Gemini Ultra: estimated 1.56T parameters (MoE architecture)</li>
          <li>Can generate both text and images as output</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Unified Multimodal Sequence Construction"
        code={`import torch
import torch.nn as nn

class UnifiedMultimodalEncoder(nn.Module):
    """Simplified unified encoder for multiple modalities."""
    def __init__(self, d_model=768):
        super().__init__()
        self.text_embed = nn.Embedding(32000, d_model)
        self.image_proj = nn.Linear(1024, d_model)  # from ViT
        self.audio_proj = nn.Linear(512, d_model)    # from audio encoder
        self.modality_embed = nn.Embedding(3, d_model)  # text=0, image=1, audio=2

    def forward(self, text_ids=None, image_feats=None, audio_feats=None):
        tokens = []
        if image_feats is not None:
            img_tok = self.image_proj(image_feats) + self.modality_embed(
                torch.ones(image_feats.shape[:2], dtype=torch.long, device=image_feats.device))
            tokens.append(img_tok)
        if text_ids is not None:
            txt_tok = self.text_embed(text_ids) + self.modality_embed(
                torch.zeros_like(text_ids))
            tokens.append(txt_tok)
        if audio_feats is not None:
            aud_tok = self.audio_proj(audio_feats) + self.modality_embed(
                2 * torch.ones(audio_feats.shape[:2], dtype=torch.long, device=audio_feats.device))
            tokens.append(aud_tok)
        return torch.cat(tokens, dim=1)  # [B, total_tokens, D]

enc = UnifiedMultimodalEncoder()
text = torch.randint(0, 32000, (1, 64))
image = torch.randn(1, 256, 1024)
combined = enc(text_ids=text, image_feats=image)
print(f"Combined multimodal sequence: {combined.shape}")  # [1, 320, 768]`}
      />

      <NoteBlock type="note" title="Early vs Late Fusion">
        <p>
          Unified models use <strong>early fusion</strong> — modalities interact from the first
          transformer layer. This contrasts with late fusion approaches (like CLIP) where modalities
          are encoded independently and only interact at the final embedding. Early fusion enables
          richer cross-modal reasoning but requires training on paired multimodal data.
        </p>
      </NoteBlock>

      <NoteBlock type="warning" title="Challenges of Unified Models">
        <p>
          Natively multimodal training introduces unique challenges: (1) balancing loss across
          modalities — text tends to dominate due to higher data volume, (2) tokenization
          inconsistencies between modalities, (3) evaluation is harder since capabilities span
          many benchmarks. Despite these challenges, the trend is clearly toward unified
          architectures that can seamlessly reason across modalities in a single forward pass.
        </p>
      </NoteBlock>
    </div>
  )
}

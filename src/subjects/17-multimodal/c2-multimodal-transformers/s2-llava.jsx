import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function TrainingStageViz() {
  const [stage, setStage] = useState(0)
  const stages = [
    { name: 'Stage 1: Alignment Pre-training', frozen: ['Vision Encoder', 'LLM'], trained: ['Projection'], data: '558K image-caption pairs', desc: 'Train only the projection layer to align visual features with the LLM input space.' },
    { name: 'Stage 2: Visual Instruction Tuning', frozen: ['Vision Encoder'], trained: ['Projection', 'LLM'], data: '665K instruction-following data', desc: 'Fine-tune the LLM and projection on multimodal instruction data.' },
  ]
  const s = stages[stage]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">LLaVA Two-Stage Training</h3>
      <div className="flex gap-2 mb-4">
        {stages.map((st, i) => (
          <button key={i} onClick={() => setStage(i)}
            className={`px-3 py-1 rounded-lg text-sm transition ${stage === i ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            Stage {i + 1}
          </button>
        ))}
      </div>
      <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">{s.name}</p>
      <div className="flex gap-3 mb-2 flex-wrap">
        {s.frozen.map(m => <span key={m} className="px-2 py-1 rounded text-xs bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400">&#10052; {m} (frozen)</span>)}
        {s.trained.map(m => <span key={m} className="px-2 py-1 rounded text-xs bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300">&#9998; {m} (trained)</span>)}
      </div>
      <p className="text-xs text-gray-500">Data: {s.data}</p>
      <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">{s.desc}</p>
    </div>
  )
}

export default function LLaVAVisualInstructionTuning() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        LLaVA (Large Language-and-Vision Assistant) connects a CLIP vision encoder to a language model
        via a simple linear projection, then fine-tunes on visual instruction-following data. Its
        simplicity and effectiveness made it a foundational architecture for open-source multimodal LLMs.
      </p>

      <DefinitionBlock title="LLaVA Architecture">
        <p>LLaVA processes an image through a vision encoder and projects visual tokens into the LLM embedding space:</p>
        <BlockMath math="H_v = W \cdot f_{\text{CLIP}}(I), \quad H_v \in \mathbb{R}^{N_v \times d}" />
        <p className="mt-2">The visual tokens <InlineMath math="H_v" /> are prepended to the text token embeddings and fed to the LLM as a unified sequence. The projection <InlineMath math="W" /> is a learned linear layer (or MLP in LLaVA-1.5).</p>
      </DefinitionBlock>

      <TrainingStageViz />

      <PythonCode
        title="LLaVA-Style Visual Projection"
        code={`import torch
import torch.nn as nn

class LLaVAProjection(nn.Module):
    """MLP projection from vision to language space (LLaVA-1.5)."""
    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, vision_features):
        return self.proj(vision_features)

def prepare_multimodal_input(vision_features, text_embeds, projector):
    """Concatenate projected visual tokens with text embeddings."""
    visual_tokens = projector(vision_features)  # [B, N_v, D_llm]
    # Prepend visual tokens to text sequence
    combined = torch.cat([visual_tokens, text_embeds], dim=1)
    return combined

proj = LLaVAProjection()
vis = torch.randn(1, 576, 1024)    # 576 patches from ViT-L/14@336px
txt = torch.randn(1, 128, 4096)    # text embeddings
combined = prepare_multimodal_input(vis, txt, proj)
print(f"Combined sequence: {combined.shape}")  # [1, 704, 4096]`}
      />

      <ExampleBlock title="Visual Instruction Data Generation">
        <p>LLaVA's key innovation was using GPT-4 to generate visual instruction-following data:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>158K unique images from COCO with bounding boxes and captions</li>
          <li>GPT-4 generates conversations, descriptions, and reasoning about images</li>
          <li>Three types: conversation (58K), detailed description (23K), complex reasoning (77K)</li>
        </ul>
      </ExampleBlock>

      <WarningBlock title="Data Quality vs Quantity">
        <p>
          LLaVA demonstrates that high-quality instruction data matters more than quantity.
          665K carefully curated examples outperform millions of noisy web-scraped pairs.
          Data quality and diversity of instruction types are critical for generalization.
        </p>
      </WarningBlock>
    </div>
  )
}

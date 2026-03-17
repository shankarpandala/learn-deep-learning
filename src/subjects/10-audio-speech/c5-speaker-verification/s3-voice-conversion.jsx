import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function VCApproachSelector() {
  const [approach, setApproach] = useState('disentangle')

  const approaches = {
    disentangle: { name: 'Disentanglement', desc: 'Separate content and speaker representations, swap speaker embedding', pros: 'Clean separation, controllable', cons: 'May lose prosody nuances', examples: 'AutoVC, VQVC+' },
    cyclegan: { name: 'CycleGAN-VC', desc: 'Unpaired voice conversion using cycle-consistency loss', pros: 'No parallel data needed', cons: 'Limited quality, mode collapse risk', examples: 'CycleGAN-VC2, CycleGAN-VC3' },
    diffusion: { name: 'Diffusion-based', desc: 'Convert via diffusion process conditioned on target speaker', pros: 'High quality, flexible', cons: 'Slow inference', examples: 'DiffVC, CoMoSpeech' },
    codec: { name: 'Codec-based', desc: 'Manipulate neural audio codec tokens to change speaker identity', pros: 'Real-time possible, high quality', cons: 'Requires good codec', examples: 'FreeVC, VALL-E X' },
  }

  const a = approaches[approach]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Voice Conversion Approaches</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.entries(approaches).map(([key, val]) => (
          <button key={key} onClick={() => setApproach(key)}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${approach === key ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="rounded-lg bg-violet-50 dark:bg-violet-900/20 p-4 space-y-2">
        <p className="text-sm text-gray-700 dark:text-gray-300">{a.desc}</p>
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div><span className="font-semibold text-green-600">Pros:</span> <span className="text-gray-600 dark:text-gray-400">{a.pros}</span></div>
          <div><span className="font-semibold text-red-500">Cons:</span> <span className="text-gray-600 dark:text-gray-400">{a.cons}</span></div>
          <div><span className="font-semibold text-violet-600">Examples:</span> <span className="text-gray-600 dark:text-gray-400">{a.examples}</span></div>
        </div>
      </div>
    </div>
  )
}

export default function VoiceConversion() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Voice conversion transforms the speaker identity of an utterance while preserving
        its linguistic content. This requires disentangling what is said from who says it,
        a fundamental challenge in speech representation learning.
      </p>

      <DefinitionBlock title="Voice Conversion Objective">
        <p>Given source speech <InlineMath math="x_s" /> from speaker <InlineMath math="s" /> and target speaker <InlineMath math="t" />:</p>
        <BlockMath math="\hat{x}_t = G(c(x_s), e_t)" />
        <p className="mt-2">
          where <InlineMath math="c(x_s)" /> extracts content (linguistic information)
          and <InlineMath math="e_t" /> is the target speaker embedding. The ideal conversion
          should satisfy: same content as source, same voice as target,
          natural prosody.
        </p>
      </DefinitionBlock>

      <VCApproachSelector />

      <ExampleBlock title="AutoVC: Information Bottleneck">
        <p>AutoVC uses a carefully tuned bottleneck to force content-speaker disentanglement:</p>
        <BlockMath math="\mathcal{L} = \|x - \text{Dec}(\text{Enc}(x)_{:\text{dim}_c}, e_\text{spk})\|^2" />
        <p className="mt-1">
          The encoder output is downsampled to a bottleneck dimension that is large enough
          to preserve phonetic content but too small to encode speaker identity. The decoder
          must rely on the separately-provided speaker embedding <InlineMath math="e_\text{spk}" />.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Simple Voice Conversion with Disentanglement"
        code={`import torch
import torch.nn as nn

class SimpleVoiceConverter(nn.Module):
    def __init__(self, mel_dim=80, content_dim=32, spk_dim=192):
        super().__init__()
        # Content encoder (bottleneck forces content-only)
        self.content_enc = nn.Sequential(
            nn.Conv1d(mel_dim, 256, 5, padding=2), nn.ReLU(),
            nn.Conv1d(256, 128, 5, padding=2), nn.ReLU(),
            nn.Conv1d(128, content_dim, 1),  # bottleneck
        )
        # Decoder: content + speaker -> mel
        self.decoder = nn.Sequential(
            nn.Conv1d(content_dim + spk_dim, 256, 5, padding=2), nn.ReLU(),
            nn.Conv1d(256, 256, 5, padding=2), nn.ReLU(),
            nn.Conv1d(256, mel_dim, 1),
        )

    def encode_content(self, mel):
        return self.content_enc(mel)

    def decode(self, content, spk_emb):
        # Expand speaker embedding to match time dimension
        spk = spk_emb.unsqueeze(-1).expand(-1, -1, content.size(-1))
        return self.decoder(torch.cat([content, spk], dim=1))

    def convert(self, source_mel, target_spk_emb):
        content = self.encode_content(source_mel)
        return self.decode(content, target_spk_emb)

model = SimpleVoiceConverter()
source_mel = torch.randn(1, 80, 200)  # source utterance
target_spk = torch.randn(1, 192)      # target speaker embedding

# Voice conversion
converted = model.convert(source_mel, target_spk)
print(f"Source mel: {source_mel.shape}")
print(f"Converted mel: {converted.shape}")  # [1, 80, 200]

# Self-reconstruction for training
source_spk = torch.randn(1, 192)
reconstructed = model.convert(source_mel, source_spk)
recon_loss = nn.functional.mse_loss(reconstructed, source_mel)
print(f"Reconstruction loss: {recon_loss.item():.4f}")`}
      />

      <WarningBlock title="Ethical Considerations">
        <p>
          Voice conversion technology can be misused for voice spoofing, fraud, and deepfakes.
          Research in <strong>anti-spoofing</strong> and <strong>deepfake detection</strong> is critical.
          The ASVspoof challenge evaluates countermeasures against synthetic speech attacks.
          Responsible development requires watermarking and detection capabilities.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Zero-Shot Voice Conversion">
        <p>
          Modern systems like FreeVC and kNN-VC enable conversion to any target speaker from
          just a few seconds of reference audio, without retraining. These leverage pre-trained
          self-supervised features (from WavLM or HuBERT) which naturally disentangle content
          from speaker characteristics at different network layers.
        </p>
      </NoteBlock>
    </div>
  )
}

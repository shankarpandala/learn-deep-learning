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
        title="Voice Conversion with OpenVoice / so-vits-svc"
        code={`# OpenVoice: instant voice cloning and conversion
# pip install openvoice-cli
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
import torch

# Step 1: Generate base speech in any language
base_tts = BaseSpeakerTTS(config_path="config.json", device="cuda")
base_tts.tts(
    text="Voice conversion preserves content, changes identity.",
    output_path="base_speech.wav",
    speaker="default",
)

# Step 2: Convert tone color to match target speaker
converter = ToneColorConverter(config_path="converter_config.json")
# Extract target speaker embedding from reference audio
target_se = converter.extract_se("target_speaker.wav")
# Apply voice conversion
converter.convert(
    audio_src_path="base_speech.wav",
    src_se=converter.extract_se("base_speech.wav"),
    tgt_se=target_se,
    output_path="converted.wav",
)

# Alternative: FreeVC (HuBERT content + speaker embedding)
# Uses self-supervised features for natural disentanglement
from speechbrain.inference.speaker import EncoderClassifier
spk_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)
# HuBERT layer 12 -> content features (speaker-invariant)
# ECAPA-TDNN -> speaker embedding (content-invariant)
# Decoder: content + speaker -> converted waveform
print("FreeVC disentangles content (HuBERT) from speaker (ECAPA)")
print("No parallel data needed, zero-shot voice conversion")`}
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

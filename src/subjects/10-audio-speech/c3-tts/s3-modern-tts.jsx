import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ModernTTSComparison() {
  const [selected, setSelected] = useState('vits')

  const models = {
    vits: { name: 'VITS', type: 'VAE + Flow + GAN', endToEnd: true, streaming: false, zeroshort: false, quality: 'Excellent', speed: 'Real-time' },
    valle: { name: 'VALL-E', type: 'Codec language model', endToEnd: true, streaming: true, zeroshort: true, quality: 'Near-human', speed: 'Moderate' },
    voicebox: { name: 'Voicebox', type: 'Flow matching', endToEnd: true, streaming: false, zeroshort: true, quality: 'Near-human', speed: 'Fast' },
    styletts2: { name: 'StyleTTS 2', type: 'Diffusion + style', endToEnd: true, streaming: false, zeroshort: false, quality: 'Human-level', speed: 'Real-time' },
  }

  const m = models[selected]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Modern TTS Systems</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.entries(models).map(([key, val]) => (
          <button key={key} onClick={() => setSelected(key)}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${selected === key ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-3 gap-3 text-sm">
        {[['Architecture', m.type], ['End-to-end', m.endToEnd ? 'Yes' : 'No'], ['Zero-shot', m.zeroshort ? 'Yes' : 'No'],
          ['Quality', m.quality], ['Speed', m.speed], ['Streaming', m.streaming ? 'Yes' : 'No']
        ].map(([label, val]) => (
          <div key={label} className="rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3">
            <p className="text-xs text-violet-600 dark:text-violet-400 font-semibold">{label}</p>
            <p className="text-gray-700 dark:text-gray-300 font-medium">{val}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function ModernTTS() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Modern TTS systems have achieved human-level naturalness through end-to-end architectures
        that combine variational inference, normalizing flows, and codec language modeling,
        enabling zero-shot voice cloning from just seconds of reference audio.
      </p>

      <DefinitionBlock title="VITS: Variational Inference with Adversarial Learning">
        <p>VITS combines a VAE, normalizing flow, and HiFi-GAN in a single end-to-end model:</p>
        <BlockMath math="\mathcal{L}_\text{VITS} = \mathcal{L}_\text{recon} + D_\text{KL}(q(z|x) \| p(z|c)) + \mathcal{L}_\text{adv} + \mathcal{L}_\text{dur}" />
        <p className="mt-2">
          The posterior encoder maps from linear spectrograms to latent <InlineMath math="z" />,
          the prior encoder maps from text to the same latent space via normalizing flows,
          and the decoder generates waveforms directly from <InlineMath math="z" />.
        </p>
      </DefinitionBlock>

      <ModernTTSComparison />

      <TheoremBlock title="VALL-E: Language Model Approach" id="valle-approach">
        <p>
          VALL-E frames TTS as a conditional language model over neural audio codec tokens.
          Given a 3-second enrollment clip, it generates speech tokens autoregressively:
        </p>
        <BlockMath math="P(\mathbf{c} | \mathbf{t}, \tilde{\mathbf{c}}) = \prod_{j=1}^{8} P(c^j | c^{<j}, \mathbf{t}, \tilde{\mathbf{c}})" />
        <p className="mt-1">
          where <InlineMath math="c^j" /> are the <InlineMath math="j" />-th codebook tokens,
          <InlineMath math="\mathbf{t}" /> is text, and <InlineMath math="\tilde{\mathbf{c}}" /> is the
          enrollment audio codec. This enables zero-shot voice cloning.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Zero-Shot Voice Cloning">
        <p>Given a 3-second reference clip of an unseen speaker, VALL-E can:</p>
        <ul className="list-disc pl-5 mt-2 space-y-1">
          <li>Preserve the speaker's voice characteristics (timbre, pitch range)</li>
          <li>Maintain emotional tone and speaking style</li>
          <li>Generate arbitrary text in that voice</li>
          <li>Handle multiple languages with a multilingual variant</li>
        </ul>
        <p className="mt-2">
          The key insight: discrete audio tokens allow treating speech as a language modeling problem,
          leveraging the power of large Transformer LMs.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Modern TTS with VITS (Coqui TTS)"
        code={`from TTS.api import TTS
import torch

# VITS: end-to-end TTS (VAE + Flow + GAN, no separate vocoder)
tts = TTS(model_name="tts_models/en/ljspeech/vits")

# Single-step inference: text -> waveform (no mel intermediate)
wav = tts.tts("VITS combines variational inference with adversarial training.")
print(f"VITS output: {len(wav)} samples ({len(wav)/22050:.2f}s)")

# Zero-shot voice cloning with YourTTS
tts_clone = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")
wav = tts_clone.tts(
    text="I can speak in any voice from a short reference clip.",
    speaker_wav="reference_audio.wav",  # 3-10s reference
    language="en",
)

# XTTS v2: latest codec-based model with streaming support
tts_xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
wav = tts_xtts.tts(
    text="This model uses neural audio codecs for high quality.",
    speaker_wav="reference_audio.wav",
    language="en",
)
print(f"XTTS output: {len(wav)} samples")
print("Supports: 17 languages, voice cloning, streaming inference")`}
      />

      <NoteBlock type="note" title="The Codec Language Model Paradigm">
        <p>
          The shift from mel spectrogram prediction to <strong>neural audio codec token prediction</strong> is
          transforming TTS. By using Encodec or SoundStream tokens, speech synthesis becomes a
          sequence-to-sequence language modeling task, enabling scaling laws similar to LLMs and
          naturally supporting zero-shot capabilities through in-context learning.
        </p>
      </NoteBlock>
    </div>
  )
}

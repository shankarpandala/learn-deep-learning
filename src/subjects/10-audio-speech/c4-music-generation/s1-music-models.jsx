import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function MusicModelComparison() {
  const [selected, setSelected] = useState('jukebox')

  const models = {
    musenet: { name: 'MuseNet', arch: 'Sparse Transformer', input: 'MIDI tokens', output: 'MIDI', training: 'Autoregressive LM', duration: '~4 min', quality: 'Good (symbolic)' },
    jukebox: { name: 'Jukebox', arch: 'VQ-VAE + Transformer', input: 'Raw audio', output: 'Raw audio', training: 'Hierarchical VQ-VAE + autoregressive priors', duration: '~1 min', quality: 'Good (audio artifacts)' },
    musiclm: { name: 'MusicLM', arch: 'AudioLM + MuLan', input: 'Text description', output: 'Audio tokens', training: 'Hierarchical token prediction', duration: '~30s', quality: 'High fidelity' },
    musicgen: { name: 'MusicGen', arch: 'Single Transformer', input: 'Text / melody', output: 'Codec tokens', training: 'Codebook delay pattern', duration: '~30s', quality: 'High fidelity' },
  }

  const m = models[selected]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Music Generation Models</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.entries(models).map(([key, val]) => (
          <button key={key} onClick={() => setSelected(key)}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${selected === key ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-3 gap-3 text-sm">
        {[['Architecture', m.arch], ['Input', m.input], ['Output', m.output],
          ['Training', m.training], ['Max duration', m.duration], ['Quality', m.quality]
        ].map(([label, val]) => (
          <div key={label} className="rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3">
            <p className="text-xs text-violet-600 dark:text-violet-400 font-semibold">{label}</p>
            <p className="text-gray-700 dark:text-gray-300">{val}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function MusicGenerationModels() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Music generation has evolved from symbolic MIDI models to systems that produce
        high-fidelity audio directly from text descriptions, leveraging advances in audio
        tokenization and large-scale language modeling.
      </p>

      <DefinitionBlock title="Jukebox: Hierarchical VQ-VAE">
        <p>Jukebox uses a three-level VQ-VAE to compress raw audio at different temporal resolutions:</p>
        <BlockMath math="x \xrightarrow{\text{Enc}_1} z_1 \xrightarrow{\text{Enc}_2} z_2 \xrightarrow{\text{Enc}_3} z_3" />
        <p className="mt-2">
          Level 3 captures high-level musical structure at 8x compression, while level 1 captures
          fine acoustic details. Autoregressive Transformers generate tokens top-down:
          <InlineMath math="z_3 \to z_2 \to z_1 \to \hat{x}" />.
        </p>
      </DefinitionBlock>

      <MusicModelComparison />

      <TheoremBlock title="MusicGen Codebook Interleaving" id="musicgen-interleave">
        <p>
          MusicGen avoids the need for multiple Transformer passes by interleaving codebook
          tokens with a delay pattern. For <InlineMath math="K" /> codebooks, each timestep
          <InlineMath math="t" /> generates codebook <InlineMath math="k" /> at position <InlineMath math="t - k" />:
        </p>
        <BlockMath math="P(c_{t,k} | c_{<t}, \text{text}) \quad \text{with delay } d_k = k" />
        <p className="mt-1">
          This reduces <InlineMath math="K" /> sequential decoding passes to a single pass with only
          <InlineMath math="K{-}1" /> steps of additional latency.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Text-to-Music Pipeline">
        <p>A modern text-to-music system follows these steps:</p>
        <ol className="list-decimal pl-5 mt-2 space-y-1">
          <li>Encode text description with a text encoder (T5 or CLAP)</li>
          <li>Generate audio tokens conditioned on text embeddings</li>
          <li>Decode tokens to waveform using neural audio codec decoder</li>
          <li>Optional: apply post-processing (loudness normalization, effects)</li>
        </ol>
        <BlockMath math="\text{``upbeat jazz piano''} \xrightarrow{T5} e_\text{text} \xrightarrow{\text{Transformer}} c_{1:T} \xrightarrow{\text{Encodec}} \hat{x}" />
      </ExampleBlock>

      <PythonCode
        title="Music Generation with MusicGen"
        code={`import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# Load MusicGen model
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# Text-conditioned generation
inputs = processor(
    text=["upbeat jazz piano solo", "calm ambient electronic"],
    padding=True,
    return_tensors="pt",
)

# Generate 8 seconds of audio at 32kHz
audio_values = model.generate(**inputs, max_new_tokens=256)
print(f"Generated audio: {audio_values.shape}")
# Shape: [2, 1, 256000] (2 samples, mono, 8s * 32kHz)

# Sampling rate for MusicGen
sampling_rate = model.config.audio_encoder.sampling_rate
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Duration: {audio_values.shape[-1] / sampling_rate:.1f}s")`}
      />

      <NoteBlock type="note" title="Symbolic vs Audio Generation">
        <p>
          <strong>Symbolic models</strong> (MuseNet, Music Transformer) generate MIDI and offer
          precise control over notes, instruments, and structure, but require a separate synthesizer.
          <strong>Audio models</strong> (Jukebox, MusicLM, MusicGen) generate waveforms directly with
          realistic timbres but less structural control. Hybrid approaches are an active research area.
        </p>
      </NoteBlock>
    </div>
  )
}

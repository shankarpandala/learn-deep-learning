import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function TTSPipelineExplorer() {
  const [stage, setStage] = useState(0)

  const stages = [
    { name: 'Text Input', desc: 'Raw text or phoneme sequence', output: '"Hello world" or /h ə l oʊ w ɜːr l d/', color: 'bg-gray-100 dark:bg-gray-800' },
    { name: 'Encoder', desc: 'Character/phoneme embeddings + Transformer/LSTM', output: 'Hidden states [seq_len, 512]', color: 'bg-violet-100 dark:bg-violet-900/30' },
    { name: 'Attention', desc: 'Location-sensitive attention for monotonic alignment', output: 'Context vector per mel frame', color: 'bg-violet-200 dark:bg-violet-900/40' },
    { name: 'Decoder', desc: 'Autoregressive mel prediction (2 frames/step)', output: 'Mel spectrogram [80, T]', color: 'bg-violet-300 dark:bg-violet-800/40' },
    { name: 'Vocoder', desc: 'WaveNet / HiFi-GAN converts mel to waveform', output: 'Audio waveform [1, T*256]', color: 'bg-violet-400 dark:bg-violet-700/40' },
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Tacotron 2 Pipeline</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {stages.map((s, i) => (
          <button key={i} onClick={() => setStage(i)}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${i === stage ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400'}`}>
            {i + 1}. {s.name}
          </button>
        ))}
      </div>
      <div className={`rounded-lg p-4 ${stages[stage].color}`}>
        <p className="font-semibold text-violet-700 dark:text-violet-300">{stages[stage].name}</p>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{stages[stage].desc}</p>
        <p className="text-sm font-mono mt-2 text-violet-600 dark:text-violet-400">{stages[stage].output}</p>
      </div>
    </div>
  )
}

export default function TacotronVocoders() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Tacotron introduced end-to-end neural TTS by predicting mel spectrograms from text,
        replacing complex traditional pipelines. Combined with neural vocoders, it produces
        near-human quality speech synthesis.
      </p>

      <DefinitionBlock title="Tacotron 2 Architecture">
        <p>Tacotron 2 uses an encoder-decoder with location-sensitive attention:</p>
        <BlockMath math="\text{mel}_t = \text{Decoder}(s_{t-1}, \text{mel}_{t-1}, c_t)" />
        <p className="mt-2">
          where <InlineMath math="c_t = \text{Attention}(s_{t-1}, h, \alpha_{t-1})" /> uses
          previous alignment weights <InlineMath math="\alpha_{t-1}" /> to encourage monotonic
          progression. The decoder predicts 2 mel frames per step with a stop token.
        </p>
      </DefinitionBlock>

      <TTSPipelineExplorer />

      <ExampleBlock title="Tacotron 2 Loss">
        <p>Training minimizes the MSE on mel spectrograms plus a binary cross-entropy stop token loss:</p>
        <BlockMath math="\mathcal{L} = \frac{1}{T}\sum_{t=1}^{T} \|\hat{m}_t - m_t\|^2 + \lambda \text{BCE}(\hat{p}_t^{\text{stop}}, p_t^{\text{stop}})" />
        <p className="mt-1">
          The model also uses a post-net (5-layer CNN) that predicts a residual to refine the mel output.
        </p>
      </ExampleBlock>

      <DefinitionBlock title="Neural Vocoders">
        <p>Vocoders convert mel spectrograms to audio waveforms. Key architectures include:</p>
        <p className="mt-2"><strong>Griffin-Lim:</strong> Iterative phase reconstruction (fast but low quality)</p>
        <p><strong>WaveNet vocoder:</strong> Autoregressive, high quality, very slow</p>
        <p><strong>HiFi-GAN:</strong> GAN-based, real-time, near-WaveNet quality</p>
        <BlockMath math="\mathcal{L}_\text{HiFi-GAN} = \mathcal{L}_\text{adv} + \lambda_\text{fm}\mathcal{L}_\text{feature} + \lambda_\text{mel}\mathcal{L}_\text{mel}" />
      </DefinitionBlock>

      <PythonCode
        title="TTS with Coqui TTS (Tacotron 2 + HiFi-GAN)"
        code={`from TTS.api import TTS
import torch

# Coqui TTS: production-ready Tacotron 2 + vocoder pipeline
# List available models (Tacotron2, VITS, etc.)
print(TTS().list_models()[:5])

# Load pre-trained Tacotron 2 with HiFi-GAN vocoder
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC_ph")

# Synthesize speech from text
wav = tts.tts(text="Hello world, this is Tacotron 2 with neural vocoding.")
print(f"Generated audio: {len(wav)} samples at 22050 Hz")
print(f"Duration: {len(wav) / 22050:.2f} seconds")

# Save to file
tts.tts_to_file(
    text="Deep learning has revolutionized speech synthesis.",
    file_path="output.wav",
)

# Multi-speaker TTS (with speaker embeddings)
tts_multi = TTS(model_name="tts_models/en/vctk/vits")
wav = tts_multi.tts(
    text="Each speaker has a learned embedding vector.",
    speaker="p225",  # VCTK speaker ID
)
print(f"Multi-speaker output: {len(wav)} samples")`}
      />

      <WarningBlock title="Attention Alignment Issues">
        <p>
          Tacotron's attention mechanism can fail to learn proper alignment, causing repeated
          words, skipped phrases, or babbling. Training tricks include guided attention loss,
          pre-trained aligners (like Montreal Forced Aligner), and replacing attention with
          duration predictors (as in FastSpeech).
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="FastSpeech: Non-Autoregressive TTS">
        <p>
          FastSpeech replaces autoregressive decoding and attention with a duration predictor,
          enabling parallel mel generation. This is 100-300x faster than Tacotron 2 and avoids
          alignment failures. FastSpeech 2 adds pitch and energy predictors for better prosody.
        </p>
      </NoteBlock>
    </div>
  )
}

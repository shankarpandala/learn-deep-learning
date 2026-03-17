import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function WhisperModelExplorer() {
  const [model, setModel] = useState('base')

  const models = {
    tiny: { params: '39M', layers: '4+4', dim: 384, heads: 6, englishWER: '7.6%', multiWER: '14.2%' },
    base: { params: '74M', layers: '6+6', dim: 512, heads: 8, englishWER: '5.0%', multiWER: '10.5%' },
    small: { params: '244M', layers: '12+12', dim: 768, heads: 12, englishWER: '3.4%', multiWER: '7.6%' },
    medium: { params: '769M', layers: '24+24', dim: 1024, heads: 16, englishWER: '2.9%', multiWER: '5.8%' },
    large: { params: '1550M', layers: '32+32', dim: 1280, heads: 20, englishWER: '2.7%', multiWER: '4.2%' },
  }

  const m = models[model]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Whisper Model Variants</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.keys(models).map(key => (
          <button key={key} onClick={() => setModel(key)}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors capitalize ${model === key ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400'}`}>
            {key}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-3 gap-3 text-sm">
        {[['Parameters', m.params], ['Layers (enc+dec)', m.layers], ['Hidden dim', m.dim],
          ['Attention heads', m.heads], ['English WER', m.englishWER], ['Multilingual WER', m.multiWER]
        ].map(([label, val]) => (
          <div key={label} className="rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3">
            <p className="text-xs text-violet-600 dark:text-violet-400 font-semibold">{label}</p>
            <p className="text-gray-700 dark:text-gray-300 font-bold">{val}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function WhisperFoundationASR() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Whisper demonstrates that scaling weakly supervised training data to 680,000 hours
        produces remarkably robust speech recognition without self-supervised pre-training,
        approaching human-level performance across languages and acoustic conditions.
      </p>

      <DefinitionBlock title="Whisper Architecture">
        <p>
          Whisper uses a standard encoder-decoder Transformer. The encoder processes 30-second
          log-mel spectrogram chunks (<InlineMath math="80 \times 3000" /> frames), and the decoder
          generates text tokens autoregressively:
        </p>
        <BlockMath math="P(y_1, \ldots, y_N | X) = \prod_{i=1}^{N} P(y_i | y_{<i}, \text{Enc}(X))" />
        <p className="mt-2">
          Special tokens encode the task: <code>&lt;|language|&gt;</code>, <code>&lt;|transcribe|&gt;</code> or
          <code>&lt;|translate|&gt;</code>, and <code>&lt;|timestamps|&gt;</code>.
        </p>
      </DefinitionBlock>

      <WhisperModelExplorer />

      <TheoremBlock title="Robustness Through Diversity" id="whisper-robustness">
        <p>
          Whisper achieves robustness without domain-specific fine-tuning by training on diverse
          internet audio. On out-of-distribution benchmarks, Whisper's effective error rate
          decreases where fine-tuned models degrade:
        </p>
        <BlockMath math="\text{WER}_{\text{OOD}} \propto \frac{1}{\sqrt{|\mathcal{D}_{\text{train}}|}}" />
        <p className="mt-1">
          This scaling suggests that data diversity, not just quantity, drives generalization.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Multitask Training Format">
        <p>Whisper's decoder handles multiple tasks via prompt tokens:</p>
        <p className="font-mono text-sm mt-2 bg-gray-100 dark:bg-gray-800 p-2 rounded">
          &lt;|startoftranscript|&gt; &lt;|en|&gt; &lt;|transcribe|&gt; &lt;|notimestamps|&gt; Hello world &lt;|endoftext|&gt;
        </p>
        <p className="mt-2">
          For translation: replace <code>&lt;|transcribe|&gt;</code> with <code>&lt;|translate|&gt;</code> to
          translate any language to English.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Using Whisper for Speech Recognition"
        code={`import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load Whisper model
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

# Simulated 30s audio at 16kHz
audio = torch.randn(16000 * 30)

# Process audio to log-mel spectrogram
input_features = processor(
    audio.numpy(), sampling_rate=16000, return_tensors="pt"
).input_features
print(f"Input features: {input_features.shape}")  # [1, 80, 3000]

# Transcribe
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="en", task="transcribe"
)
generated_ids = model.generate(
    input_features, forced_decoder_ids=forced_decoder_ids
)
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(f"Transcription: {transcription[0]}")`}
      />

      <NoteBlock type="note" title="Beyond Whisper: Universal Speech Models">
        <p>
          Following Whisper, models like USM (Google) and MMS (Meta) scale to 1000+ languages.
          The trend is toward <strong>universal speech foundation models</strong> that handle ASR,
          translation, language ID, and speaker tasks in a single architecture, trained on
          millions of hours of diverse audio data.
        </p>
      </NoteBlock>
    </div>
  )
}

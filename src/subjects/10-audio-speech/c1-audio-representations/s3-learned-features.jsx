import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function FeatureComparison() {
  const [selected, setSelected] = useState('wav2vec2')

  const models = {
    wav2vec2: { name: 'wav2vec 2.0', input: 'Raw waveform', dim: 768, pretraining: 'Contrastive + masked prediction', data: '960h LibriSpeech' },
    hubert: { name: 'HuBERT', input: 'Raw waveform', dim: 768, pretraining: 'Offline clustering + masked prediction', data: '960h LibriSpeech' },
    whisper: { name: 'Whisper encoder', input: 'Log-mel spectrogram', dim: 1280, pretraining: 'Supervised multitask', data: '680k hours web audio' },
    beats: { name: 'BEATs', input: 'Log-mel spectrogram', dim: 768, pretraining: 'Audio event tokenizer + masked prediction', data: 'AudioSet' },
  }

  const m = models[selected]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Self-Supervised Audio Models</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.entries(models).map(([key, val]) => (
          <button key={key} onClick={() => setSelected(key)}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${selected === key ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3">
          <p className="text-xs text-violet-600 dark:text-violet-400 font-semibold">Input</p>
          <p className="text-gray-700 dark:text-gray-300">{m.input}</p>
        </div>
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3">
          <p className="text-xs text-violet-600 dark:text-violet-400 font-semibold">Hidden dim</p>
          <p className="text-gray-700 dark:text-gray-300">{m.dim}</p>
        </div>
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3">
          <p className="text-xs text-violet-600 dark:text-violet-400 font-semibold">Pre-training</p>
          <p className="text-gray-700 dark:text-gray-300">{m.pretraining}</p>
        </div>
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3">
          <p className="text-xs text-violet-600 dark:text-violet-400 font-semibold">Training data</p>
          <p className="text-gray-700 dark:text-gray-300">{m.data}</p>
        </div>
      </div>
    </div>
  )
}

export default function LearnedAudioFeatures() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Rather than hand-crafted spectral features, modern systems learn audio representations
        directly from raw waveforms or spectrograms using self-supervised pre-training,
        analogous to BERT and GPT in NLP.
      </p>

      <DefinitionBlock title="Convolutional Feature Encoder">
        <p>Models like wav2vec 2.0 process raw waveforms with a multi-layer 1D CNN:</p>
        <BlockMath math="z_t = \text{CNN}(x_{t \cdot s : t \cdot s + k})" />
        <p className="mt-2">
          The encoder uses <InlineMath math="7" /> temporal convolution blocks with strides
          that downsample 16 kHz audio to 50 Hz (one vector every 20 ms), producing latent
          representations <InlineMath math="z_t \in \mathbb{R}^{512}" />.
        </p>
      </DefinitionBlock>

      <FeatureComparison />

      <ExampleBlock title="wav2vec 2.0 Contrastive Loss">
        <p>During pre-training, masked positions are predicted via contrastive learning:</p>
        <BlockMath math="\mathcal{L} = -\log \frac{\exp(\text{sim}(c_t, q_t) / \kappa)}{\sum_{q' \in Q_t} \exp(\text{sim}(c_t, q') / \kappa)}" />
        <p className="mt-1">
          where <InlineMath math="c_t" /> is the Transformer context output, <InlineMath math="q_t" /> is
          the quantized target, and <InlineMath math="Q_t" /> includes distractors from other masked positions.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Extracting Learned Features with HuggingFace"
        code={`import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# Load pre-trained wav2vec 2.0
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# Process raw audio (16kHz)
waveform = torch.randn(1, 16000)  # 1 second of audio
inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# CNN features: [batch, time_steps, 512]
cnn_features = outputs.extract_features
print(f"CNN features: {cnn_features.shape}")

# Transformer features: [batch, time_steps, 768]
hidden_states = outputs.last_hidden_state
print(f"Hidden states: {hidden_states.shape}")

# Use as features for downstream tasks
# e.g., add a classification head for speaker ID
classifier = torch.nn.Linear(768, 100)  # 100 speakers
pooled = hidden_states.mean(dim=1)       # mean pooling
logits = classifier(pooled)
print(f"Speaker logits: {logits.shape}")`}
      />

      <WarningBlock title="Computational Cost">
        <p>
          Self-supervised audio models are computationally expensive. wav2vec 2.0 Base has 95M
          parameters, and processing a 10-second clip requires significant GPU memory. For
          resource-constrained settings, distilled models or log-mel features remain practical.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Feature Extraction vs Fine-tuning">
        <p>
          Pre-trained audio models can be used in two modes: <strong>frozen feature extraction</strong> (fast,
          good for small datasets) or <strong>full fine-tuning</strong> (better performance, needs more data).
          Intermediate approaches like fine-tuning only the top layers offer a balance.
        </p>
      </NoteBlock>
    </div>
  )
}

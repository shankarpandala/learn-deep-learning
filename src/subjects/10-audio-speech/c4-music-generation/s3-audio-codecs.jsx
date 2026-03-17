import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function CodecExplorer() {
  const [numCodebooks, setNumCodebooks] = useState(8)
  const [codebookSize, setCodebookSize] = useState(1024)
  const [frameRate, setFrameRate] = useState(75)

  const bitsPerSecond = numCodebooks * Math.log2(codebookSize) * frameRate
  const compressionRatio = (16000 * 16) / bitsPerSecond

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Neural Codec Calculator</h3>
      <div className="flex flex-wrap gap-4 mb-4">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Codebooks: {numCodebooks}
          <input type="range" min={1} max={16} step={1} value={numCodebooks} onChange={e => setNumCodebooks(Number(e.target.value))} className="w-24 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Size: {codebookSize}
          <input type="range" min={256} max={4096} step={256} value={codebookSize} onChange={e => setCodebookSize(Number(e.target.value))} className="w-24 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Frame rate: {frameRate} Hz
          <input type="range" min={25} max={150} step={25} value={frameRate} onChange={e => setFrameRate(Number(e.target.value))} className="w-24 accent-violet-500" />
        </label>
      </div>
      <div className="grid grid-cols-3 gap-3 text-sm">
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3">
          <p className="text-xs text-violet-600 dark:text-violet-400 font-semibold">Bitrate</p>
          <p className="text-xl font-bold text-violet-600">{(bitsPerSecond / 1000).toFixed(1)} kbps</p>
        </div>
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3">
          <p className="text-xs text-violet-600 dark:text-violet-400 font-semibold">Compression</p>
          <p className="text-xl font-bold text-violet-600">{compressionRatio.toFixed(1)}x</p>
        </div>
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3">
          <p className="text-xs text-violet-600 dark:text-violet-400 font-semibold">Tokens/second</p>
          <p className="text-xl font-bold text-violet-600">{numCodebooks * frameRate}</p>
        </div>
      </div>
    </div>
  )
}

export default function NeuralAudioCodecs() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Neural audio codecs like Encodec and SoundStream compress audio into discrete tokens
        using residual vector quantization, enabling language model-based audio generation
        and extremely low-bitrate compression with high perceptual quality.
      </p>

      <DefinitionBlock title="Residual Vector Quantization (RVQ)">
        <p>RVQ applies multiple rounds of vector quantization to the residual:</p>
        <BlockMath math="r_0 = z, \quad q_k = \text{VQ}_k(r_{k-1}), \quad r_k = r_{k-1} - q_k" />
        <p className="mt-2">
          After <InlineMath math="K" /> codebooks, the reconstructed vector
          is <InlineMath math="\hat{z} = \sum_{k=1}^{K} q_k" />. Each codebook captures progressively
          finer details, with the first codebook storing the coarsest approximation.
        </p>
      </DefinitionBlock>

      <CodecExplorer />

      <TheoremBlock title="Encodec Architecture" id="encodec-architecture">
        <p>
          Encodec uses an encoder-decoder CNN with RVQ in the bottleneck:
        </p>
        <BlockMath math="x \xrightarrow{\text{Enc}} z \xrightarrow{\text{RVQ}} \hat{z} \xrightarrow{\text{Dec}} \hat{x}" />
        <p className="mt-1">
          The encoder downsamples by <InlineMath math="320\times" /> (at 24 kHz input, producing 75 frames/sec).
          Training combines reconstruction loss, adversarial loss (multi-scale discriminator),
          and commitment loss for stable quantization:
        </p>
        <BlockMath math="\mathcal{L} = \lambda_r \mathcal{L}_\text{recon} + \lambda_a \mathcal{L}_\text{adv} + \lambda_c \sum_{k=1}^{K} \|z - \text{sg}[q_k]\|^2" />
      </TheoremBlock>

      <ExampleBlock title="Codec Tokens as Language">
        <p>
          With 8 codebooks of size 1024 at 75 Hz, one second of audio becomes a
          matrix of <InlineMath math="8 \times 75 = 600" /> discrete tokens.
          This enables treating audio as a sequence modeling problem:
        </p>
        <ul className="list-disc pl-5 mt-2 space-y-1">
          <li><strong>VALL-E:</strong> Autoregressive generation of codec tokens for TTS</li>
          <li><strong>MusicGen:</strong> Interleaved codebook prediction for music</li>
          <li><strong>AudioPaLM:</strong> Unified text and audio tokens in one LM</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Using Encodec for Audio Tokenization"
        code={`import torch
from transformers import EncodecModel, AutoProcessor

# Load Encodec
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# Encode audio to discrete tokens
audio = torch.randn(1, 1, 24000)  # 1 second at 24kHz
inputs = processor(raw_audio=audio.squeeze().numpy(), sampling_rate=24000, return_tensors="pt")

with torch.no_grad():
    encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])

# Discrete codes: [batch, num_codebooks, time_frames]
codes = encoder_outputs.audio_codes
print(f"Audio codes: {codes.shape}")  # [1, 1, 8, 75]
print(f"Code range: [{codes.min()}, {codes.max()}]")

# Decode back to audio
with torch.no_grad():
    decoded = model.decode(codes, encoder_outputs.audio_scales)
print(f"Reconstructed audio: {decoded.audio_values.shape}")`}
      />

      <NoteBlock type="note" title="Beyond Compression: Audio as Tokens">
        <p>
          Neural audio codecs have become the <strong>tokenizer for audio</strong>, playing the same
          role as BPE for text. This unification enables multimodal models that seamlessly handle
          text, speech, music, and sound effects within a single Transformer architecture,
          opening the door to truly universal audio-language models.
        </p>
      </NoteBlock>
    </div>
  )
}

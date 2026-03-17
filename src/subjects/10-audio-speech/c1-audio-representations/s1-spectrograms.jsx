import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function SpectrogramVisualizer() {
  const [windowSize, setWindowSize] = useState(1024)
  const [hopLength, setHopLength] = useState(256)
  const [useMel, setUseMel] = useState(false)

  const timeFrames = Math.floor(16000 / hopLength)
  const freqBins = useMel ? 80 : windowSize / 2

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Spectrogram Parameter Explorer</h3>
      <div className="flex flex-wrap items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Window: {windowSize}
          <input type="range" min={256} max={4096} step={256} value={windowSize} onChange={e => setWindowSize(Number(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Hop: {hopLength}
          <input type="range" min={64} max={1024} step={64} value={hopLength} onChange={e => setHopLength(Number(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          <input type="checkbox" checked={useMel} onChange={e => setUseMel(e.target.checked)} className="accent-violet-500" />
          Mel scale
        </label>
      </div>
      <div className="grid grid-cols-2 gap-4 text-sm text-gray-700 dark:text-gray-300">
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3">
          <p className="font-semibold text-violet-700 dark:text-violet-300">Time frames (1s audio)</p>
          <p className="text-2xl font-bold text-violet-600">{timeFrames}</p>
        </div>
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3">
          <p className="font-semibold text-violet-700 dark:text-violet-300">Frequency bins</p>
          <p className="text-2xl font-bold text-violet-600">{freqBins}</p>
        </div>
      </div>
      <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">
        Output shape: <strong>[{freqBins}, {timeFrames}]</strong> &mdash; freq resolution: {(16000 / windowSize).toFixed(1)} Hz, time resolution: {(hopLength / 16000 * 1000).toFixed(1)} ms
      </p>
    </div>
  )
}

export default function AudioRepresentations() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Raw audio waveforms are high-dimensional time-domain signals. Spectrograms convert them into
        compact time-frequency representations that are far more effective as neural network inputs,
        revealing structure invisible in the waveform.
      </p>

      <DefinitionBlock title="Short-Time Fourier Transform (STFT)">
        <p>The STFT decomposes a signal into overlapping windowed segments and applies the DFT to each:</p>
        <BlockMath math="X(t, f) = \sum_{n=0}^{N-1} x[n + tH] \cdot w[n] \cdot e^{-j2\pi fn/N}" />
        <p className="mt-2">
          where <InlineMath math="w[n]" /> is the window function of length <InlineMath math="N" />,
          and <InlineMath math="H" /> is the hop length. The <strong>spectrogram</strong> is <InlineMath math="|X(t,f)|^2" />.
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="Mel Scale">
        <p>The mel scale maps frequencies to approximate human pitch perception:</p>
        <BlockMath math="m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)" />
        <p className="mt-2">
          A mel spectrogram applies triangular filter banks spaced linearly on the mel scale,
          compressing high frequencies where human perception has lower resolution.
        </p>
      </DefinitionBlock>

      <SpectrogramVisualizer />

      <ExampleBlock title="Time-Frequency Trade-off">
        <p>
          With a window of <InlineMath math="N = 1024" /> at 16 kHz sample rate, the frequency
          resolution is <InlineMath math="\Delta f = 16000/1024 \approx 15.6" /> Hz, but the time
          resolution is <InlineMath math="1024/16000 = 64" /> ms. Smaller windows improve temporal
          precision at the cost of frequency resolution:
        </p>
        <BlockMath math="\Delta f \cdot \Delta t \geq \frac{1}{4\pi}" />
      </ExampleBlock>

      <PythonCode
        title="Computing Spectrograms with torchaudio"
        code={`import torch
import torchaudio
import torchaudio.transforms as T

# Load audio (16 kHz mono)
waveform, sr = torchaudio.load("speech.wav")
print(f"Waveform: {waveform.shape}")  # [1, num_samples]

# Standard spectrogram (STFT)
spectrogram = T.Spectrogram(n_fft=1024, hop_length=256)
spec = spectrogram(waveform)
print(f"Spectrogram: {spec.shape}")  # [1, 513, time_frames]

# Mel spectrogram (80 mel bins)
mel_spectrogram = T.MelSpectrogram(
    sample_rate=sr, n_fft=1024, hop_length=256, n_mels=80
)
mel_spec = mel_spectrogram(waveform)
print(f"Mel spectrogram: {mel_spec.shape}")  # [1, 80, time_frames]

# Log-mel spectrogram (standard input for speech models)
log_mel = torch.log(mel_spec.clamp(min=1e-9))
print(f"Log-mel: min={log_mel.min():.2f}, max={log_mel.max():.2f}")`}
      />

      <NoteBlock type="note" title="Why Log-Mel Spectrograms?">
        <p>
          Nearly all modern speech and audio models use <strong>log-mel spectrograms</strong> as input.
          The mel scale matches human perception, the log transform compresses dynamic range
          (mimicking the ear's logarithmic loudness response), and the 2D representation enables
          reuse of powerful vision architectures like CNNs and Vision Transformers.
        </p>
      </NoteBlock>
    </div>
  )
}

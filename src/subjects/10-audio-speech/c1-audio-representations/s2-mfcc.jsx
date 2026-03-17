import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function MFCCPipeline() {
  const [numCoeffs, setNumCoeffs] = useState(13)
  const [nMels, setNMels] = useState(40)
  const [step, setStep] = useState(4)

  const steps = [
    { name: 'Waveform', shape: '[T]', desc: 'Raw audio signal' },
    { name: 'STFT', shape: '[F, N]', desc: 'Short-Time Fourier Transform' },
    { name: 'Power Spectrum', shape: `[${nMels > 64 ? 512 : 256}, N]`, desc: '|STFT|^2' },
    { name: 'Mel Filter Bank', shape: `[${nMels}, N]`, desc: `${nMels} triangular filters` },
    { name: 'Log Mel', shape: `[${nMels}, N]`, desc: 'Log compression' },
    { name: 'DCT', shape: `[${numCoeffs}, N]`, desc: `Keep first ${numCoeffs} coefficients` },
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">MFCC Extraction Pipeline</h3>
      <div className="flex flex-wrap gap-4 mb-4">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Mel bins: {nMels}
          <input type="range" min={20} max={128} step={4} value={nMels} onChange={e => setNMels(Number(e.target.value))} className="w-24 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Coefficients: {numCoeffs}
          <input type="range" min={6} max={40} step={1} value={numCoeffs} onChange={e => setNumCoeffs(Number(e.target.value))} className="w-24 accent-violet-500" />
        </label>
      </div>
      <div className="flex flex-wrap gap-2">
        {steps.map((s, i) => (
          <button key={i} onClick={() => setStep(i)}
            className={`rounded-lg px-3 py-2 text-xs font-medium transition-colors ${i === step ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400'}`}>
            {i + 1}. {s.name}
          </button>
        ))}
      </div>
      <div className="mt-3 rounded-lg bg-violet-50 dark:bg-violet-900/20 p-4">
        <p className="font-semibold text-violet-700 dark:text-violet-300">{steps[step].name}</p>
        <p className="text-sm text-gray-600 dark:text-gray-400">{steps[step].desc}</p>
        <p className="text-sm font-mono mt-1 text-violet-600 dark:text-violet-400">Shape: {steps[step].shape}</p>
      </div>
    </div>
  )
}

export default function MFCCFilterBanks() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Mel-Frequency Cepstral Coefficients (MFCCs) were the dominant audio feature for decades.
        While deep learning models now often learn features end-to-end, understanding MFCCs
        provides insight into perceptual audio processing.
      </p>

      <DefinitionBlock title="MFCC Computation">
        <p>MFCCs apply the Discrete Cosine Transform to log-mel filter bank energies:</p>
        <BlockMath math="c_k = \sum_{m=1}^{M} \log(S_m) \cos\!\left[\frac{\pi k(m - 0.5)}{M}\right]" />
        <p className="mt-2">
          where <InlineMath math="S_m" /> is the energy in the <InlineMath math="m" />-th mel
          filter bank, and we keep only the first <InlineMath math="K" /> coefficients (typically 13).
        </p>
      </DefinitionBlock>

      <MFCCPipeline />

      <TheoremBlock title="Decorrelation via DCT" id="dct-decorrelation">
        <p>
          The DCT approximately decorrelates the log-mel features, acting as a compact
          representation of the spectral envelope. The first coefficient <InlineMath math="c_0" /> captures
          overall energy, while higher coefficients capture increasingly fine spectral detail.
          This decorrelation was critical for GMM-HMM systems with diagonal covariances.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Delta and Delta-Delta Features">
        <p>
          Standard practice appends first and second derivatives to capture dynamics:
        </p>
        <BlockMath math="\Delta c_k[t] = \frac{\sum_{n=1}^{N} n(c_k[t+n] - c_k[t-n])}{2\sum_{n=1}^{N} n^2}" />
        <p className="mt-1">
          This triples the feature dimension from 13 to 39, providing velocity and acceleration of spectral changes.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Extracting MFCCs with torchaudio"
        code={`import torch
import torchaudio
import torchaudio.transforms as T

waveform, sr = torchaudio.load("speech.wav")

# MFCC extraction
mfcc_transform = T.MFCC(
    sample_rate=sr,
    n_mfcc=13,
    melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 40}
)
mfccs = mfcc_transform(waveform)
print(f"MFCCs shape: {mfccs.shape}")  # [1, 13, time_frames]

# Compute deltas and delta-deltas
deltas = torchaudio.functional.compute_deltas(mfccs)
delta_deltas = torchaudio.functional.compute_deltas(deltas)

# Stack: [1, 39, time_frames]
features = torch.cat([mfccs, deltas, delta_deltas], dim=1)
print(f"Full features: {features.shape}")

# Compare with log-mel (modern preference)
mel_spec = T.MelSpectrogram(sample_rate=sr, n_mels=80, hop_length=256)(waveform)
log_mel = torch.log(mel_spec.clamp(min=1e-9))
print(f"Log-mel shape: {log_mel.shape}")  # [1, 80, time_frames]`}
      />

      <NoteBlock type="note" title="MFCCs vs Log-Mel in Deep Learning">
        <p>
          Modern deep learning systems typically prefer <strong>log-mel spectrograms</strong> over MFCCs.
          The DCT step in MFCCs discards information that neural networks can learn to use.
          However, MFCCs remain relevant in low-resource settings and as a pedagogical bridge
          to understanding perceptual audio features.
        </p>
      </NoteBlock>
    </div>
  )
}

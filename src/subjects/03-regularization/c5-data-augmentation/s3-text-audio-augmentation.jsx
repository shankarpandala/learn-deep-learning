import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function TextAugDemo() {
  const [method, setMethod] = useState('synonym')
  const original = 'The quick brown fox jumps over the lazy dog'

  const augmented = {
    synonym: 'The fast brown fox leaps over the idle dog',
    deletion: 'The quick fox jumps over lazy dog',
    swap: 'The quick brown jumps fox over the lazy dog',
    insertion: 'The very quick brown fox jumps swiftly over the lazy dog',
    backtranslation: 'The swift brown fox leaps over the indolent dog',
  }

  const words = original.split(' ')
  const augWords = augmented[method].split(' ')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Text Augmentation Demo</h3>
      <div className="flex flex-wrap gap-2 mb-3">
        {Object.keys(augmented).map(m => (
          <button key={m} onClick={() => setMethod(m)}
            className={`rounded px-3 py-1 text-xs ${method === m ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300'}`}>
            {m === 'backtranslation' ? 'Back-Translation' : m.charAt(0).toUpperCase() + m.slice(1)}
          </button>
        ))}
      </div>
      <div className="space-y-2 text-sm">
        <div className="rounded bg-gray-50 dark:bg-gray-800 p-3">
          <span className="text-xs text-gray-500 block mb-1">Original:</span>
          <span className="text-gray-700 dark:text-gray-300">{original}</span>
        </div>
        <div className="rounded bg-violet-50 dark:bg-violet-900/20 p-3">
          <span className="text-xs text-violet-500 block mb-1">Augmented ({method}):</span>
          <span className="text-gray-700 dark:text-gray-300">
            {augWords.map((w, i) => {
              const changed = !words.includes(w) || (words[i] !== w && method !== 'insertion')
              return <span key={i} className={changed ? 'font-bold text-violet-600 dark:text-violet-400' : ''}>{w} </span>
            })}
          </span>
        </div>
      </div>
    </div>
  )
}

function SpecAugmentViz() {
  const [maskType, setMaskType] = useState('both')
  const W = 320, H = 160, rows = 16, cols = 32

  const cellW = W / cols, cellH = H / rows
  const freqMask = { start: 5, end: 9 }
  const timeMask = { start: 12, end: 20 }

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">SpecAugment Visualization</h3>
      <div className="flex gap-2 mb-3">
        {['freq', 'time', 'both'].map(t => (
          <button key={t} onClick={() => setMaskType(t)}
            className={`rounded px-3 py-1 text-xs ${maskType === t ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300'}`}>
            {t === 'freq' ? 'Freq Mask' : t === 'time' ? 'Time Mask' : 'Both'}
          </button>
        ))}
      </div>
      <svg width={W} height={H} className="mx-auto block">
        {Array.from({ length: rows }, (_, r) => Array.from({ length: cols }, (_, c) => {
          const v = Math.sin(r * 0.5 + c * 0.2) * 0.3 + 0.5 + Math.sin(c * 0.4) * 0.2
          const masked = ((maskType !== 'time' && r >= freqMask.start && r <= freqMask.end) ||
            (maskType !== 'freq' && c >= timeMask.start && c <= timeMask.end))
          return <rect key={`${r}${c}`} x={c * cellW} y={r * cellH} width={cellW} height={cellH}
            fill={masked ? '#1f2937' : `hsl(263,${Math.round(v * 70 + 20)}%,${Math.round(v * 40 + 30)}%)`} />
        }))}
      </svg>
    </div>
  )
}

export default function TextAudioAugmentation() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Text and audio domains require specialized augmentation approaches that respect
        the discrete nature of language and the spectral structure of audio.
      </p>

      <DefinitionBlock title="Text Augmentation Techniques">
        <ul className="list-disc ml-4 space-y-1 mt-2">
          <li><strong>Synonym replacement</strong>: replace words with synonyms from WordNet</li>
          <li><strong>Random deletion</strong>: remove words with probability <InlineMath math="p" /></li>
          <li><strong>Random swap</strong>: swap positions of two random words</li>
          <li><strong>Back-translation</strong>: translate to another language and back</li>
        </ul>
      </DefinitionBlock>

      <TextAugDemo />

      <DefinitionBlock title="SpecAugment">
        <p>
          SpecAugment applies augmentation directly to log-mel spectrograms with two operations:
        </p>
        <BlockMath math="\text{FreqMask}: X[f_0 : f_0 + f, :] = 0, \quad \text{TimeMask}: X[:, t_0 : t_0 + t] = 0" />
        <p className="mt-2">
          where <InlineMath math="f" /> and <InlineMath math="t" /> are randomly chosen mask widths.
          This is simple, effective, and requires no external data.
        </p>
      </DefinitionBlock>

      <SpecAugmentViz />

      <ExampleBlock title="SpecAugment Results">
        <p>
          SpecAugment reduced word error rate on LibriSpeech from 3.9% to 2.8% without
          additional data. Frequency masking provides speaker robustness; time masking handles temporal distortions.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Text & Audio Augmentation in PyTorch"
        code={`import torch
import random

# --- Text Augmentation (EDA: Easy Data Augmentation) ---
def synonym_replace(words, n=1):
    syns = {'quick': 'fast', 'jumps': 'leaps', 'lazy': 'idle'}
    new = words.copy()
    for _ in range(n):
        i = random.randint(0, len(new) - 1)
        if new[i] in syns: new[i] = syns[new[i]]
    return new

def random_deletion(words, p=0.1):
    return [w for w in words if random.random() > p] or [words[0]]

text = "The quick brown fox jumps over the lazy dog"
print("Synonym:", ' '.join(synonym_replace(text.split())))
print("Deletion:", ' '.join(random_deletion(text.split(), p=0.2)))

# --- SpecAugment for Audio ---
def spec_augment(spec, freq_mask=15, time_mask=20):
    cloned = spec.clone()
    F, T = cloned.shape
    f = random.randint(0, min(freq_mask, F))
    f0 = random.randint(0, F - f)
    cloned[f0:f0+f, :] = 0
    t = random.randint(0, min(time_mask, T))
    t0 = random.randint(0, T - t)
    cloned[:, t0:t0+t] = 0
    return cloned

spec = torch.randn(80, 200)  # 80 mel bins, 200 time steps
augmented = spec_augment(spec)
print(f"Zeroed: {(augmented == 0).sum().item()} / {spec.numel()}")`}
      />

      <WarningBlock title="Text Augmentation Caveats">
        <p>
          Text augmentation can change semantics easily. Synonym replacement may alter meaning
          in context. Back-translation quality depends on the translation model. For large
          language models, augmentation is less critical since pretraining provides regularization.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Modality-Specific Considerations">
        <p>
          Each modality has unique invariances. The best augmentation strategy encodes the
          invariances specific to your task: flips for images, paraphrase for text, speed
          changes and noise for audio.
        </p>
      </NoteBlock>
    </div>
  )
}

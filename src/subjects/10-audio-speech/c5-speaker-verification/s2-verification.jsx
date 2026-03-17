import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function VerificationThresholdDemo() {
  const [threshold, setThreshold] = useState(0.5)

  const pairs = [
    { label: 'Same speaker', score: 0.85, actual: true },
    { label: 'Same speaker', score: 0.72, actual: true },
    { label: 'Different', score: 0.31, actual: false },
    { label: 'Same speaker', score: 0.48, actual: true },
    { label: 'Different', score: 0.22, actual: false },
    { label: 'Different', score: 0.55, actual: false },
  ]

  const tp = pairs.filter(p => p.actual && p.score >= threshold).length
  const fp = pairs.filter(p => !p.actual && p.score >= threshold).length
  const fn = pairs.filter(p => p.actual && p.score < threshold).length
  const tn = pairs.filter(p => !p.actual && p.score < threshold).length

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Verification Threshold Explorer</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Threshold: {threshold.toFixed(2)}
        <input type="range" min={0.1} max={0.9} step={0.05} value={threshold} onChange={e => setThreshold(Number(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <div className="space-y-1 mb-3">
        {pairs.map((p, i) => {
          const accepted = p.score >= threshold
          const correct = accepted === p.actual
          return (
            <div key={i} className="flex items-center gap-2 text-sm">
              <div className="w-32 text-gray-600 dark:text-gray-400">{p.label}</div>
              <div className="flex-1 h-4 bg-gray-100 dark:bg-gray-800 rounded relative">
                <div className="h-full rounded bg-violet-400" style={{ width: `${p.score * 100}%` }} />
                <div className="absolute top-0 h-full border-l-2 border-red-500" style={{ left: `${threshold * 100}%` }} />
              </div>
              <span className={`text-xs font-bold ${correct ? 'text-green-600' : 'text-red-500'}`}>
                {correct ? 'Correct' : 'Error'}
              </span>
            </div>
          )
        })}
      </div>
      <div className="grid grid-cols-4 gap-2 text-sm text-center">
        <div className="bg-green-100 dark:bg-green-900/30 rounded p-2"><span className="font-bold text-green-700">TP: {tp}</span></div>
        <div className="bg-red-100 dark:bg-red-900/30 rounded p-2"><span className="font-bold text-red-600">FP: {fp}</span></div>
        <div className="bg-red-100 dark:bg-red-900/30 rounded p-2"><span className="font-bold text-red-600">FN: {fn}</span></div>
        <div className="bg-green-100 dark:bg-green-900/30 rounded p-2"><span className="font-bold text-green-700">TN: {tn}</span></div>
      </div>
    </div>
  )
}

export default function SpeakerVerificationID() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Speaker verification determines whether two utterances belong to the same speaker,
        while speaker identification classifies an utterance among known speakers. Diarization
        extends this to segment multi-speaker audio into who spoke when.
      </p>

      <DefinitionBlock title="Speaker Verification">
        <p>Given enrollment embedding <InlineMath math="e_\text{enr}" /> and test embedding <InlineMath math="e_\text{test}" />:</p>
        <BlockMath math="\text{score} = \cos(e_\text{enr}, e_\text{test}) = \frac{e_\text{enr}^\top e_\text{test}}{\|e_\text{enr}\| \|e_\text{test}\|}" />
        <p className="mt-2">
          Accept if <InlineMath math="\text{score} \geq \tau" />, reject otherwise.
          The threshold <InlineMath math="\tau" /> controls the trade-off between false acceptance rate (FAR)
          and false rejection rate (FRR).
        </p>
      </DefinitionBlock>

      <VerificationThresholdDemo />

      <TheoremBlock title="Equal Error Rate (EER)" id="eer-metric">
        <p>
          The EER is the operating point where FAR equals FRR:
        </p>
        <BlockMath math="\text{EER} = \text{FAR}(\tau^*) = \text{FRR}(\tau^*)" />
        <p className="mt-1">
          State-of-the-art systems achieve EER below 1% on VoxCeleb1. The minDCF metric
          weighs false acceptances and rejections differently for real-world applications.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Speaker Diarization Pipeline">
        <p>Modern neural diarization systems follow these steps:</p>
        <ol className="list-decimal pl-5 mt-2 space-y-1">
          <li><strong>VAD:</strong> Voice Activity Detection to find speech segments</li>
          <li><strong>Segmentation:</strong> Split into overlapping windows (1.5-3s)</li>
          <li><strong>Embedding:</strong> Extract speaker embeddings per segment</li>
          <li><strong>Clustering:</strong> Agglomerative or spectral clustering of embeddings</li>
          <li><strong>Resegmentation:</strong> Refine boundaries with a neural model</li>
        </ol>
      </ExampleBlock>

      <PythonCode
        title="Speaker Verification System"
        code={`import torch
import torch.nn.functional as F

def verify_speaker(model, audio_enr, audio_test, threshold=0.5):
    """Verify if two audio clips are from the same speaker."""
    with torch.no_grad():
        emb_enr = model(audio_enr)    # [1, 192]
        emb_test = model(audio_test)   # [1, 192]

    # Cosine similarity
    score = F.cosine_similarity(emb_enr, emb_test).item()
    return score >= threshold, score

# Simulate verification with random embeddings
emb_same = F.normalize(torch.randn(1, 192), dim=-1)
emb_similar = F.normalize(emb_same + 0.1 * torch.randn(1, 192), dim=-1)
emb_different = F.normalize(torch.randn(1, 192), dim=-1)

score_same = F.cosine_similarity(emb_same, emb_similar).item()
score_diff = F.cosine_similarity(emb_same, emb_different).item()

print(f"Same speaker similarity: {score_same:.4f}")
print(f"Different speaker similarity: {score_diff:.4f}")

# Compute EER (simplified)
def compute_eer(scores_pos, scores_neg, n_thresholds=100):
    thresholds = torch.linspace(0, 1, n_thresholds)
    for tau in thresholds:
        far = (scores_neg >= tau).float().mean()
        frr = (scores_pos < tau).float().mean()
        if far <= frr:
            return tau.item(), ((far + frr) / 2).item()
    return 0.5, 0.5

scores_pos = torch.tensor([0.85, 0.72, 0.91, 0.68, 0.79])
scores_neg = torch.tensor([0.31, 0.22, 0.15, 0.42, 0.28])
threshold, eer = compute_eer(scores_pos, scores_neg)
print(f"EER: {eer:.4f} at threshold {threshold:.4f}")`}
      />

      <NoteBlock type="note" title="End-to-End Neural Diarization">
        <p>
          EEND (End-to-End Neural Diarization) replaces the traditional pipeline with a single
          neural network that directly predicts speaker activity for each frame. It handles
          overlapping speech naturally and can be combined with encoder-based approaches
          (EEND-VC) for handling variable numbers of speakers.
        </p>
      </NoteBlock>
    </div>
  )
}

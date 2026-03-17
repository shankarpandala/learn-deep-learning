import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function EmbeddingSpaceDemo() {
  const [metric, setMetric] = useState('cosine')
  const speakers = [
    { name: 'Speaker A', x: 0.8, y: 0.6, color: '#8b5cf6' },
    { name: 'Speaker B', x: 0.3, y: 0.8, color: '#f97316' },
    { name: 'Speaker C', x: 0.7, y: 0.2, color: '#06b6d4' },
  ]

  const W = 300, H = 250

  const cosine = (a, b) => {
    const dot = a.x * b.x + a.y * b.y
    const na = Math.sqrt(a.x * a.x + a.y * a.y)
    const nb = Math.sqrt(b.x * b.x + b.y * b.y)
    return dot / (na * nb)
  }
  const euclidean = (a, b) => Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

  const distFn = metric === 'cosine' ? cosine : euclidean

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Speaker Embedding Space</h3>
      <div className="flex gap-4 mb-3">
        {['cosine', 'euclidean'].map(m => (
          <button key={m} onClick={() => setMetric(m)}
            className={`rounded-lg px-3 py-1 text-sm font-medium capitalize ${metric === m ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'}`}>
            {m}
          </button>
        ))}
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <rect width={W} height={H} fill="none" stroke="#e5e7eb" rx={8} />
        {speakers.map((s, i) => (
          <g key={i}>
            <circle cx={s.x * (W - 40) + 20} cy={(1 - s.y) * (H - 40) + 20} r={8} fill={s.color} opacity={0.8} />
            <text x={s.x * (W - 40) + 20} y={(1 - s.y) * (H - 40) + 25} fontSize={9} fill={s.color} textAnchor="middle">{s.name}</text>
          </g>
        ))}
        {speakers.map((a, i) => speakers.slice(i + 1).map((b, j) => (
          <line key={`${i}-${j}`} x1={a.x * (W - 40) + 20} y1={(1 - a.y) * (H - 40) + 20}
            x2={b.x * (W - 40) + 20} y2={(1 - b.y) * (H - 40) + 20}
            stroke="#d1d5db" strokeWidth={0.8} strokeDasharray="3,3" />
        )))}
      </svg>
      <div className="mt-2 grid grid-cols-3 gap-2 text-xs text-center">
        <p className="text-gray-600 dark:text-gray-400">A-B: {distFn(speakers[0], speakers[1]).toFixed(3)}</p>
        <p className="text-gray-600 dark:text-gray-400">A-C: {distFn(speakers[0], speakers[2]).toFixed(3)}</p>
        <p className="text-gray-600 dark:text-gray-400">B-C: {distFn(speakers[1], speakers[2]).toFixed(3)}</p>
      </div>
    </div>
  )
}

export default function SpeakerEmbeddings() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Speaker embeddings map variable-length utterances to fixed-dimensional vectors that
        capture speaker identity, enabling verification, identification, and diarization
        through simple distance comparisons in embedding space.
      </p>

      <DefinitionBlock title="x-vector Architecture">
        <p>The x-vector system uses a TDNN (Time-Delay Neural Network) with statistical pooling:</p>
        <BlockMath math="e = W_2 \cdot \text{ReLU}(W_1 \cdot [\mu(h), \sigma(h)])" />
        <p className="mt-2">
          Frame-level features <InlineMath math="h_t" /> are extracted by TDNN layers, then aggregated
          via mean <InlineMath math="\mu" /> and standard deviation <InlineMath math="\sigma" /> pooling
          to produce a fixed-size utterance-level embedding.
        </p>
      </DefinitionBlock>

      <EmbeddingSpaceDemo />

      <TheoremBlock title="ECAPA-TDNN" id="ecapa-tdnn">
        <p>
          ECAPA-TDNN improves x-vectors with three key innovations:
        </p>
        <BlockMath math="h_l = \text{SE}(\text{Res2Net}(\text{TDNN}_l(h_{l-1})))" />
        <p className="mt-1">
          (1) <strong>Res2Net</strong> blocks capture multi-scale features,
          (2) <strong>Squeeze-Excitation</strong> (SE) performs channel attention,
          and (3) <strong>Attentive statistical pooling</strong> learns frame importance weights:
        </p>
        <BlockMath math="e = \sum_t \alpha_t h_t, \quad \alpha_t = \text{softmax}(w^\top \tanh(Vh_t + b))" />
      </TheoremBlock>

      <ExampleBlock title="Training with AAM-Softmax">
        <p>Speaker embeddings are trained with Additive Angular Margin Softmax:</p>
        <BlockMath math="\mathcal{L} = -\log \frac{e^{s \cos(\theta_{y} + m)}}{e^{s \cos(\theta_{y} + m)} + \sum_{j \neq y} e^{s \cos \theta_j}}" />
        <p className="mt-1">
          where <InlineMath math="s" /> is a scale factor, <InlineMath math="m" /> is the angular margin,
          and <InlineMath math="\theta_y" /> is the angle between the embedding and the class center.
          This forces inter-class separation in angular space.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Speaker Embedding Extraction"
        code={`import torch
import torch.nn as nn

class SimpleSpeakerEncoder(nn.Module):
    def __init__(self, input_dim=80, embed_dim=192):
        super().__init__()
        self.tdnn = nn.Sequential(
            nn.Conv1d(input_dim, 512, 5, padding=2), nn.ReLU(),
            nn.Conv1d(512, 512, 3, dilation=2, padding=2), nn.ReLU(),
            nn.Conv1d(512, 512, 3, dilation=3, padding=3), nn.ReLU(),
        )
        # Attentive statistical pooling
        self.attention = nn.Sequential(
            nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.embed = nn.Linear(1024, embed_dim)  # mean + std

    def forward(self, mel):  # mel: [B, 80, T]
        h = self.tdnn(mel)  # [B, 512, T]
        # Attention weights
        alpha = self.attention(h.transpose(1, 2)).squeeze(-1)  # [B, T]
        alpha = torch.softmax(alpha, dim=-1).unsqueeze(1)  # [B, 1, T]
        # Weighted statistics
        mean = (alpha * h).sum(dim=-1)
        var = (alpha * h**2).sum(dim=-1) - mean**2
        std = torch.sqrt(var.clamp(min=1e-9))
        stats = torch.cat([mean, std], dim=-1)
        return nn.functional.normalize(self.embed(stats), dim=-1)

model = SimpleSpeakerEncoder()
mel = torch.randn(4, 80, 200)
embeddings = model(mel)
print(f"Speaker embeddings: {embeddings.shape}")  # [4, 192]
# Cosine similarity between speakers
sim = torch.mm(embeddings, embeddings.T)
print(f"Similarity matrix:\\n{sim}")`}
      />

      <NoteBlock type="note" title="Self-Supervised Speaker Representations">
        <p>
          Pre-trained models like wav2vec 2.0 and WavLM produce excellent speaker embeddings
          when fine-tuned, often outperforming purpose-built speaker encoders. The SUPERB
          benchmark shows that general audio representations transfer well to speaker tasks,
          suggesting shared underlying structure in speech representations.
        </p>
      </NoteBlock>
    </div>
  )
}

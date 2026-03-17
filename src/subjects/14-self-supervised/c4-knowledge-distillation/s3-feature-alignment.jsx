import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function TransferBenchmark() {
  const [task, setTask] = useState('classification')
  const tasks = {
    classification: { name: 'ImageNet Classification', supervised: 76.5, simclr: 71.7, dino: 77.3, dinov2: 83.5, metric: 'Top-1 Acc (%)' },
    segmentation: { name: 'ADE20K Segmentation', supervised: 45.1, simclr: 39.2, dino: 44.6, dinov2: 49.0, metric: 'mIoU' },
    detection: { name: 'COCO Detection', supervised: 38.2, simclr: 35.8, dino: 39.1, dinov2: 42.5, metric: 'AP50' },
    retrieval: { name: 'Image Retrieval', supervised: 70.1, simclr: 65.4, dino: 73.8, dinov2: 78.2, metric: 'mAP' },
  }
  const t = tasks[task]
  const maxVal = Math.max(t.supervised, t.simclr, t.dino, t.dinov2)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-2 text-base font-bold text-gray-800 dark:text-gray-200">Transfer Learning Benchmark</h3>
      <div className="flex gap-2 mb-3 flex-wrap">
        {Object.entries(tasks).map(([key, v]) => (
          <button key={key} onClick={() => setTask(key)}
            className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${task === key ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400'}`}>
            {v.name.split(' ').slice(-1)[0]}
          </button>
        ))}
      </div>
      <div className="flex gap-3 justify-center items-end h-28">
        {[
          { label: 'Supervised', val: t.supervised, color: 'bg-gray-400' },
          { label: 'SimCLR', val: t.simclr, color: 'bg-violet-300' },
          { label: 'DINO', val: t.dino, color: 'bg-violet-400' },
          { label: 'DINOv2', val: t.dinov2, color: 'bg-violet-600' },
        ].map(({ label, val, color }) => (
          <div key={label} className="flex flex-col items-center">
            <div className={`w-12 ${color} rounded-t transition-all`} style={{ height: `${(val / maxVal) * 85}px` }} />
            <span className="text-[9px] text-gray-500 mt-1">{label}</span>
            <span className="text-[9px] font-semibold">{val}</span>
          </div>
        ))}
      </div>
      <p className="text-xs text-gray-500 text-center mt-1">{t.name} — {t.metric} (linear probe, ViT-L)</p>
    </div>
  )
}

export default function FeatureAlignment() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The ultimate test of self-supervised representations is their transfer performance.
        Understanding how to evaluate, align, and effectively use pre-trained features for
        downstream tasks is crucial for practical applications.
      </p>

      <DefinitionBlock title="Evaluation Protocols">
        <p>Standard protocols for evaluating self-supervised features:</p>
        <ul className="list-disc ml-5 mt-2 space-y-1">
          <li><strong>Linear probe</strong>: Train a single linear layer on frozen features</li>
          <li><strong>k-NN evaluation</strong>: Nearest-neighbor classification with no training</li>
          <li><strong>Fine-tuning</strong>: Update all parameters on downstream task</li>
          <li><strong>Few-shot</strong>: Linear probe with limited labeled data (1%, 10%)</li>
        </ul>
        <BlockMath math="\text{Linear probe: } \min_W \mathcal{L}(W f_\theta(\mathbf{x}), y), \quad \theta \text{ frozen}" />
      </DefinitionBlock>

      <TransferBenchmark />

      <TheoremBlock title="Feature Alignment and Uniformity" id="alignment-uniformity">
        <p>Good representation quality requires two properties on the unit hypersphere:</p>
        <BlockMath math="\mathcal{L}_{\text{align}} = \mathbb{E}_{(x,x^+)}\left[\|f(x) - f(x^+)\|^2\right]" />
        <BlockMath math="\mathcal{L}_{\text{uniform}} = \log \mathbb{E}_{(x,y)}\left[e^{-2\|f(x) - f(y)\|^2}\right]" />
        <p className="mt-2">
          <strong>Alignment</strong>: positive pairs should be close. <strong>Uniformity</strong>: features should
          be uniformly distributed (high entropy). These two metrics predict downstream performance.
        </p>
      </TheoremBlock>

      <ExampleBlock title="When to Fine-tune vs Freeze">
        <p>
          <strong>Freeze features</strong> when: target domain is similar to pre-training data,
          labeled data is very limited, or you need fast iteration. <strong>Fine-tune</strong> when:
          target domain differs significantly (medical, satellite imagery), sufficient labeled data
          exists, or maximum performance is needed. A middle ground: fine-tune only the last few layers.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Feature Evaluation: Linear Probe and k-NN"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearProbe(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, features):
        return self.linear(features.detach())  # detach = frozen backbone

def knn_evaluate(train_features, train_labels, test_features, test_labels, k=20):
    """k-NN classification with no training."""
    train_features = F.normalize(train_features, dim=-1)
    test_features = F.normalize(test_features, dim=-1)

    # Cosine similarity
    sim = test_features @ train_features.T  # (N_test, N_train)
    topk_sim, topk_idx = sim.topk(k, dim=-1)

    # Weighted voting
    topk_labels = train_labels[topk_idx]  # (N_test, k)
    predictions = topk_labels.mode(dim=-1).values

    accuracy = (predictions == test_labels).float().mean()
    return accuracy.item()

# Alignment and uniformity metrics
def alignment(x, y):
    return (x - y).norm(dim=-1).pow(2).mean()

def uniformity(x, t=2):
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()

features = F.normalize(torch.randn(256, 768), dim=-1)
pos_pairs = features + 0.1 * torch.randn_like(features)
pos_pairs = F.normalize(pos_pairs, dim=-1)
print(f"Alignment: {alignment(features, pos_pairs):.4f} (lower = better)")
print(f"Uniformity: {uniformity(features):.4f} (lower = better)")`}
      />

      <NoteBlock type="note" title="Self-Supervised Features in Practice">
        <p>
          Self-supervised pre-training has become the default initialization for many vision
          applications. DINOv2 features serve as drop-in replacements for ImageNet-supervised
          features across classification, detection, segmentation, and retrieval. The key practical
          insight: invest in the best available pre-trained backbone and adapt minimally to your task.
        </p>
      </NoteBlock>
    </div>
  )
}

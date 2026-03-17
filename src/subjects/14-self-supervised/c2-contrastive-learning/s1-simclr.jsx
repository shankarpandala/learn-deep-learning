import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function TemperatureViz() {
  const [tau, setTau] = useState(0.5)

  const sims = [-0.5, 0.0, 0.3, 0.7, 0.9, 1.0]
  const softmaxVals = sims.map(s => Math.exp(s / tau))
  const sumExp = softmaxVals.reduce((a, b) => a + b, 0)
  const probs = softmaxVals.map(v => v / sumExp)
  const maxProb = Math.max(...probs)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Temperature Effect on NT-Xent</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        tau = {tau.toFixed(2)}
        <input type="range" min={0.05} max={2} step={0.05} value={tau} onChange={e => setTau(parseFloat(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <div className="flex gap-2 items-end justify-center h-28">
        {sims.map((s, i) => (
          <div key={i} className="flex flex-col items-center">
            <div className={`w-10 rounded-t transition-all ${i === sims.length - 1 ? 'bg-violet-500' : 'bg-violet-300'}`}
              style={{ height: `${(probs[i] / maxProb) * 90}px` }} />
            <span className="text-[9px] text-gray-500 mt-1">sim={s}</span>
            <span className="text-[9px] text-violet-600">{probs[i].toFixed(3)}</span>
          </div>
        ))}
      </div>
      <p className="text-xs text-gray-500 text-center mt-1">
        {tau < 0.2 ? 'Very low tau: only the highest similarity matters (hard)' :
         tau < 0.7 ? 'Moderate tau: good discrimination between similarities' :
         'High tau: similarities become nearly uniform (too soft)'}
      </p>
    </div>
  )
}

export default function SimCLR() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        SimCLR (Simple Contrastive Learning of Representations) is a foundational contrastive learning
        framework. It learns representations by maximizing agreement between differently augmented
        views of the same image while pushing apart representations of different images.
      </p>

      <DefinitionBlock title="NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)">
        <p>For a positive pair <InlineMath math="(i, j)" /> within a batch of <InlineMath math="2N" /> augmented samples:</p>
        <BlockMath math="\ell_{i,j} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{k \neq i}\, \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau)}" />
        <p className="mt-2">
          where <InlineMath math="\text{sim}(\mathbf{u}, \mathbf{v}) = \mathbf{u}^\top \mathbf{v} / (\|\mathbf{u}\|\|\mathbf{v}\|)" /> is cosine similarity and <InlineMath math="\tau" /> is the temperature.
        </p>
      </DefinitionBlock>

      <TemperatureViz />

      <TheoremBlock title="SimCLR Framework" id="simclr-framework">
        <p>The four components of SimCLR:</p>
        <ol className="list-decimal ml-5 mt-2 space-y-1">
          <li><strong>Augmentations</strong> <InlineMath math="T" />: Random crop + color jitter + Gaussian blur</li>
          <li><strong>Encoder</strong> <InlineMath math="f" />: ResNet backbone producing <InlineMath math="\mathbf{h} = f(\tilde{\mathbf{x}})" /></li>
          <li><strong>Projection head</strong> <InlineMath math="g" />: MLP mapping <InlineMath math="\mathbf{z} = g(\mathbf{h})" /> (discarded after training)</li>
          <li><strong>NT-Xent loss</strong>: Contrastive objective on the projected representations</li>
        </ol>
      </TheoremBlock>

      <PythonCode
        title="SimCLR Loss Implementation"
        code={`import torch
import torch.nn.functional as F

def simclr_loss(z1, z2, temperature=0.5):
    """NT-Xent loss for SimCLR.
    z1, z2: (B, D) embeddings of two augmented views.
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    z = F.normalize(z, dim=-1)

    # Cosine similarity matrix (2B x 2B)
    sim = z @ z.T / temperature

    # Mask out self-similarity
    mask = ~torch.eye(2 * B, dtype=bool, device=z.device)
    sim = sim.masked_fill(~mask, -1e9)

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(B, 2 * B),
        torch.arange(0, B)
    ], dim=0).to(z.device)

    loss = F.cross_entropy(sim, labels)
    return loss

# Example usage
z1 = F.normalize(torch.randn(256, 128), dim=-1)
z2 = F.normalize(torch.randn(256, 128), dim=-1)
loss = simclr_loss(z1, z2, temperature=0.5)
print(f"SimCLR loss (B=256): {loss.item():.3f}")
print(f"Random baseline: {torch.log(torch.tensor(2*256-1.0)):.3f}")`}
      />

      <ExampleBlock title="Why Large Batches Matter">
        <p>
          SimCLR uses negatives from within the batch. Larger batches provide more negatives,
          creating a harder contrastive task. SimCLR v1 used batch size 4096 (8192 augmented views);
          performance degrades significantly below 256. This is a key limitation addressed by MoCo.
        </p>
      </ExampleBlock>

      <NoteBlock type="note" title="The Projection Head Is Critical">
        <p>
          Representations before the projection head (<InlineMath math="\mathbf{h}" />) transfer better than
          those after it (<InlineMath math="\mathbf{z}" />). The projection head discards information useful for
          downstream tasks but irrelevant to the contrastive objective (e.g., color information lost
          to augmentation). Always evaluate on <InlineMath math="\mathbf{h}" />, not <InlineMath math="\mathbf{z}" />.
        </p>
      </NoteBlock>
    </div>
  )
}

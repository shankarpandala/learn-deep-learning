import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function MomentumViz() {
  const [momentum, setMomentum] = useState(0.999)
  const [queueSize, setQueueSize] = useState(65536)

  const halfLife = Math.log(0.5) / Math.log(momentum)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">MoCo: Momentum & Queue</h3>
      <div className="flex gap-4 mb-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          momentum: {momentum.toFixed(3)}
          <input type="range" min={0.9} max={0.9999} step={0.001} value={momentum} onChange={e => setMomentum(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          queue: {queueSize.toLocaleString()}
          <input type="range" min={4096} max={131072} step={4096} value={queueSize} onChange={e => setQueueSize(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <div className="flex gap-8 justify-center items-center">
        <div className="text-center">
          <div className="w-16 h-16 rounded-lg bg-violet-500 flex items-center justify-center text-white text-xs font-bold">Query<br/>Encoder</div>
          <p className="text-[9px] text-gray-500 mt-1">gradient update</p>
        </div>
        <div className="text-violet-400 text-lg">&rarr;</div>
        <div className="text-center">
          <div className="w-16 h-16 rounded-lg bg-violet-300 flex items-center justify-center text-white text-xs font-bold">Key<br/>Encoder</div>
          <p className="text-[9px] text-gray-500 mt-1">m={momentum.toFixed(3)}</p>
        </div>
        <div className="text-violet-400 text-lg">&rarr;</div>
        <div className="text-center">
          <div className="w-20 h-16 rounded-lg bg-orange-300 flex items-center justify-center text-white text-xs font-bold">Queue<br/>{(queueSize/1024).toFixed(0)}K keys</div>
          <p className="text-[9px] text-gray-500 mt-1">FIFO negatives</p>
        </div>
      </div>
      <p className="text-xs text-gray-500 text-center mt-2">
        Half-life: ~{halfLife.toFixed(0)} steps | Effective negatives: {queueSize.toLocaleString()} (vs batch size for SimCLR)
      </p>
    </div>
  )
}

export default function MoCo() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Momentum Contrast (MoCo) decouples the number of negatives from batch size by maintaining
        a queue of encoded keys. A momentum-updated encoder ensures the keys are consistent, enabling
        contrastive learning with standard batch sizes.
      </p>

      <DefinitionBlock title="MoCo Framework">
        <p>MoCo maintains a query encoder <InlineMath math="f_q" /> and a momentum key encoder <InlineMath math="f_k" />:</p>
        <BlockMath math="\theta_k \leftarrow m \cdot \theta_k + (1 - m) \cdot \theta_q" />
        <p className="mt-2">The InfoNCE loss with queue negatives:</p>
        <BlockMath math="\mathcal{L}_q = -\log \frac{\exp(\mathbf{q} \cdot \mathbf{k}_+ / \tau)}{\exp(\mathbf{q} \cdot \mathbf{k}_+ / \tau) + \sum_{i=0}^{K} \exp(\mathbf{q} \cdot \mathbf{k}_i / \tau)}" />
        <p className="mt-1">where <InlineMath math="\mathbf{k}_+" /> is the positive key and <InlineMath math="\mathbf{k}_i" /> are queue negatives.</p>
      </DefinitionBlock>

      <MomentumViz />

      <ExampleBlock title="MoCo v1 to v3 Evolution">
        <p>
          <strong>MoCo v1</strong>: Queue + momentum encoder with ResNet.
          <strong>MoCo v2</strong>: Adds SimCLR's MLP projection head and stronger augmentations.
          <strong>MoCo v3</strong>: Adapts to Vision Transformers, removes the queue (uses batch
          negatives like SimCLR but with momentum encoder for stability), and adds a prediction head.
        </p>
      </ExampleBlock>

      <PythonCode
        title="MoCo v2 Core Implementation"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class MoCo(nn.Module):
    def __init__(self, encoder, dim=128, K=65536, m=0.999, tau=0.2):
        super().__init__()
        self.K, self.m, self.tau = K, m, tau

        self.encoder_q = encoder  # query encoder (gradient)
        self.encoder_k = type(encoder)()  # key encoder (momentum)

        # Initialize key encoder = query encoder
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False

        # Queue of negative keys
        self.register_buffer("queue", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self):
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = self.m * p_k.data + (1 - self.m) * p_q.data

    @torch.no_grad()
    def enqueue(self, keys):
        B = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + B] = keys.T
        self.queue_ptr[0] = (ptr + B) % self.K

    def forward(self, x_q, x_k):
        q = F.normalize(self.encoder_q(x_q), dim=-1)

        with torch.no_grad():
            self.momentum_update()
            k = F.normalize(self.encoder_k(x_k), dim=-1)

        # Positive logits: (B, 1)
        l_pos = torch.einsum('bd,bd->b', q, k).unsqueeze(-1) / self.tau
        # Negative logits: (B, K)
        l_neg = torch.einsum('bd,dk->bk', q, self.queue.clone().detach()) / self.tau

        logits = torch.cat([l_pos, l_neg], dim=-1)
        labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)

        self.enqueue(k)
        return F.cross_entropy(logits, labels)

print("MoCo: 65K negatives with batch size 256")
print("Key insight: momentum encoder keeps queue keys consistent")`}
      />

      <NoteBlock type="note" title="MoCo vs SimCLR Trade-offs">
        <p>
          SimCLR requires large batch sizes (4096+) and large GPU memory. MoCo achieves comparable
          results with batch size 256 by maintaining a large queue. However, MoCo's momentum
          encoder adds complexity. In practice, MoCo v3 and DINO (momentum-based) have proven more
          effective for Vision Transformers than pure SimCLR-style approaches.
        </p>
      </NoteBlock>
    </div>
  )
}

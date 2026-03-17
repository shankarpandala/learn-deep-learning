import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function SamplingViz() {
  const [sampleSize, setSampleSize] = useState(2)
  const neighbors = [0, 1, 2, 3, 4, 5, 6, 7]
  const [seed, setSeed] = useState(0)
  const sampled = new Set()
  let s = seed
  while (sampled.size < Math.min(sampleSize, neighbors.length)) {
    s = (s * 1103515245 + 12345) & 0x7fffffff
    sampled.add(neighbors[s % neighbors.length])
  }

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Neighbor Sampling</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Sample size: {sampleSize}
          <input type="range" min={1} max={8} step={1} value={sampleSize} onChange={e => setSampleSize(parseInt(e.target.value))} className="w-24 accent-violet-500" />
        </label>
        <button onClick={() => setSeed(seed + 1)} className="px-3 py-1 rounded bg-violet-500 text-white text-sm hover:bg-violet-600">Resample</button>
      </div>
      <div className="flex items-center gap-6 justify-center">
        <div className="w-14 h-14 rounded-full bg-violet-500 flex items-center justify-center text-white font-bold text-sm">target</div>
        <div className="flex flex-wrap gap-2 max-w-[200px]">
          {neighbors.map(n => (
            <div key={n} className={`w-10 h-10 rounded-full flex items-center justify-center text-xs font-bold ${sampled.has(n) ? 'bg-violet-200 dark:bg-violet-800 text-violet-700 dark:text-violet-300 border-2 border-violet-500' : 'bg-gray-100 dark:bg-gray-800 text-gray-400 border border-gray-300 dark:border-gray-600'}`}>
              n{n}
            </div>
          ))}
        </div>
      </div>
      <p className="text-center text-xs text-gray-500 mt-2">Sample {sampleSize} of 8 neighbors per layer (reduces computation)</p>
    </div>
  )
}

export default function GraphSAGE() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        GraphSAGE (SAmple and aggreGatE) enables inductive learning on graphs by sampling
        and aggregating features from a fixed-size neighborhood, making it scalable to
        large, evolving graphs.
      </p>

      <DefinitionBlock title="GraphSAGE Algorithm">
        <p>For each layer <InlineMath math="k" /> and node <InlineMath math="v" />:</p>
        <BlockMath math="\mathbf{h}_{\mathcal{N}(v)}^{(k)} = \text{AGGREGATE}_k\!\left(\left\{\mathbf{h}_u^{(k-1)} : u \in \mathcal{S}(v)\right\}\right)" />
        <BlockMath math="\mathbf{h}_v^{(k)} = \sigma\!\left(W^{(k)} \cdot \text{CONCAT}\!\left(\mathbf{h}_v^{(k-1)}, \mathbf{h}_{\mathcal{N}(v)}^{(k)}\right)\right)" />
        <p className="mt-2">where <InlineMath math="\mathcal{S}(v)" /> is a fixed-size sample of neighbors.</p>
      </DefinitionBlock>

      <SamplingViz />

      <ExampleBlock title="Aggregator Choices">
        <p><strong>Mean</strong>: <InlineMath math="\text{AGG} = \text{mean}(\{h_u\})" />. Simple, equivalent to GCN.</p>
        <p><strong>Pool</strong>: <InlineMath math="\text{AGG} = \max(\{\sigma(W_\text{pool} h_u + b)\})" />. Non-linear transform before pooling.</p>
        <p><strong>LSTM</strong>: Apply LSTM on a random permutation of neighbors. More expressive but sensitive to ordering.</p>
        <p>Mean aggregator is most commonly used due to simplicity and competitive performance.</p>
      </ExampleBlock>

      <DefinitionBlock title="Mini-Batch Training">
        <p>GraphSAGE enables mini-batch training on large graphs through neighbor sampling:</p>
        <BlockMath math="\text{Layer } K: \text{sample } S_K \text{ neighbors} \to \text{Layer } K-1: \text{sample } S_{K-1} \text{ per node} \to \cdots" />
        <p className="mt-2">Total computation per target node: <InlineMath math="O(\prod_{k=1}^K S_k)" />.
        With <InlineMath math="K=2, S_1=25, S_2=10" />, each node aggregates from up to 250 second-hop neighbors.</p>
      </DefinitionBlock>

      <PythonCode
        title="GraphSAGE with Neighbor Sampling in PyG"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader

class GraphSAGEModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

# Mini-batch training with neighbor sampling
# loader = NeighborLoader(
#     data,
#     num_neighbors=[25, 10],  # sample sizes per layer
#     batch_size=256,
#     input_nodes=data.train_mask,
# )
# for batch in loader:
#     out = model(batch.x, batch.edge_index)
#     loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])

# Key advantage: scales to millions of nodes!
model = GraphSAGEModel(16, 64, 7)
x = torch.randn(100, 16)
edge_index = torch.randint(0, 100, (2, 500))
out = model(x, edge_index)
print(f"Output shape: {out.shape}")`}
      />

      <WarningBlock title="Sampling Variance">
        <p>
          Neighbor sampling introduces variance into gradient estimates. Larger sample sizes
          reduce variance but increase computation. Layer-wise sampling (as in GraphSAGE)
          can lead to exponential neighborhood expansion. Alternatives like ClusterGCN and
          GraphSAINT sample subgraphs instead to reduce this issue.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Inductive Capability">
        <p>
          Unlike GCN (which uses a fixed adjacency matrix), GraphSAGE learns aggregation functions
          that generalize to unseen nodes and graphs. This makes it ideal for production systems
          where the graph evolves over time (e.g., new users joining a social network).
        </p>
      </NoteBlock>
    </div>
  )
}

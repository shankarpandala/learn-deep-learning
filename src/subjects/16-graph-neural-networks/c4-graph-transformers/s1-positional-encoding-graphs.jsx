import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function PEComparisonViz() {
  const [peType, setPeType] = useState('laplacian')
  const nodes = ['v0', 'v1', 'v2', 'v3', 'v4']
  const laplacianPE = [[0.45, -0.37], [0.45, 0.37], [0.45, 0.60], [0.45, -0.60], [0.45, 0.00]]
  const rwPE = [[0.33, 0.11], [0.50, 0.25], [0.33, 0.11], [0.50, 0.25], [0.25, 0.06]]
  const pe = peType === 'laplacian' ? laplacianPE : rwPE

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Graph Positional Encodings</h3>
      <div className="flex items-center gap-2 mb-3">
        <button onClick={() => setPeType('laplacian')} className={`px-3 py-1 rounded text-sm ${peType === 'laplacian' ? 'bg-violet-500 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-600'}`}>Laplacian PE</button>
        <button onClick={() => setPeType('rwse')} className={`px-3 py-1 rounded text-sm ${peType === 'rwse' ? 'bg-violet-500 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-600'}`}>Random Walk SE</button>
      </div>
      <div className="flex gap-2 justify-center flex-wrap">
        {nodes.map((n, i) => (
          <div key={i} className="px-3 py-2 rounded-lg bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-center">
            <div className="text-xs text-gray-500 font-bold">{n}</div>
            <div className="text-xs font-mono text-violet-600 dark:text-violet-400">
              [{pe[i][0].toFixed(2)}, {pe[i][1].toFixed(2)}]
            </div>
          </div>
        ))}
      </div>
      <p className="text-center text-xs text-gray-500 mt-2">
        {peType === 'laplacian' ? 'Laplacian PE: eigenvectors of graph Laplacian (sign ambiguity!)' : 'RWSE: diagonal of random walk matrix powers (no sign ambiguity)'}
      </p>
    </div>
  )
}

export default function PositionalEncodingGraphs() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Unlike sequences, graphs have no canonical ordering of nodes. Positional encodings for
        graphs must encode structural information (centrality, distance, substructure) without
        relying on a fixed node ordering.
      </p>

      <DefinitionBlock title="Laplacian Positional Encoding">
        <p>Use the first <InlineMath math="k" /> non-trivial eigenvectors of the normalized Laplacian:</p>
        <BlockMath math="\text{PE}(v_i) = [\phi_2(v_i), \phi_3(v_i), \ldots, \phi_{k+1}(v_i)] \in \mathbb{R}^k" />
        <p className="mt-2">where <InlineMath math="L\phi_j = \lambda_j \phi_j" />. These capture graph structure at multiple scales.</p>
      </DefinitionBlock>

      <TheoremBlock title="Sign Ambiguity Problem" id="sign-ambiguity">
        <p>Eigenvectors are defined up to sign: if <InlineMath math="\phi" /> is an eigenvector,
        so is <InlineMath math="-\phi" />. This means Laplacian PE is not unique:</p>
        <BlockMath math="\phi_j \text{ and } -\phi_j \text{ are equally valid}" />
        <p className="mt-2">Solutions: (1) use sign-invariant networks (SignNet), (2) use random sign
        augmentation during training, (3) use random walk encodings which avoid the issue entirely.</p>
      </TheoremBlock>

      <DefinitionBlock title="Random Walk Structural Encoding (RWSE)">
        <p>The landing probability of a random walk returning to node <InlineMath math="i" /> after <InlineMath math="k" /> steps:</p>
        <BlockMath math="\text{RWSE}(v_i) = \left[(\hat{A}^1)_{ii}, (\hat{A}^2)_{ii}, \ldots, (\hat{A}^K)_{ii}\right]" />
        <p className="mt-2">where <InlineMath math="\hat{A} = D^{-1}A" />. This encodes local structural information
        (degree at <InlineMath math="k=1" />, triangle count at <InlineMath math="k=3" />, etc.) without sign ambiguity.</p>
      </DefinitionBlock>

      <PEComparisonViz />

      <ExampleBlock title="Other Graph PEs">
        <p><strong>Distance encoding</strong>: shortest-path distances between node pairs.</p>
        <p><strong>Node degree</strong>: simplest structural feature, often surprisingly effective.</p>
        <p><strong>Learnable PE</strong>: Initialize random PE per node, learn via backpropagation.</p>
        <p>Most graph transformers combine multiple PE types for best results.</p>
      </ExampleBlock>

      <PythonCode
        title="Computing Graph Positional Encodings"
        code={`import torch
import numpy as np
from scipy import sparse

def laplacian_pe(edge_index, num_nodes, k=8):
    """Compute Laplacian Positional Encoding."""
    # Build adjacency and Laplacian
    row, col = edge_index
    A = sparse.coo_matrix((np.ones(len(row)), (row, col)),
                          shape=(num_nodes, num_nodes))
    A = A + A.T  # symmetrize
    D = sparse.diags(np.array(A.sum(1)).flatten())
    L = D - A
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(np.array(A.sum(1)).flatten() + 1e-8))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    # Compute smallest eigenvectors (skip first = constant)
    eigenvalues, eigenvectors = sparse.linalg.eigsh(L_norm, k=k+1, which='SM')
    pe = torch.FloatTensor(eigenvectors[:, 1:k+1])  # skip first
    return pe

def random_walk_se(edge_index, num_nodes, walk_length=16):
    """Compute Random Walk Structural Encoding."""
    row, col = edge_index
    A = sparse.coo_matrix((np.ones(len(row)), (row, col)),
                          shape=(num_nodes, num_nodes)).tocsr()
    D_inv = sparse.diags(1.0 / (np.array(A.sum(1)).flatten() + 1e-8))
    RW = D_inv @ A  # random walk matrix
    pe = torch.zeros(num_nodes, walk_length)
    RW_power = sparse.eye(num_nodes)
    for k in range(walk_length):
        RW_power = RW_power @ RW
        pe[:, k] = torch.FloatTensor(RW_power.diagonal())
    return pe

# Example: 10 nodes, random edges
edge_index = np.array([np.random.randint(0, 10, 30), np.random.randint(0, 10, 30)])
lap_pe = laplacian_pe(edge_index, 10, k=4)
rw_pe = random_walk_se(edge_index, 10, walk_length=8)
print(f"Laplacian PE: {lap_pe.shape}, RWSE: {rw_pe.shape}")`}
      />

      <NoteBlock type="note" title="PE in Practice">
        <p>
          Positional encodings are typically concatenated with or added to node features before
          being fed into the graph transformer. The PE dimension is a hyperparameter (commonly 16-64).
          GPS (General Powerful Scalable) uses both Laplacian PE and RWSE together.
        </p>
      </NoteBlock>
    </div>
  )
}

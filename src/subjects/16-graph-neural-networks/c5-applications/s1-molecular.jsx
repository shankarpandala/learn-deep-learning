import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function MoleculeGraphViz() {
  const [showFeatures, setShowFeatures] = useState(false)
  const atoms = [
    { x: 80, y: 80, elem: 'C', color: '#374151' },
    { x: 160, y: 40, elem: 'C', color: '#374151' },
    { x: 240, y: 80, elem: 'O', color: '#dc2626' },
    { x: 160, y: 120, elem: 'N', color: '#2563eb' },
    { x: 80, y: 160, elem: 'C', color: '#374151' },
  ]
  const bonds = [
    { from: 0, to: 1, type: 'single' },
    { from: 1, to: 2, type: 'double' },
    { from: 1, to: 3, type: 'single' },
    { from: 3, to: 4, type: 'single' },
    { from: 4, to: 0, type: 'single' },
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Molecule as a Graph</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          <input type="checkbox" checked={showFeatures} onChange={e => setShowFeatures(e.target.checked)} className="accent-violet-500" />
          Show node features
        </label>
      </div>
      <svg width={320} height={200} className="mx-auto block">
        {bonds.map((b, k) => {
          const offset = b.type === 'double' ? 3 : 0
          return (
            <g key={k}>
              <line x1={atoms[b.from].x} y1={atoms[b.from].y - offset} x2={atoms[b.to].x} y2={atoms[b.to].y - offset} stroke="#9ca3af" strokeWidth={2} />
              {b.type === 'double' && <line x1={atoms[b.from].x} y1={atoms[b.from].y + offset} x2={atoms[b.to].x} y2={atoms[b.to].y + offset} stroke="#9ca3af" strokeWidth={2} />}
            </g>
          )
        })}
        {atoms.map((a, i) => (
          <g key={i}>
            <circle cx={a.x} cy={a.y} r={20} fill={a.color} opacity={0.85} />
            <text x={a.x} y={a.y + 5} textAnchor="middle" fill="white" fontSize={14} fontWeight="bold">{a.elem}</text>
            {showFeatures && <text x={a.x} y={a.y + 32} textAnchor="middle" fill="#7c3aed" fontSize={8}>[Z, deg, hyb, ...]</text>}
          </g>
        ))}
      </svg>
      <p className="text-center text-xs text-gray-500 mt-1">Atoms = nodes, bonds = edges, with rich chemical features</p>
    </div>
  )
}

export default function MolecularPropertyPrediction() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Molecules are naturally represented as graphs, with atoms as nodes and bonds as edges.
        GNNs have become the dominant approach for molecular property prediction, with applications
        in drug discovery, materials science, and chemical engineering.
      </p>

      <DefinitionBlock title="Molecular Graph Representation">
        <p>Node features (per atom):</p>
        <BlockMath math="\mathbf{x}_i = [\text{atomic\_num}, \text{degree}, \text{formal\_charge}, \text{hybridization}, \text{aromaticity}, \ldots]" />
        <p className="mt-2">Edge features (per bond):</p>
        <BlockMath math="\mathbf{e}_{ij} = [\text{bond\_type}, \text{is\_conjugated}, \text{is\_ring}, \text{stereo}]" />
      </DefinitionBlock>

      <MoleculeGraphViz />

      <DefinitionBlock title="Graph-Level Property Prediction">
        <p>For molecular properties (solubility, toxicity, binding affinity):</p>
        <BlockMath math="\hat{y} = \text{MLP}\!\left(\bigoplus_{i \in \mathcal{V}} \mathbf{h}_i^{(L)}\right)" />
        <p className="mt-2">The readout <InlineMath math="\bigoplus" /> aggregates all atom representations into a single
        molecular fingerprint. Sum pooling preserves size information; mean pooling is size-invariant.</p>
      </DefinitionBlock>

      <ExampleBlock title="Key Benchmarks">
        <p><strong>ogbg-molhiv</strong>: Predict HIV inhibition (41K molecules, binary classification, AUC-ROC ~80%).</p>
        <p><strong>PCQM4Mv2</strong>: Predict HOMO-LUMO gap (3.8M molecules, regression, MAE ~0.085 eV).</p>
        <p><strong>QM9</strong>: 12 quantum chemical properties for 134K small molecules.</p>
        <p>State-of-the-art models (GPS, Graphormer) achieve chemical accuracy on several properties.</p>
      </ExampleBlock>

      <PythonCode
        title="Molecular GNN with PyG and RDKit"
        code={`import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.datasets import MoleculeNet

# Load HIV dataset
dataset = MoleculeNet(root='/tmp/hiv', name='HIV')
print(f"Molecules: {len(dataset)}, Features: {dataset.num_features}")

class MolGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1, n_layers=4):
        super().__init__()
        self.atom_encoder = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.layers.append(GINEConv(mlp, edge_dim=3))
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.atom_encoder(x.float())
        for conv in self.layers:
            h = h + conv(h, edge_index, edge_attr.float())  # residual
            h = h.relu()
        h_mol = global_add_pool(h, batch)  # sum over atoms
        return self.output(h_mol)

model = MolGNN(in_dim=9, hidden_dim=128)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")`}
      />

      <NoteBlock type="note" title="3D Molecular Graphs">
        <p>
          Many molecular properties depend on 3D geometry (distances, angles, dihedral angles).
          Equivariant GNNs like SchNet, DimeNet, and PaiNN incorporate 3D coordinates as
          continuous edge features, respecting physical symmetries (rotation, translation, reflection).
          This is crucial for force fields and protein structure prediction.
        </p>
      </NoteBlock>
    </div>
  )
}

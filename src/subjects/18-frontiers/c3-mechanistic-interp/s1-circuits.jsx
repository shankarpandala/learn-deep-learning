import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function CircuitExplorer() {
  const [circuit, setCircuit] = useState('induction')
  const circuits = {
    induction: { name: 'Induction Heads', desc: 'Copy patterns from earlier context: if "A B ... A" appears, predict "B". Found in layer 1-2 pairs.', mechanism: 'Head 0 in L1 attends to previous tokens → Head 5 in L2 attends to token after previous occurrence of current token → copies that token to output' },
    ioi: { name: 'IOI (Indirect Object)', desc: 'In "Alice gave Bob the ball. Alice gave ___", predict "Bob" not "Alice".', mechanism: 'Duplicate token heads detect repeated names → S-inhibition heads suppress the repeated name → Name mover heads promote the non-repeated name to output' },
    superposition: { name: 'Superposition', desc: 'Networks represent more features than dimensions by encoding sparse features in overlapping directions.', mechanism: 'With D dimensions and F >> D features, each feature is a direction in R^D. Features are approximately orthogonal when sparse, allowing interference to be tolerable.' },
  }
  const c = circuits[circuit]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Known Neural Circuits</h3>
      <div className="flex gap-2 mb-3 flex-wrap">
        {Object.entries(circuits).map(([key, val]) => (
          <button key={key} onClick={() => setCircuit(key)}
            className={`px-3 py-1 rounded-lg text-sm transition ${circuit === key ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20 text-sm space-y-2">
        <p className="font-medium text-violet-700 dark:text-violet-300">{c.name}</p>
        <p className="text-gray-600 dark:text-gray-400">{c.desc}</p>
        <p className="text-xs text-gray-500"><strong>Mechanism:</strong> {c.mechanism}</p>
      </div>
    </div>
  )
}

export default function CircuitsFeatures() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Mechanistic interpretability aims to reverse-engineer neural networks by identifying
        meaningful features (directions in activation space) and circuits (compositions of features
        across layers). This approach treats models as programs to be understood, not black boxes.
      </p>

      <DefinitionBlock title="Features and Circuits">
        <p>A <strong>feature</strong> is a direction in activation space that corresponds to a human-interpretable concept. A <strong>circuit</strong> is a subgraph of the network that implements a specific computation:</p>
        <BlockMath math="\text{Circuit} = \{(\text{feature}_i^{(l)}, W_{ij}^{(l \to l+1)}) : \text{implements behavior } f\}" />
        <p className="mt-2">The <strong>linear representation hypothesis</strong> claims that features are represented as directions in activation space: <InlineMath math="v_f \in \mathbb{R}^d" /> where <InlineMath math="\langle h, v_f \rangle" /> measures the presence of feature <InlineMath math="f" /> in hidden state <InlineMath math="h" />.</p>
      </DefinitionBlock>

      <CircuitExplorer />

      <ExampleBlock title="Induction Heads: A Universal Circuit">
        <p>Induction heads are found in virtually all transformer LLMs and implement in-context pattern matching:</p>
        <ol className="list-decimal list-inside mt-2 space-y-1">
          <li>A "previous token head" in layer L attends to the token before the current token</li>
          <li>An "induction head" in layer L+1 uses this to find where the current token appeared before</li>
          <li>It then copies the <em>next</em> token from that earlier occurrence to the output</li>
        </ol>
        <p className="mt-2">This two-layer circuit is responsible for much of in-context learning ability and appears to form during a sharp phase transition in training ("induction bump").</p>
      </ExampleBlock>

      <PythonCode
        title="Analyzing Attention Patterns for Circuits"
        code={`import torch
import torch.nn.functional as F

def compute_attention_pattern(Q, K, mask=None):
    """Compute attention weights for circuit analysis."""
    d_k = Q.shape[-1]
    attn = Q @ K.transpose(-2, -1) / d_k**0.5
    if mask is not None:
        attn = attn.masked_fill(mask == 0, -1e9)
    return F.softmax(attn, dim=-1)

def find_induction_heads(attn_weights, offset=1):
    """Score attention heads for induction behavior.

    Induction heads attend to positions where the current
    token appeared previously, shifted by 'offset'.
    High score on [seq-offset] diagonal = induction head.
    """
    # attn_weights: [heads, seq_len, seq_len]
    H, S, _ = attn_weights.shape
    scores = torch.zeros(H)
    for h in range(H):
        # Check attention on the offset-shifted diagonal
        diag_sum = 0
        count = 0
        for i in range(offset, S):
            diag_sum += attn_weights[h, i, i - offset].item()
            count += 1
        scores[h] = diag_sum / count if count > 0 else 0
    return scores

# Simulate: 12 heads, seq_len=32
attn = torch.rand(12, 32, 32)
attn = F.softmax(attn * 5, dim=-1)  # sharpen
# Make head 5 an induction head (attend to offset-1 diagonal)
for i in range(1, 32):
    attn[5, i, :] *= 0.01
    attn[5, i, i-1] = 0.9
scores = find_induction_heads(attn)
print("Induction head scores:", [f"H{i}:{s:.2f}" for i, s in enumerate(scores)])`}
      />

      <NoteBlock type="note" title="The Superposition Hypothesis">
        <p>
          Neural networks appear to represent more features than they have dimensions, using
          <strong> superposition</strong> — encoding multiple features as nearly-orthogonal directions
          in the same space. This makes individual neurons <em>polysemantic</em> (responding to
          multiple unrelated concepts), complicating interpretability. Sparse autoencoders aim to
          untangle superposition into monosemantic features.
        </p>
      </NoteBlock>
    </div>
  )
}

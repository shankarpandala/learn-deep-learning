import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function PatchingExplorer() {
  const [patchType, setPatchType] = useState('activation')
  const types = {
    activation: { name: 'Activation Patching', desc: 'Replace activations at a specific position and layer from a clean run with those from a corrupted run. Measure how much the output changes.', formula: '\\Delta y = f(h_{\\text{clean}}) - f(h_{\\text{clean}} \\text{ with } h^{(l)}_i \\leftarrow h^{(l)}_{i,\\text{corrupt}})' },
    path: { name: 'Path Patching', desc: 'Patch only the connection between two specific components (e.g., head A to head B), isolating the causal effect of a specific pathway.', formula: '\\Delta y = f(h_{\\text{clean}}) - f(h_{\\text{clean}} \\text{ with edge } A{\\to}B \\text{ corrupted})' },
    resample: { name: 'Causal Scrubbing', desc: 'Systematically test a proposed computational graph by resampling activations at each node. If the hypothesis is correct, resampling should preserve behavior.', formula: '\\text{Match}(G) = 1 - \\frac{\\mathbb{E}[\\text{KL}(f(x) \\| f_{\\text{scrubbed}}(x))]}{\\mathbb{E}[\\text{KL}(f(x) \\| \\text{uniform})]}' },
  }
  const t = types[patchType]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Interpretability Techniques</h3>
      <div className="flex gap-2 mb-3 flex-wrap">
        {Object.entries(types).map(([key, val]) => (
          <button key={key} onClick={() => setPatchType(key)}
            className={`px-3 py-1 rounded-lg text-sm transition ${patchType === key ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20 text-sm space-y-2">
        <p className="text-gray-600 dark:text-gray-400">{t.desc}</p>
        <BlockMath math={t.formula} />
      </div>
    </div>
  )
}

export default function ProbingActivationPatching() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Probing classifiers and activation patching are two key techniques for understanding what
        information neural networks encode and how that information flows through the network to
        produce outputs.
      </p>

      <DefinitionBlock title="Linear Probing">
        <p>Train a linear classifier on frozen internal representations to test what information is encoded:</p>
        <BlockMath math="\hat{y} = \text{softmax}(W_{\text{probe}} \cdot h^{(l)}_i + b)" />
        <p className="mt-2">where <InlineMath math="h^{(l)}_i" /> is the hidden state at layer <InlineMath math="l" />, position <InlineMath math="i" />. High probe accuracy indicates the information is linearly accessible. The probe should be <em>simple</em> (linear) to avoid learning the task itself rather than detecting pre-existing representations.</p>
      </DefinitionBlock>

      <PatchingExplorer />

      <ExampleBlock title="What Probes Reveal About LLMs">
        <p>Linear probes on GPT-2 and LLaMA have uncovered:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>Part-of-speech: 97%+ accuracy from early layers (layer 2-3)</li>
          <li>Syntactic parse trees: recoverable from middle layers</li>
          <li>Factual knowledge (entity types, relationships): peaks in middle layers</li>
          <li>Next-token prediction: only accurate from final layers</li>
          <li>Pattern: low layers = syntax, middle = semantics, high = task-specific</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Linear Probing and Activation Patching"
        code={`import torch
import torch.nn as nn

class LinearProbe(nn.Module):
    """Linear probe for testing information in representations."""
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, hidden_states):
        return self.linear(hidden_states.detach())  # Detach!

def activation_patching(model, clean_input, corrupt_input, layer, position):
    """Measure causal importance of activation at (layer, position).

    1. Run model on clean input, save all activations
    2. Run model on corrupt input, save activation at target (layer, pos)
    3. Run model on clean input but replace target activation with corrupt one
    4. Measure change in output logits
    """
    # Get clean activations and output
    clean_acts = {}
    def save_hook(name):
        def hook(module, input, output):
            clean_acts[name] = output.detach().clone()
        return hook

    # Register hooks on each layer (pseudocode)
    # ... hooks = [model.layers[i].register_hook(save_hook(f"layer_{i}"))]

    # Get corrupt activation at target location
    # corrupt_act = run_model(corrupt_input).activations[layer][position]

    # Patch: replace clean activation with corrupt one at (layer, position)
    # patched_output = run_model_with_patch(clean_input, layer, position, corrupt_act)

    # Measure effect: large change = this activation is important
    # effect = (clean_output - patched_output).norm()

    print("Activation patching workflow:")
    print("1. Clean run -> save activations + output logits")
    print("2. Corrupt run -> save target activation")
    print("3. Patched run -> replace one activation, measure output change")
    print("4. Large change = causally important component")

activation_patching(None, None, None, layer=10, position=15)`}
      />

      <NoteBlock type="note" title="Probing Limitations">
        <p>
          Probing has important caveats: (1) high probe accuracy does not mean the model <em>uses</em>
          that information — it might be an artifact, (2) low probe accuracy does not mean the
          information is absent — it might be encoded nonlinearly, (3) probe complexity matters —
          an MLP probe can "learn" to extract information not linearly present. Activation patching
          provides stronger causal evidence than probing alone.
        </p>
      </NoteBlock>
    </div>
  )
}

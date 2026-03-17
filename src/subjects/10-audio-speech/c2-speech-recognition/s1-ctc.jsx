import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function CTCAlignmentDemo() {
  const [showBlank, setShowBlank] = useState(true)
  const target = 'cat'
  const alignments = [
    { path: ['-', 'c', 'c', 'a', '-', 't', '-'], collapsed: 'cat' },
    { path: ['c', '-', 'a', 'a', 'a', 't', '-'], collapsed: 'cat' },
    { path: ['-', '-', 'c', 'a', 't', '-', '-'], collapsed: 'cat' },
    { path: ['c', 'a', '-', '-', '-', 't', 't'], collapsed: 'cat' },
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">CTC Alignment Paths</h3>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-3">
        Multiple alignments collapse to the same output &quot;{target}&quot;
      </p>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        <input type="checkbox" checked={showBlank} onChange={e => setShowBlank(e.target.checked)} className="accent-violet-500" />
        Highlight blank tokens
      </label>
      <div className="space-y-2">
        {alignments.map((a, i) => (
          <div key={i} className="flex items-center gap-1">
            <span className="text-xs text-gray-400 w-6">#{i + 1}</span>
            {a.path.map((ch, j) => (
              <span key={j} className={`w-8 h-8 flex items-center justify-center rounded text-sm font-mono font-bold
                ${ch === '-' ? (showBlank ? 'bg-violet-100 text-violet-400 dark:bg-violet-900/30 dark:text-violet-500' : 'bg-gray-100 text-gray-400 dark:bg-gray-800') : 'bg-violet-500 text-white'}`}>
                {ch === '-' ? '\u03B5' : ch}
              </span>
            ))}
            <span className="text-sm text-gray-500 ml-2">&rarr; &quot;{a.collapsed}&quot;</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function CTCLossDecoding() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Connectionist Temporal Classification (CTC) enables training sequence models without
        explicit alignment between input and output. It sums over all valid alignments,
        making it the foundation of modern end-to-end ASR systems.
      </p>

      <DefinitionBlock title="CTC Loss">
        <p>
          Given input sequence <InlineMath math="X" /> of length <InlineMath math="T" /> and target
          label sequence <InlineMath math="Y" />, CTC defines:
        </p>
        <BlockMath math="P(Y|X) = \sum_{\pi \in \mathcal{B}^{-1}(Y)} \prod_{t=1}^{T} P(\pi_t | X)" />
        <p className="mt-2">
          where <InlineMath math="\mathcal{B}" /> is the collapsing function that removes blanks
          (<InlineMath math="\epsilon" />) and merges repeated characters. The loss
          is <InlineMath math="\mathcal{L} = -\log P(Y|X)" />.
        </p>
      </DefinitionBlock>

      <CTCAlignmentDemo />

      <TheoremBlock title="Forward-Backward Algorithm" id="ctc-forward-backward">
        <p>
          Computing the CTC loss exactly via enumeration is intractable. The forward variable
          <InlineMath math="\alpha(t, s)" /> gives the probability of emitting the first <InlineMath math="s" /> labels
          in <InlineMath math="t" /> steps:
        </p>
        <BlockMath math="\alpha(t, s) = [\alpha(t{-}1, s) + \alpha(t{-}1, s{-}1) + \alpha(t{-}1, s{-}2)] \cdot P(l'_s | x_t)" />
        <p className="mt-1">
          The third term is included only if <InlineMath math="l'_s \neq \epsilon" /> and
          <InlineMath math="l'_s \neq l'_{s-2}" />. This runs in <InlineMath math="O(T \cdot |Y'|)" /> time.
        </p>
      </TheoremBlock>

      <ExampleBlock title="CTC Decoding Strategies">
        <p><strong>Greedy decoding:</strong> Take <InlineMath math="\arg\max" /> at each timestep, then collapse.</p>
        <p><strong>Beam search:</strong> Maintain top-k paths, merging those that collapse to the same output.</p>
        <p><strong>With language model:</strong></p>
        <BlockMath math="\hat{Y} = \arg\max_Y \log P_\text{CTC}(Y|X) + \lambda \log P_\text{LM}(Y) + \beta |Y|" />
      </ExampleBlock>

      <PythonCode
        title="CTC Loss in PyTorch"
        code={`import torch
import torch.nn as nn

# Simulated ASR output
T, B, C = 50, 2, 29  # time, batch, vocab (26 chars + space + apostrophe + blank)
log_probs = torch.randn(T, B, C).log_softmax(dim=2)

# Targets (variable length)
targets = torch.tensor([3, 1, 20, 0, 8, 5, 12, 12, 15])  # "cat hello"
target_lengths = torch.tensor([3, 5])  # "cat" and "hello"
input_lengths = torch.tensor([T, T])

# CTC loss
ctc_loss = nn.CTCLoss(blank=28, zero_infinity=True)
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
print(f"CTC loss: {loss.item():.4f}")

# Greedy decoding
predictions = log_probs[:, 0, :].argmax(dim=-1)
# Collapse: remove blanks and repeated characters
collapsed = []
prev = -1
for p in predictions:
    if p != 28 and p != prev:
        collapsed.append(p.item())
    prev = p
print(f"Decoded indices: {collapsed}")`}
      />

      <NoteBlock type="note" title="CTC Assumptions & Limitations">
        <p>
          CTC assumes <strong>conditional independence</strong> between output tokens at each timestep,
          which limits its ability to model language structure. This is why CTC is often combined
          with an external language model or used alongside attention-based decoders in hybrid systems.
        </p>
      </NoteBlock>
    </div>
  )
}

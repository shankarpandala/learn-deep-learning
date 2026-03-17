import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExerciseBlock from '../../../components/content/ExerciseBlock.jsx'

function PerplexityCalculator() {
  const [probs, setProbs] = useState([0.2, 0.5, 0.3, 0.8, 0.1])

  const logProb = probs.reduce((s, p) => s + Math.log2(Math.max(p, 1e-10)), 0)
  const entropy = -logProb / probs.length
  const perplexity = Math.pow(2, entropy)

  const updateProb = (i, val) => {
    const next = [...probs]
    next[i] = parseFloat(val) || 0.01
    setProbs(next)
  }

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Perplexity Calculator</h3>
      <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">Adjust the model's predicted probability for each word:</p>
      <div className="flex flex-wrap gap-3 mb-4">
        {probs.map((p, i) => (
          <label key={i} className="text-xs text-gray-600 dark:text-gray-400">
            w<sub>{i + 1}</sub>:
            <input type="number" min={0.01} max={1} step={0.05} value={p}
              onChange={e => updateProb(i, e.target.value)}
              className="ml-1 w-16 rounded border px-1 py-0.5 text-xs dark:bg-gray-800 dark:border-gray-600" />
          </label>
        ))}
      </div>
      <div className="flex gap-6 text-sm">
        <span className="text-gray-600 dark:text-gray-400">Cross-entropy: <span className="font-bold text-violet-600 dark:text-violet-400">{entropy.toFixed(3)} bits</span></span>
        <span className="text-gray-600 dark:text-gray-400">Perplexity: <span className="font-bold text-violet-600 dark:text-violet-400">{perplexity.toFixed(2)}</span></span>
      </div>
    </div>
  )
}

export default function Perplexity() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Perplexity is the standard intrinsic evaluation metric for language models. It measures
        how well a model predicts a held-out test set and can be interpreted as the weighted
        average number of choices the model is uncertain between at each step.
      </p>

      <DefinitionBlock title="Perplexity">
        <p>For a test sequence <InlineMath math="w_1, w_2, \ldots, w_N" />:</p>
        <BlockMath math="\text{PPL}(W) = P(w_1, w_2, \ldots, w_N)^{-1/N} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i \mid w_{<i})\right)" />
        <p className="mt-2">Lower perplexity indicates a better model. A perplexity of <InlineMath math="k" /> means the model is as uncertain as choosing uniformly among <InlineMath math="k" /> options.</p>
      </DefinitionBlock>

      <TheoremBlock title="Perplexity and Cross-Entropy" id="ppl-entropy">
        <p>Perplexity is the exponentiation of the cross-entropy loss:</p>
        <BlockMath math="\text{PPL} = 2^{H(P, Q)} = 2^{-\frac{1}{N}\sum_{i=1}^{N}\log_2 P(w_i \mid w_{<i})}" />
        <p className="mt-2">where <InlineMath math="H(P, Q)" /> is the cross-entropy between the true distribution and the model distribution.</p>
      </TheoremBlock>

      <PerplexityCalculator />

      <DefinitionBlock title="Bits-per-Character (BPC)">
        <p>For character-level models, the analogous metric is bits-per-character:</p>
        <BlockMath math="\text{BPC} = -\frac{1}{C}\sum_{i=1}^{C}\log_2 P(c_i \mid c_{<i})" />
        <p className="mt-2">where <InlineMath math="C" /> is the total number of characters. English text typically achieves around 1.0-1.3 BPC with strong models.</p>
      </DefinitionBlock>

      <ExampleBlock title="Comparing Models by Perplexity">
        <p>Typical perplexities on Penn Treebank:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>Trigram + Kneser-Ney: ~140</li>
          <li>LSTM LM: ~60</li>
          <li>Transformer LM: ~20-30</li>
          <li>GPT-2 (large): ~18</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Computing Perplexity in PyTorch"
        code={`import torch
import torch.nn.functional as F
import math

def compute_perplexity(model, dataloader, device="cpu"):
    """Compute perplexity over a dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)

            logits = model(input_ids)  # (B, T, V)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += targets.numel()

    avg_nll = total_loss / total_tokens
    perplexity = math.exp(avg_nll)
    return perplexity

# Example usage
# ppl = compute_perplexity(model, test_loader)
# print(f"Test perplexity: {ppl:.2f}")`}
      />

      <ExerciseBlock title="Exercise: Perplexity Bounds">
        <p>
          A language model has a vocabulary of 50,000 words. What is the perplexity of a uniform
          model that assigns equal probability to every word? If the model achieves a perplexity of
          25, how many bits per word does this correspond to?
        </p>
        <p className="mt-2 text-sm text-gray-500">
          Hint: <InlineMath math="\text{PPL}_{\text{uniform}} = |V|" /> and <InlineMath math="\text{bits} = \log_2(\text{PPL})" />.
        </p>
      </ExerciseBlock>

      <NoteBlock type="note" title="Caveats of Perplexity">
        <p>
          Perplexity is only comparable across models using the same vocabulary and tokenization.
          A BPE-based model and a word-level model cannot be directly compared via perplexity.
          Additionally, lower perplexity does not always mean better performance on downstream tasks.
        </p>
      </NoteBlock>
    </div>
  )
}

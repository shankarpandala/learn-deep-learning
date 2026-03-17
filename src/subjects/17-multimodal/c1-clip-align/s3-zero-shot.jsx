import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ZeroShotDemo() {
  const [selectedClass, setSelectedClass] = useState(0)
  const classes = ['a photo of a cat', 'a photo of a dog', 'a photo of a bird', 'a photo of a car']
  const similarities = [
    [0.92, 0.15, 0.08, 0.02],
    [0.12, 0.89, 0.10, 0.03],
    [0.06, 0.08, 0.94, 0.01],
    [0.03, 0.02, 0.01, 0.97],
  ]
  const probs = similarities[selectedClass]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Zero-Shot Classification Demo</h3>
      <p className="text-sm text-gray-500 mb-3">Select an image type to see similarity scores across text prompts</p>
      <div className="flex gap-2 mb-4 flex-wrap">
        {['Cat', 'Dog', 'Bird', 'Car'].map((name, i) => (
          <button key={i} onClick={() => setSelectedClass(i)}
            className={`px-3 py-1 rounded-lg text-sm transition ${selectedClass === i ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {name} Image
          </button>
        ))}
      </div>
      <div className="space-y-2">
        {classes.map((cls, j) => (
          <div key={j} className="flex items-center gap-3">
            <span className="text-xs text-gray-500 w-40 truncate">"{cls}"</span>
            <div className="flex-1 h-5 bg-gray-100 dark:bg-gray-800 rounded overflow-hidden">
              <div className="h-full bg-violet-500 rounded transition-all duration-300" style={{ width: `${probs[j] * 100}%` }} />
            </div>
            <span className="text-xs font-mono w-12 text-right text-gray-600 dark:text-gray-400">{(probs[j] * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function ZeroShotClassification() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        CLIP's shared embedding space enables zero-shot classification — recognizing categories
        never seen during training by comparing image embeddings with text descriptions of classes.
        This eliminates the need for labeled training data for new tasks.
      </p>

      <DefinitionBlock title="Zero-Shot Classification with CLIP">
        <p>Given an image <InlineMath math="I" /> and candidate class names <InlineMath math="\{c_1, \ldots, c_K\}" />, form text prompts like "a photo of a [class]". The predicted class is:</p>
        <BlockMath math="\hat{y} = \arg\max_{k} \frac{\exp(\text{sim}(f_I(I), f_T(c_k))/\tau)}{\sum_{j=1}^{K}\exp(\text{sim}(f_I(I), f_T(c_j))/\tau)}" />
      </DefinitionBlock>

      <ZeroShotDemo />

      <ExampleBlock title="Prompt Engineering for Zero-Shot CLIP">
        <p>The choice of text prompt significantly affects performance. Using ensembles of prompts improves accuracy:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li><code>"a photo of a dog"</code> — baseline template</li>
          <li><code>"a centered satellite photo of a dog"</code> — domain-specific</li>
          <li><code>"a good photo of a dog"</code>, <code>"a bad photo of a dog"</code> — quality variations</li>
          <li>Ensemble: average embeddings across <InlineMath math="M" /> prompts per class</li>
        </ul>
        <BlockMath math="\bar{t}_k = \frac{1}{M}\sum_{m=1}^{M} f_T(\text{prompt}_m(c_k))" />
      </ExampleBlock>

      <PythonCode
        title="Zero-Shot Classification with CLIP"
        code={`import torch
import torch.nn.functional as F

def zero_shot_classify(image_features, text_features, temperature=0.01):
    """Zero-shot classification using precomputed CLIP features.

    Args:
        image_features: [N, D] normalized image embeddings
        text_features: [K, D] normalized text embeddings for K classes
    Returns:
        predictions: [N] predicted class indices
        probs: [N, K] class probabilities
    """
    # Cosine similarity -> softmax probabilities
    logits = image_features @ text_features.T / temperature
    probs = F.softmax(logits, dim=-1)
    predictions = probs.argmax(dim=-1)
    return predictions, probs

# Example: 100 images, 10 classes, 512-dim features
image_feats = F.normalize(torch.randn(100, 512), dim=-1)
text_feats = F.normalize(torch.randn(10, 512), dim=-1)

preds, probs = zero_shot_classify(image_feats, text_feats)
print(f"Predictions shape: {preds.shape}")
print(f"Top class probabilities: {probs.max(dim=-1).values[:5]}")`}
      />

      <NoteBlock type="note" title="Image-Text Retrieval">
        <p>
          The same embedding space supports bidirectional retrieval: given an image, find the most
          relevant texts (image-to-text retrieval), or given a text query, find matching images
          (text-to-image retrieval). This is the backbone of many visual search systems.
        </p>
      </NoteBlock>

      <NoteBlock type="warning" title="Distribution Shift Limitations">
        <p>
          While CLIP is robust to many distribution shifts, it can still fail on specialized domains
          (medical imaging, satellite imagery) where internet-scraped training data provides poor coverage.
          Fine-tuning or domain-specific adapters are often needed.
        </p>
      </NoteBlock>
    </div>
  )
}

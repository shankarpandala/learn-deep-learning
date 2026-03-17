import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ParamComparison() {
  const [spatialSize, setSpatialSize] = useState(7)
  const [channels, setChannels] = useState(512)
  const [numClasses, setNumClasses] = useState(1000)

  const fcParams = spatialSize * spatialSize * channels * numClasses
  const gapParams = channels * numClasses

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">FC vs Global Pooling Parameter Count</h3>
      <div className="grid grid-cols-3 gap-3 mb-4">
        {[
          ['Spatial', spatialSize, setSpatialSize, 1, 14],
          ['Channels', channels, setChannels, 64, 2048],
          ['Classes', numClasses, setNumClasses, 10, 1000],
        ].map(([label, val, setter, min, max]) => (
          <label key={label} className="text-sm text-gray-600 dark:text-gray-400">
            {label}: <strong>{val}</strong>
            <input type="range" min={min} max={max} step={label === 'Channels' ? 64 : 1} value={val} onChange={e => setter(parseInt(e.target.value))} className="w-full accent-violet-500" />
          </label>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3 text-center">
        <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-800">
          <p className="text-xs text-gray-500">FC Layer</p>
          <p className="text-lg font-bold text-gray-700 dark:text-gray-300">{fcParams.toLocaleString()}</p>
        </div>
        <div className="p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20">
          <p className="text-xs text-gray-500">GAP + Linear</p>
          <p className="text-lg font-bold text-violet-700 dark:text-violet-300">{gapParams.toLocaleString()}</p>
        </div>
      </div>
      <p className="text-center mt-2 text-sm text-gray-600 dark:text-gray-400">
        Reduction: <strong className="text-violet-600">{(fcParams / gapParams).toFixed(0)}x</strong> fewer parameters
      </p>
    </div>
  )
}

export default function GlobalPooling() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Global Average Pooling (GAP) replaces the fully connected layers at the end of a CNN
        by averaging each feature map into a single value. Introduced by Network-in-Network
        and adopted by GoogLeNet and ResNet, GAP dramatically reduces parameters and overfitting.
      </p>

      <DefinitionBlock title="Global Average Pooling">
        <BlockMath math="z_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} X_c[i, j]" />
        <p className="mt-2">
          Converts a <InlineMath math="C \times H \times W" /> feature volume into a{' '}
          <InlineMath math="C" />-dimensional vector, one value per channel.
        </p>
      </DefinitionBlock>

      <TheoremBlock title="Parameter Reduction" id="gap-reduction">
        <p>For a feature map of size <InlineMath math="C \times H \times W" /> and <InlineMath math="K" /> output classes:</p>
        <BlockMath math="\text{FC params} = C \cdot H \cdot W \cdot K, \quad \text{GAP + Linear} = C \cdot K" />
        <p className="mt-2">
          The reduction factor is <InlineMath math="H \times W" />, typically{' '}
          <InlineMath math="7 \times 7 = 49\times" /> in standard architectures.
        </p>
      </TheoremBlock>

      <ParamComparison />

      <ExampleBlock title="GAP Enables Any Input Size">
        <p>
          Since GAP pools over all spatial positions regardless of size, a model using GAP
          can accept inputs of any spatial resolution at test time. A model trained on{' '}
          <InlineMath math="224 \times 224" /> can also classify{' '}
          <InlineMath math="320 \times 320" /> images without modification.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Global Pooling in PyTorch"
        code={`import torch
import torch.nn as nn

x = torch.randn(1, 512, 7, 7)

# Global Average Pooling
gap = nn.AdaptiveAvgPool2d(1)
out = gap(x).squeeze(-1).squeeze(-1)
print(f"GAP: {x.shape} -> {out.shape}")  # [1, 512]

# Global Max Pooling
gmp = nn.AdaptiveMaxPool2d(1)
out_max = gmp(x).squeeze(-1).squeeze(-1)
print(f"GMP: {x.shape} -> {out_max.shape}")  # [1, 512]

# Typical classification head
classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(512, 1000)
)
logits = classifier(x)
print(f"Classifier output: {logits.shape}")  # [1, 1000]
print(f"Classifier params: {sum(p.numel() for p in classifier.parameters()):,}")`}
      />

      <NoteBlock type="note" title="GeM Pooling">
        <p>
          Generalized Mean (GeM) pooling generalizes both max and average pooling:{' '}
          <InlineMath math="z_c = \left(\frac{1}{HW}\sum X_c^p\right)^{1/p}" /> where{' '}
          <InlineMath math="p" /> is a learnable parameter. When <InlineMath math="p=1" /> it
          becomes average pooling; as <InlineMath math="p \to \infty" /> it approaches max pooling.
        </p>
      </NoteBlock>
    </div>
  )
}

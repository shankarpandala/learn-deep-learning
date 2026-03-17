import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ParamComparison() {
  const [channels, setChannels] = useState(512)
  const [spatialSize, setSpatialSize] = useState(7)
  const [numClasses, setNumClasses] = useState(1000)

  const fcParams = channels * spatialSize * spatialSize * numClasses
  const gapParams = channels * numClasses
  const reduction = ((1 - gapParams / fcParams) * 100).toFixed(1)

  const fmt = (n) => n > 1e6 ? (n / 1e6).toFixed(1) + 'M' : (n / 1e3).toFixed(0) + 'K'

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">FC vs GAP Parameter Count</h3>
      <div className="grid grid-cols-3 gap-3 mb-4">
        {[
          ['Channels', channels, setChannels, 64, 2048],
          ['Spatial', spatialSize, setSpatialSize, 1, 14],
          ['Classes', numClasses, setNumClasses, 10, 1000],
        ].map(([label, val, setter, min, max]) => (
          <label key={label} className="text-sm text-gray-600 dark:text-gray-400">
            {label}: <strong>{val}</strong>
            <input type="range" min={min} max={max} step={label === 'Classes' ? 10 : 1} value={val} onChange={e => setter(parseInt(e.target.value))} className="w-full accent-violet-500" />
          </label>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div className="p-3 rounded bg-gray-50 dark:bg-gray-800">
          <p className="font-semibold text-gray-700 dark:text-gray-300">Flatten + FC</p>
          <p className="text-violet-600 dark:text-violet-400 font-mono">{fmt(fcParams)} params</p>
        </div>
        <div className="p-3 rounded bg-gray-50 dark:bg-gray-800">
          <p className="font-semibold text-gray-700 dark:text-gray-300">GAP + FC</p>
          <p className="text-violet-600 dark:text-violet-400 font-mono">{fmt(gapParams)} params ({reduction}% fewer)</p>
        </div>
      </div>
    </div>
  )
}

export default function GlobalPooling() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Global Average Pooling (GAP) aggregates each feature map into a single value, replacing
        large fully connected layers and significantly reducing parameter count and overfitting risk.
      </p>

      <DefinitionBlock title="Global Average Pooling">
        <BlockMath math="z_c = \frac{1}{H \times W} \sum_{i=1}^{H}\sum_{j=1}^{W} x_{c,i,j}" />
        <p className="mt-2">
          Produces one scalar per channel. A <InlineMath math="C \times H \times W" /> tensor becomes
          a <InlineMath math="C \times 1 \times 1" /> vector.
        </p>
      </DefinitionBlock>

      <TheoremBlock title="Parameter Reduction" id="gap-params">
        <p>Replacing flatten + FC with GAP + FC:</p>
        <BlockMath math="\text{FC: } C \cdot H \cdot W \cdot N_{\text{classes}} \quad \rightarrow \quad \text{GAP+FC: } C \cdot N_{\text{classes}}" />
        <p className="mt-2">
          Reduction factor: <InlineMath math="H \times W" />. For <InlineMath math="7 \times 7" /> feature maps,
          this is a 49x reduction in classifier parameters.
        </p>
      </TheoremBlock>

      <ParamComparison />

      <ExampleBlock title="NIN to ResNet">
        <p>
          GAP was introduced in <strong>Network in Network</strong> (Lin et al., 2013) and became
          standard from GoogLeNet and ResNet onward. It enforces correspondence between feature maps
          and categories, acting as a structural regularizer.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Global Pooling in Practice"
        code={`import torch
import torch.nn as nn

x = torch.randn(1, 512, 7, 7)

# Global Average Pooling
gap = nn.AdaptiveAvgPool2d((1, 1))
pooled = gap(x).squeeze(-1).squeeze(-1)  # [1, 512]

# Global Max Pooling
gmp = nn.AdaptiveMaxPool2d((1, 1))
max_pooled = gmp(x).squeeze(-1).squeeze(-1)  # [1, 512]

# Classification head with GAP (ResNet-style)
classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(512, 1000)
)
logits = classifier(x)
print(f"Feature maps: {x.shape}")       # [1, 512, 7, 7]
print(f"Logits: {logits.shape}")         # [1, 1000]
print(f"Classifier params: {sum(p.numel() for p in classifier.parameters()):,}")`}
      />

      <NoteBlock type="note" title="Global Pooling Variants">
        <p>
          <strong>GeM (Generalized Mean Pooling)</strong> uses a learnable power parameter <InlineMath math="p" />:
          <InlineMath math="\;z_c = \left(\frac{1}{HW}\sum x_{c,i,j}^p\right)^{1/p}" />. When <InlineMath math="p=1" /> it
          reduces to average pooling; as <InlineMath math="p \to \infty" /> it approaches max pooling.
          GeM is popular in image retrieval tasks.
        </p>
      </NoteBlock>
    </div>
  )
}

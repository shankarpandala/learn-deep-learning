import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function PolicyViz() {
  const [numOps, setNumOps] = useState(2)
  const [magnitude, setMagnitude] = useState(9)

  const ops = ['Rotate', 'ShearX', 'TranslateY', 'AutoContrast', 'Equalize', 'Posterize', 'Solarize', 'Color', 'Brightness', 'Sharpness']

  const selectedOps = Array.from({ length: numOps }, (_, i) => {
    const idx = (i * 3 + magnitude) % ops.length
    return { name: ops[idx], mag: Math.min(magnitude + i, 10) }
  })

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">RandAugment Policy Sampler</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          N (ops): {numOps}
          <input type="range" min={1} max={4} step={1} value={numOps} onChange={e => setNumOps(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          M (magnitude): {magnitude}
          <input type="range" min={1} max={10} step={1} value={magnitude} onChange={e => setMagnitude(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <div className="flex gap-3 flex-wrap">
        {selectedOps.map((op, i) => (
          <div key={i} className="rounded-lg bg-violet-50 dark:bg-violet-900/20 px-4 py-3 text-center">
            <p className="text-sm font-semibold text-violet-700 dark:text-violet-300">{op.name}</p>
            <div className="mt-1 h-2 w-20 rounded bg-gray-200 dark:bg-gray-700">
              <div className="h-full rounded bg-violet-500" style={{ width: `${op.mag * 10}%` }} />
            </div>
            <p className="text-xs text-gray-500 mt-1">mag: {op.mag}</p>
          </div>
        ))}
      </div>
      <p className="text-xs text-gray-500 mt-3">
        RandAugment randomly selects {numOps} operations with uniform magnitude {magnitude}/10 for each image.
      </p>
    </div>
  )
}

export default function AdvancedAugmentation() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Manual augmentation design requires domain expertise and extensive tuning.
        AutoAugment and RandAugment automate this process by learning or simplifying
        augmentation policy selection.
      </p>

      <DefinitionBlock title="AutoAugment">
        <p>
          Uses reinforcement learning to search for optimal augmentation policies. A policy
          consists of <InlineMath math="N" /> sub-policies, each with two operations specified by:
        </p>
        <BlockMath math="\text{Sub-policy} = \{(op_1, p_1, m_1), (op_2, p_2, m_2)\}" />
        <p className="mt-2">
          where <InlineMath math="op" /> is the operation, <InlineMath math="p" /> is the probability
          of applying it, and <InlineMath math="m" /> is the magnitude. The search space has
          <InlineMath math="\sim 10^{32}" /> possible policies.
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="RandAugment">
        <p>
          Drastically simplifies the search space to just two parameters:
        </p>
        <BlockMath math="\text{RandAugment}(N, M): \text{apply } N \text{ random ops, each with magnitude } M" />
        <p className="mt-2">
          <InlineMath math="N" /> and <InlineMath math="M" /> can be tuned with simple grid search.
          Despite its simplicity, RandAugment matches or exceeds AutoAugment performance.
        </p>
      </DefinitionBlock>

      <PolicyViz />

      <TheoremBlock title="Why RandAugment Works" id="randaugment-theory">
        <p>
          The key insight is that optimal augmentation magnitude tends to scale with model
          and dataset size. A single shared magnitude parameter <InlineMath math="M" /> captures
          this relationship, eliminating the need for per-operation magnitude tuning:
        </p>
        <BlockMath math="M^* \propto \log(\text{model size} \times \text{dataset size})" />
      </TheoremBlock>

      <ExampleBlock title="TrivialAugment">
        <p>
          TrivialAugment (2021) further simplifies: apply exactly <strong>one</strong> random
          operation with a random magnitude per image. No hyperparameters to tune at all.
          Surprisingly, this matches RandAugment on ImageNet while being even simpler.
        </p>
      </ExampleBlock>

      <PythonCode
        title="RandAugment and AutoAugment in PyTorch"
        code={`import torch
from torchvision import transforms

# RandAugment: N operations, magnitude M (0-31 scale in torchvision)
train_transform_rand = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),  # N=2, M=9
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# AutoAugment with ImageNet policy
train_transform_auto = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# TrivialAugment
train_transform_trivial = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Tuning RandAugment: simple grid search over N and M
for N in [1, 2, 3]:
    for M in [5, 9, 14]:
        aug = transforms.RandAugment(num_ops=N, magnitude=M)
        print(f"RandAugment(N={N}, M={M})")`}
      />

      <NoteBlock type="note" title="Choosing an Augmentation Strategy">
        <p>
          Start with <strong>TrivialAugment</strong> for zero-config baseline.
          Use <strong>RandAugment</strong> if you can afford to tune N and M (typically N=2,
          M=9 for ImageNet-scale). <strong>AutoAugment</strong> is mainly historical — the
          search cost rarely justifies the marginal gain over RandAugment.
        </p>
      </NoteBlock>
    </div>
  )
}

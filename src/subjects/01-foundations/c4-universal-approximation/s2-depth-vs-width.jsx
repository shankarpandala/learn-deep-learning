import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function DepthWidthViz() {
  const [depth, setDepth] = useState(3)
  const [width, setWidth] = useState(4)
  const W = 420, H = 260, padL = 40, padR = 40

  const layers = depth + 2 // input + hidden layers + output
  const layerX = i => padL + (i / (layers - 1)) * (W - padL - padR)
  const nodeY = (j, count) => H / 2 + (j - (count - 1) / 2) * Math.min(30, (H - 40) / count)

  const layerSizes = [2, ...Array(depth).fill(width), 1]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Network Architecture Visualizer</h3>
      <div className="flex flex-wrap items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Depth (hidden layers): {depth}
          <input type="range" min={1} max={6} step={1} value={depth} onChange={e => setDepth(parseInt(e.target.value))} className="w-24 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Width: {width}
          <input type="range" min={2} max={8} step={1} value={width} onChange={e => setWidth(parseInt(e.target.value))} className="w-24 accent-violet-500" />
        </label>
        <span className="text-sm text-gray-500 dark:text-gray-400">
          Params: {layerSizes.reduce((s, n, i) => i === 0 ? s : s + (layerSizes[i - 1] + 1) * n, 0)}
        </span>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        {layerSizes.map((size, li) =>
          li > 0 && layerSizes[li - 1] <= 8 && size <= 8 &&
          Array.from({ length: layerSizes[li - 1] }, (_, pi) =>
            Array.from({ length: size }, (_, ni) => (
              <line key={`e-${li}-${pi}-${ni}`} x1={layerX(li - 1)} y1={nodeY(pi, layerSizes[li - 1])}
                x2={layerX(li)} y2={nodeY(ni, size)} stroke="#d1d5db" strokeWidth={0.5} />
            ))
          )
        )}
        {layerSizes.map((size, li) =>
          Array.from({ length: Math.min(size, 8) }, (_, ni) => (
            <circle key={`n-${li}-${ni}`} cx={layerX(li)} cy={nodeY(ni, Math.min(size, 8))}
              r={6} fill={li === 0 ? '#8b5cf6' : li === layerSizes.length - 1 ? '#f97316' : '#a78bfa'} />
          ))
        )}
        {layerSizes.map((_, li) => (
          <text key={`l-${li}`} x={layerX(li)} y={H - 5} fontSize={9} fill="#9ca3af" textAnchor="middle">
            {li === 0 ? 'Input' : li === layerSizes.length - 1 ? 'Output' : `H${li}`}
          </text>
        ))}
      </svg>
    </div>
  )
}

export default function DepthVsWidth() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        While the UAT shows that width alone suffices for universal approximation, depth
        provides exponentially more efficient representations. This insight is the theoretical
        foundation for why deep learning works better than wide, shallow networks.
      </p>

      <TheoremBlock title="Exponential Separation (Telgarsky, 2016)" id="depth-separation">
        <p>
          There exist functions computable by networks of depth <InlineMath math="k" /> with
          polynomial width, that cannot be approximated by networks of
          depth <InlineMath math="O(k^{1/3})" /> unless their width is exponential:
        </p>
        <BlockMath math="\text{depth } k, \text{ width } O(1) \quad \text{vs} \quad \text{depth } O(k^{1/3}), \text{ width } 2^{\Omega(k^{1/3})}" />
        <p>
          Specifically, a ReLU network with <InlineMath math="L" /> layers
          and <InlineMath math="O(1)" /> neurons per layer can produce
          functions with <InlineMath math="O(2^L)" /> linear regions.
        </p>
      </TheoremBlock>

      <DefinitionBlock title="Linear Regions in ReLU Networks">
        <p>
          A ReLU network partitions input space into linear regions where the function is affine.
          The maximum number of linear regions for depth <InlineMath math="L" /> and
          width <InlineMath math="w" /> is:
        </p>
        <BlockMath math="\text{regions} \leq \left(\prod_{l=1}^{L-1} \lfloor w/n \rfloor\right) \sum_{j=0}^{n} \binom{w}{j}" />
        <p className="mt-2">
          The product over layers makes this grow exponentially with depth but only
          polynomially with width.
        </p>
      </DefinitionBlock>

      <DepthWidthViz />

      <ExampleBlock title="Compositionality: Why Depth Wins">
        <p>
          Consider representing a function like <InlineMath math="f(x) = h_3(h_2(h_1(x)))" /> where
          each <InlineMath math="h_i" /> requires <InlineMath math="k" /> neurons. A deep network
          needs <InlineMath math="3k" /> neurons total, while a shallow network trying to represent
          the same composed function may need <InlineMath math="O(k^3)" /> neurons. Real-world
          features are naturally compositional: edges compose into textures, textures into parts,
          parts into objects.
        </p>
      </ExampleBlock>

      <TheoremBlock title="Feature Hierarchy" id="feature-hierarchy">
        <p>
          Deep networks learn hierarchical representations. In layer <InlineMath math="l" />,
          each neuron&apos;s effective receptive field grows with depth. The representation
          at each layer can be seen as:
        </p>
        <BlockMath math="h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)})" />
        <p>
          Each layer builds increasingly abstract features. This hierarchy aligns with
          the compositional structure of natural data, explaining why depth is so effective
          for images, language, and structured data.
        </p>
      </TheoremBlock>

      <PythonCode
        title="Comparing Shallow vs Deep Networks"
        code={`import torch
import torch.nn as nn

x = torch.linspace(-2, 2, 1000).unsqueeze(1)
y = torch.sin(10 * x) * torch.exp(-x**2)

shallow = nn.Sequential(  # ~1600 params, 1 hidden layer
    nn.Linear(1, 800), nn.ReLU(), nn.Linear(800, 1))

deep = nn.Sequential(     # ~1600 params, 4 hidden layers
    nn.Linear(1, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU(),
    nn.Linear(20, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU(),
    nn.Linear(20, 1))

count = lambda m: sum(p.numel() for p in m.parameters())
print(f"Shallow: {count(shallow)} params, Deep: {count(deep)} params")

for name, model in [("Shallow", shallow), ("Deep", deep)]:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(3000):
        loss = nn.MSELoss()(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"{name} final MSE: {loss.item():.6f}")`}
      />

      <WarningBlock title="Depth Is Not Free">
        <p>
          Deeper networks face vanishing/exploding gradients, are harder to optimize, and
          require techniques like residual connections, batch normalization, and careful
          initialization. The practical benefits of depth only materialize with these
          engineering advances. Without them, very deep networks may perform worse than
          shallower alternatives.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Practical Guidance">
        <p>
          In practice, moderate depth (3-10 layers for MLPs, 50-150 for ConvNets) with skip
          connections outperforms both very shallow and extremely deep architectures without
          residuals. The lottery ticket hypothesis suggests that within large networks, small
          subnetworks exist that can match the full network&apos;s performance.
        </p>
      </NoteBlock>
    </div>
  )
}

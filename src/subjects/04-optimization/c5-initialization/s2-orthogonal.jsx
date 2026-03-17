import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function SingularValueViz() {
  const [initType, setInitType] = useState('orthogonal')
  const W = 380, H = 160, n = 20

  const svGaussian = Array.from({ length: n }, (_, i) => {
    const x = (i + 1) / n
    return 1.0 + 0.6 * Math.exp(-3 * x) - 0.3 * x + 0.15 * Math.sin(i)
  }).sort((a, b) => b - a)

  const svOrthogonal = Array.from({ length: n }, () => 1.0)

  const svLSUV = Array.from({ length: n }, (_, i) => {
    return 1.0 + 0.05 * Math.sin(i * 2) - 0.02 * Math.cos(i)
  }).sort((a, b) => b - a)

  const svs = initType === 'orthogonal' ? svOrthogonal : initType === 'lsuv' ? svLSUV : svGaussian
  const maxSV = Math.max(...svGaussian) * 1.2
  const sx = W / (n + 1), sy = (H - 30) / maxSV, barW = sx * 0.7

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Singular Value Distribution</h3>
      <div className="flex gap-2 mb-3">
        {['gaussian', 'orthogonal', 'lsuv'].map(t => (
          <button key={t} onClick={() => setInitType(t)}
            className={`px-3 py-1 rounded text-xs font-medium ${initType === t ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'}`}>
            {t === 'gaussian' ? 'Gaussian' : t === 'orthogonal' ? 'Orthogonal' : 'LSUV'}
          </button>
        ))}
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={0} y1={H - 20} x2={W} y2={H - 20} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={0} y1={H - 20 - 1.0 * sy} x2={W} y2={H - 20 - 1.0 * sy} stroke="#f97316" strokeWidth={0.8} strokeDasharray="3,3" />
        {svs.map((v, i) => (
          <rect key={i} x={i * sx + (sx - barW) / 2 + sx / 2} y={H - 20 - v * sy} width={barW} height={v * sy} fill="#8b5cf6" rx={2} opacity={0.7} />
        ))}
      </svg>
      <div className="mt-1 text-center text-xs text-gray-500">Orange = ideal σ=1 | Bars = singular values (sorted)</div>
    </div>
  )
}

export default function OrthogonalInit() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Orthogonal initialization sets weight matrices to (scaled) orthogonal matrices, ensuring
        all singular values are equal. LSUV extends this with a data-dependent calibration pass.
      </p>

      <DefinitionBlock title="Orthogonal Initialization">
        <BlockMath math="W = Q \quad \text{where } Q^TQ = I \text{ (from QR decomposition of random matrix)}" />
        <p className="mt-2">
          All singular values of <InlineMath math="W" /> are exactly 1, so
          <InlineMath math="\|Wx\| = \|x\|" /> — the transform is norm-preserving. Optionally
          scale by a gain factor <InlineMath math="g" /> for specific activations.
        </p>
      </DefinitionBlock>

      <SingularValueViz />

      <TheoremBlock title="Dynamical Isometry" id="dynamical-isometry">
        <p>
          A network satisfies dynamical isometry when the singular values of the input-output
          Jacobian are concentrated near 1:
        </p>
        <BlockMath math="\sigma_i\!\left(\frac{\partial f(x)}{\partial x}\right) \approx 1, \quad \forall i" />
        <p>
          Orthogonal initialization achieves this for linear networks. For non-linear networks,
          careful combination with activation choice is needed.
        </p>
      </TheoremBlock>

      <DefinitionBlock title="LSUV (Layer-Sequential Unit-Variance)">
        <p>A data-dependent initialization procedure:</p>
        <BlockMath math="\text{1. Initialize } W_l \text{ orthogonally}" />
        <BlockMath math="\text{2. Forward pass a mini-batch through layers 1..}l" />
        <BlockMath math="\text{3. Scale } W_l \leftarrow W_l / \text{std}(\text{output}_l) \text{ until Var} \approx 1" />
        <p className="mt-2">Repeat for each layer sequentially. This accounts for actual non-linearities.</p>
      </DefinitionBlock>

      <ExampleBlock title="When Orthogonal Init Helps">
        <p>
          Orthogonal init is especially beneficial for RNNs and very deep networks (50+ layers)
          where repeated matrix multiplication causes exponential growth or decay with non-isometric
          weights. For standard Transformers and CNNs with normalization, the benefit is smaller.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Orthogonal & LSUV in PyTorch"
        code={`import torch
import torch.nn as nn

# Orthogonal initialization
layer = nn.Linear(256, 256, bias=False)
nn.init.orthogonal_(layer.weight, gain=1.0)

# Verify: singular values should all be ~1
U, S, V = torch.linalg.svd(layer.weight.data)
print(f"Singular values: min={S.min():.4f}, max={S.max():.4f}")

# LSUV-style initialization
def lsuv_init(model, data, tol=0.1, max_iter=10):
    model.eval()
    hooks, outputs = [], {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            def hook_fn(m, inp, out, n=name):
                outputs[n] = out.detach()
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                for _ in range(max_iter):
                    model(data)
                    std = outputs[name].std()
                    if abs(std - 1.0) < tol: break
                    module.weight.data /= std

    for h in hooks: h.remove()

model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
lsuv_init(model, torch.randn(64, 128))
print("LSUV initialization complete")`}
      />

      <WarningBlock title="Orthogonal Init Requires Square-ish Matrices">
        <p>
          For non-square weight matrices, the orthogonal initialization produces a semi-orthogonal
          matrix. When <InlineMath math="n_{\text{out}} \gg n_{\text{in}}" /> or vice versa, the
          singular value guarantee weakens. In such cases, He init may be more robust.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="RNN-Specific Benefit">
        <p>
          For RNN hidden-to-hidden weights, orthogonal init is nearly essential. It prevents
          the vanishing/exploding gradient problem inherent in repeated matrix multiplication
          across time steps. LSTMs and GRUs partially address this architecturally.
        </p>
      </NoteBlock>
    </div>
  )
}

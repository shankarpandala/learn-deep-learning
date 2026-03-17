import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function WassersteinViz() {
  const [overlap, setOverlap] = useState(2.0)
  const W = 360, H = 120, cx = W / 2

  const p1 = cx - overlap * 30
  const p2 = cx + overlap * 30
  const jsDiv = overlap > 0.5 ? Math.log(2).toFixed(3) : '0.000'
  const wDist = (overlap * 0.5).toFixed(3)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Wasserstein vs JS Divergence</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Distribution separation: {overlap.toFixed(1)}
        <input type="range" min={0} max={4} step={0.1} value={overlap} onChange={e => setOverlap(parseFloat(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <ellipse cx={p1} cy={H / 2} rx={40} ry={30} fill="#8b5cf6" opacity={0.3} />
        <ellipse cx={p2} cy={H / 2} rx={40} ry={30} fill="#f97316" opacity={0.3} />
        <text x={p1} y={H / 2 + 4} textAnchor="middle" className="text-[10px] fill-violet-700">P_r</text>
        <text x={p2} y={H / 2 + 4} textAnchor="middle" className="text-[10px] fill-orange-700">P_g</text>
      </svg>
      <div className="flex justify-center gap-6 text-xs text-gray-600 mt-1">
        <span>JS: {jsDiv} {overlap > 0.5 ? '(saturated)' : ''}</span>
        <span className="text-violet-600 font-semibold">Wasserstein: {wDist} (smooth gradient)</span>
      </div>
    </div>
  )
}

export default function DCGANWGAN() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        DCGAN introduced architectural guidelines for stable convolutional GANs, while WGAN replaced
        the JS divergence with the Wasserstein distance, providing meaningful gradients even when
        distributions do not overlap.
      </p>

      <DefinitionBlock title="DCGAN Architecture Guidelines">
        <p>Key design principles for stable convolutional GANs:</p>
        <ul className="list-disc ml-5 mt-2 space-y-1">
          <li>Replace pooling with strided convolutions (D) and transposed convolutions (G)</li>
          <li>Use batch normalization in both G and D (except D input and G output)</li>
          <li>Remove fully connected layers (use global average pooling)</li>
          <li>G uses ReLU (output: Tanh); D uses LeakyReLU throughout</li>
        </ul>
      </DefinitionBlock>

      <DefinitionBlock title="Wasserstein Distance (Earth Mover's)">
        <BlockMath math="W(p_r, p_g) = \inf_{\gamma \in \Pi(p_r, p_g)} \mathbb{E}_{(x,y) \sim \gamma}\left[\|x - y\|\right]" />
        <p className="mt-2">The WGAN critic (not a discriminator) objective with Kantorovich-Rubinstein duality:</p>
        <BlockMath math="\max_{\|D\|_L \leq 1} \; \mathbb{E}_{x \sim p_r}[D(x)] - \mathbb{E}_{x \sim p_g}[D(x)]" />
      </DefinitionBlock>

      <WassersteinViz />

      <TheoremBlock title="Gradient Penalty (WGAN-GP)" id="wgan-gp">
        <p>Instead of weight clipping, WGAN-GP enforces the Lipschitz constraint via a gradient penalty:</p>
        <BlockMath math="\mathcal{L}_{\text{GP}} = \lambda \, \mathbb{E}_{\hat{x}}\left[\left(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1\right)^2\right]" />
        <p className="mt-2">
          where <InlineMath math="\hat{x} = \epsilon x + (1 - \epsilon) G(z)" /> with <InlineMath math="\epsilon \sim U(0,1)" />.
        </p>
      </TheoremBlock>

      <PythonCode
        title="WGAN-GP Training Step"
        code={`import torch
import torch.nn as nn
import torch.autograd as autograd

def gradient_penalty(critic, real, fake, device='cpu', lam=10):
    B = real.size(0)
    eps = torch.rand(B, 1, 1, 1, device=device).expand_as(real)
    interpolated = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_inter = critic(interpolated)
    grads = autograd.grad(
        outputs=d_inter, inputs=interpolated,
        grad_outputs=torch.ones_like(d_inter),
        create_graph=True, retain_graph=True,
    )[0]
    grads = grads.view(B, -1)
    gp = lam * ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# Training loop sketch (single step)
# critic_loss = D(fake).mean() - D(real).mean() + gradient_penalty(D, real, fake)
# gen_loss = -D(G(z)).mean()
print("WGAN-GP: Critic maximizes E[D(real)] - E[D(fake)]")
print("Generator minimizes -E[D(G(z))]")`}
      />

      <WarningBlock title="Do Not Use Batch Norm with WGAN-GP">
        <p>
          Batch normalization creates dependencies between samples in a mini-batch, which
          violates the per-sample gradient penalty assumption. Use layer normalization or
          instance normalization in the critic instead.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Spectral Normalization (SN-GAN)">
        <p>
          An alternative to gradient penalty: normalize each weight matrix <InlineMath math="W" /> by
          its spectral norm <InlineMath math="\sigma(W)" /> to enforce Lipschitz continuity.
          Simpler and more computationally efficient than WGAN-GP.
        </p>
      </NoteBlock>
    </div>
  )
}

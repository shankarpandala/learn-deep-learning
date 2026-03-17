import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function GANTrainingViz() {
  const [step, setStep] = useState(0)
  const dLoss = Math.max(0.1, 0.7 - step * 0.02 + Math.sin(step * 0.3) * 0.1)
  const gLoss = Math.max(0.3, 2.5 - step * 0.05 + Math.cos(step * 0.2) * 0.15)
  const W = 380, H = 140, pad = 30

  const dPoints = Array.from({ length: step + 1 }, (_, i) => {
    const v = Math.max(0.1, 0.7 - i * 0.02 + Math.sin(i * 0.3) * 0.1)
    return `${pad + i * (W - 2 * pad) / 40},${H - pad - v * (H - 2 * pad) / 3}`
  }).join(' ')
  const gPoints = Array.from({ length: step + 1 }, (_, i) => {
    const v = Math.max(0.3, 2.5 - i * 0.05 + Math.cos(i * 0.2) * 0.15)
    return `${pad + i * (W - 2 * pad) / 40},${H - pad - v * (H - 2 * pad) / 3}`
  }).join(' ')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">GAN Training Dynamics</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Step: {step}
        <input type="range" min={0} max={40} step={1} value={step} onChange={e => setStep(parseInt(e.target.value))} className="w-40 accent-violet-500" />
        <span className="text-xs">D loss: {dLoss.toFixed(2)} | G loss: {gLoss.toFixed(2)}</span>
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#d1d5db" strokeWidth={0.5} />
        {step > 0 && <polyline points={dPoints} fill="none" stroke="#8b5cf6" strokeWidth={2} />}
        {step > 0 && <polyline points={gPoints} fill="none" stroke="#f97316" strokeWidth={2} />}
      </svg>
      <div className="flex justify-center gap-4 text-xs mt-1">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-violet-500" /> D loss</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-orange-500" /> G loss</span>
      </div>
    </div>
  )
}

export default function GANBasics() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Generative Adversarial Networks pit two networks against each other: a generator that creates
        fake data and a discriminator that distinguishes real from fake. This adversarial game drives
        both networks to improve, ultimately producing realistic samples.
      </p>

      <DefinitionBlock title="GAN Minimax Objective">
        <BlockMath math="\min_G \max_D \; \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]" />
        <p className="mt-2">
          <InlineMath math="G" /> maps noise <InlineMath math="\mathbf{z}" /> to data space;
          <InlineMath math="D" /> outputs the probability that its input is real.
        </p>
      </DefinitionBlock>

      <TheoremBlock title="Optimal Discriminator" id="optimal-discriminator">
        <p>For a fixed generator <InlineMath math="G" />, the optimal discriminator is:</p>
        <BlockMath math="D^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_G(\mathbf{x})}" />
        <p className="mt-2">
          Substituting back, the generator minimizes the Jensen-Shannon divergence
          <InlineMath math="\text{JSD}(p_{\text{data}} \| p_G)" />.
        </p>
      </TheoremBlock>

      <GANTrainingViz />

      <PythonCode
        title="Simple GAN in PyTorch"
        code={`import torch
import torch.nn as nn

latent_dim = 64

G = nn.Sequential(
    nn.Linear(latent_dim, 256), nn.ReLU(),
    nn.Linear(256, 512), nn.ReLU(),
    nn.Linear(512, 784), nn.Tanh(),  # output in [-1, 1]
)
D = nn.Sequential(
    nn.Linear(784, 512), nn.LeakyReLU(0.2),
    nn.Linear(512, 256), nn.LeakyReLU(0.2),
    nn.Linear(256, 1), nn.Sigmoid(),
)

criterion = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Single training step
real = torch.randn(32, 784)  # placeholder for real data
z = torch.randn(32, latent_dim)
fake = G(z)

# Train D
d_real = D(real)
d_fake = D(fake.detach())
d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))

# Train G (non-saturating loss)
g_loss = criterion(D(fake), torch.ones_like(d_fake))
print(f"D loss: {d_loss.item():.3f}, G loss: {g_loss.item():.3f}")`}
      />

      <ExampleBlock title="Non-Saturating Loss">
        <p>
          In practice, instead of <InlineMath math="\log(1 - D(G(\mathbf{z})))" />, the generator
          maximizes <InlineMath math="\log D(G(\mathbf{z}))" />. This provides stronger gradients
          early in training when <InlineMath math="D" /> easily rejects fakes.
        </p>
      </ExampleBlock>

      <NoteBlock type="note" title="Training Instability">
        <p>
          GANs are notoriously hard to train. Common issues include mode collapse (generator produces
          limited variety), training oscillation, and vanishing gradients when the discriminator
          becomes too strong. Techniques like spectral normalization, gradient penalty, and careful
          learning rate scheduling help stabilize training.
        </p>
      </NoteBlock>
    </div>
  )
}

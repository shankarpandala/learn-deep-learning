import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ReparamTrickViz() {
  const [mu, setMu] = useState(0)
  const [logvar, setLogvar] = useState(0)
  const sigma = Math.exp(0.5 * logvar)
  const samples = Array.from({ length: 20 }, (_, i) => {
    const eps = -2 + i * 0.21
    return mu + sigma * eps
  })

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Reparameterization Trick</h3>
      <div className="flex gap-4 mb-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          mu: {mu.toFixed(1)}
          <input type="range" min={-3} max={3} step={0.1} value={mu} onChange={e => setMu(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          log_var: {logvar.toFixed(1)} (sigma={sigma.toFixed(2)})
          <input type="range" min={-2} max={2} step={0.1} value={logvar} onChange={e => setLogvar(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <svg width={400} height={60} className="mx-auto block">
        <line x1={0} y1={30} x2={400} y2={30} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={200 + mu * 30} y1={10} x2={200 + mu * 30} y2={50} stroke="#8b5cf6" strokeWidth={2} />
        {samples.map((s, i) => (
          <circle key={i} cx={200 + s * 30} cy={30} r={3} fill="#8b5cf6" opacity={0.5} />
        ))}
        <text x={200 + mu * 30} y={55} textAnchor="middle" className="text-[10px] fill-violet-500">mu</text>
      </svg>
      <p className="text-xs text-center text-gray-500 mt-1">
        z = mu + sigma * epsilon, where epsilon ~ N(0,1)
      </p>
    </div>
  )
}

export default function VAE() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Variational Autoencoders (VAEs) turn autoencoders into proper generative models by imposing
        a probabilistic structure on the latent space, enabling sampling of new data points.
      </p>

      <DefinitionBlock title="VAE Objective (ELBO)">
        <p>The VAE maximizes the Evidence Lower Bound:</p>
        <BlockMath math="\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\theta(\mathbf{z}|\mathbf{x})}\left[\log p_\phi(\mathbf{x}|\mathbf{z})\right] - D_{\text{KL}}\left(q_\theta(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})\right)" />
        <p className="mt-2">
          The first term is reconstruction quality; the second regularizes the posterior
          <InlineMath math="q_\theta(\mathbf{z}|\mathbf{x})" /> toward the prior <InlineMath math="p(\mathbf{z}) = \mathcal{N}(0, I)" />.
        </p>
      </DefinitionBlock>

      <TheoremBlock title="KL Divergence (Gaussian)" id="kl-gaussian">
        <p>For a diagonal Gaussian encoder <InlineMath math="q(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))" />:</p>
        <BlockMath math="D_{\text{KL}} = -\frac{1}{2}\sum_{j=1}^{d}\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)" />
      </TheoremBlock>

      <ReparamTrickViz />

      <ExampleBlock title="Why Reparameterization?">
        <p>
          We cannot backpropagate through a stochastic sampling step. The reparameterization trick
          rewrites <InlineMath math="\mathbf{z} \sim q_\theta(\mathbf{z}|\mathbf{x})" /> as:
        </p>
        <BlockMath math="\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)" />
        <p>Now gradients flow through <InlineMath math="\boldsymbol{\mu}" /> and <InlineMath math="\boldsymbol{\sigma}" /> directly.</p>
      </ExampleBlock>

      <PythonCode
        title="VAE in PyTorch"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 256)
        self.fc4 = nn.Linear(256, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(x_hat, x, mu, logvar):
    recon = F.binary_cross_entropy(x_hat, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl

model = VAE()
x = torch.sigmoid(torch.randn(8, 784))
x_hat, mu, logvar = model(x)
loss = vae_loss(x_hat, x, mu, logvar)
print(f"ELBO loss: {loss.item():.1f}")`}
      />

      <NoteBlock type="note" title="Posterior Collapse">
        <p>
          A common failure mode where the decoder ignores <InlineMath math="\mathbf{z}" /> and the encoder
          collapses to the prior. Solutions include KL annealing (warming up the KL weight from 0 to 1),
          free bits (minimum KL per dimension), and cyclical schedules.
        </p>
      </NoteBlock>
    </div>
  )
}

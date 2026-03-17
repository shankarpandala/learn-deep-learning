import{j as e,r as p}from"./vendor-DpISuAX6.js";import{r as t}from"./vendor-katex-CbWCYdth.js";import{D as f,E as b,P as u,N as g,T as _,W as j}from"./subject-01-foundations-D0A1VJsr.js";function N(){const[a,h]=p.useState(8),r=64,i=(a/r*100).toFixed(1),l=[r,32,a,32,r],n=160;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Autoencoder Bottleneck"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Latent dim: ",a,e.jsx("input",{type:"range",min:2,max:32,step:1,value:a,onChange:o=>h(parseInt(o.target.value)),className:"w-40 accent-violet-500"}),e.jsxs("span",{className:"text-xs",children:["(",i,"% of input)"]})]}),e.jsxs("svg",{width:400,height:n+30,className:"mx-auto block",children:[l.map((o,d)=>{const c=40+d*80,m=o/r*n,x=(n-m)/2+10,y=d===2?"#8b5cf6":"#a78bfa";return e.jsxs("g",{children:[e.jsx("rect",{x:c,y:x,width:30,height:m,rx:4,fill:y,opacity:.8}),e.jsx("text",{x:c+15,y:n+25,textAnchor:"middle",className:"text-xs fill-gray-500",children:o}),d<l.length-1&&e.jsx("line",{x1:c+30,y1:n/2+10,x2:c+80,y2:n/2+10,stroke:"#d1d5db",strokeWidth:1})]},d)}),e.jsx("text",{x:55,y:n+25,textAnchor:"middle",className:"text-[10px] fill-gray-400",children:"input"}),e.jsx("text",{x:215,y:n+25,textAnchor:"middle",className:"text-[10px] fill-violet-500 font-bold",children:"z"})]})]})}function k(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"An autoencoder learns to compress input data into a lower-dimensional latent representation and then reconstruct it. The bottleneck forces the network to discover meaningful structure in the data."}),e.jsxs(f,{title:"Autoencoder",children:[e.jsxs("p",{children:["An autoencoder consists of an encoder ",e.jsx(t.InlineMath,{math:"f_\\theta"})," and decoder ",e.jsx(t.InlineMath,{math:"g_\\phi"}),":"]}),e.jsx(t.BlockMath,{math:"\\mathbf{z} = f_\\theta(\\mathbf{x}), \\quad \\hat{\\mathbf{x}} = g_\\phi(\\mathbf{z})"}),e.jsx("p",{className:"mt-2",children:"Training minimizes reconstruction loss:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L}(\\theta, \\phi) = \\| \\mathbf{x} - g_\\phi(f_\\theta(\\mathbf{x})) \\|^2"})]}),e.jsx(N,{}),e.jsxs(b,{title:"Dimensionality Reduction vs PCA",children:[e.jsx("p",{children:"A single-layer linear autoencoder with MSE loss learns the same subspace as PCA. However, deep nonlinear autoencoders capture manifolds that PCA cannot:"}),e.jsx(t.BlockMath,{math:"\\text{PCA: } \\mathbf{z} = W^\\top \\mathbf{x}, \\quad \\text{AE: } \\mathbf{z} = f_\\theta(\\mathbf{x})"})]}),e.jsx(u,{title:"Simple Autoencoder in PyTorch",code:`import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),  # pixel values in [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

model = Autoencoder(latent_dim=16)
x = torch.randn(8, 784)
x_hat, z = model(x)
print(f"Input: {x.shape}, Latent: {z.shape}, Recon: {x_hat.shape}")
loss = nn.MSELoss()(x_hat, torch.sigmoid(x))
print(f"Reconstruction loss: {loss.item():.4f}")`}),e.jsx(g,{type:"note",title:"Denoising Autoencoders",children:e.jsxs("p",{children:["A ",e.jsx("strong",{children:"denoising autoencoder"})," (DAE) corrupts the input with noise and trains the network to reconstruct the clean version. This prevents the autoencoder from learning the identity function and encourages robust feature extraction: ",e.jsx(t.InlineMath,{math:"\\mathcal{L} = \\|\\mathbf{x} - g_\\phi(f_\\theta(\\tilde{\\mathbf{x}}))\\|^2"}),"."]})}),e.jsx(g,{type:"note",title:"Applications",children:e.jsx("p",{children:"Autoencoders power anomaly detection (high reconstruction error signals anomalies), data compression, feature learning for downstream tasks, and serve as building blocks for variational autoencoders and diffusion models."})})]})}const se=Object.freeze(Object.defineProperty({__proto__:null,default:k},Symbol.toStringTag,{value:"Module"}));function M(){const[a,h]=p.useState(0),[r,i]=p.useState(0),l=Math.exp(.5*r),n=Array.from({length:20},(s,o)=>{const d=-2+o*.21;return a+l*d});return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Reparameterization Trick"}),e.jsxs("div",{className:"flex gap-4 mb-3 flex-wrap",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["mu: ",a.toFixed(1),e.jsx("input",{type:"range",min:-3,max:3,step:.1,value:a,onChange:s=>h(parseFloat(s.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["log_var: ",r.toFixed(1)," (sigma=",l.toFixed(2),")",e.jsx("input",{type:"range",min:-2,max:2,step:.1,value:r,onChange:s=>i(parseFloat(s.target.value)),className:"w-28 accent-violet-500"})]})]}),e.jsxs("svg",{width:400,height:60,className:"mx-auto block",children:[e.jsx("line",{x1:0,y1:30,x2:400,y2:30,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("line",{x1:200+a*30,y1:10,x2:200+a*30,y2:50,stroke:"#8b5cf6",strokeWidth:2}),n.map((s,o)=>e.jsx("circle",{cx:200+s*30,cy:30,r:3,fill:"#8b5cf6",opacity:.5},o)),e.jsx("text",{x:200+a*30,y:55,textAnchor:"middle",className:"text-[10px] fill-violet-500",children:"mu"})]}),e.jsx("p",{className:"text-xs text-center text-gray-500 mt-1",children:"z = mu + sigma * epsilon, where epsilon ~ N(0,1)"})]})}function z(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Variational Autoencoders (VAEs) turn autoencoders into proper generative models by imposing a probabilistic structure on the latent space, enabling sampling of new data points."}),e.jsxs(f,{title:"VAE Objective (ELBO)",children:[e.jsx("p",{children:"The VAE maximizes the Evidence Lower Bound:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_{\\text{ELBO}} = \\mathbb{E}_{q_\\theta(\\mathbf{z}|\\mathbf{x})}\\left[\\log p_\\phi(\\mathbf{x}|\\mathbf{z})\\right] - D_{\\text{KL}}\\left(q_\\theta(\\mathbf{z}|\\mathbf{x}) \\| p(\\mathbf{z})\\right)"}),e.jsxs("p",{className:"mt-2",children:["The first term is reconstruction quality; the second regularizes the posterior",e.jsx(t.InlineMath,{math:"q_\\theta(\\mathbf{z}|\\mathbf{x})"})," toward the prior ",e.jsx(t.InlineMath,{math:"p(\\mathbf{z}) = \\mathcal{N}(0, I)"}),"."]})]}),e.jsxs(_,{title:"KL Divergence (Gaussian)",id:"kl-gaussian",children:[e.jsxs("p",{children:["For a diagonal Gaussian encoder ",e.jsx(t.InlineMath,{math:"q(\\mathbf{z}|\\mathbf{x}) = \\mathcal{N}(\\boldsymbol{\\mu}, \\text{diag}(\\boldsymbol{\\sigma}^2))"}),":"]}),e.jsx(t.BlockMath,{math:"D_{\\text{KL}} = -\\frac{1}{2}\\sum_{j=1}^{d}\\left(1 + \\log\\sigma_j^2 - \\mu_j^2 - \\sigma_j^2\\right)"})]}),e.jsx(M,{}),e.jsxs(b,{title:"Why Reparameterization?",children:[e.jsxs("p",{children:["We cannot backpropagate through a stochastic sampling step. The reparameterization trick rewrites ",e.jsx(t.InlineMath,{math:"\\mathbf{z} \\sim q_\\theta(\\mathbf{z}|\\mathbf{x})"})," as:"]}),e.jsx(t.BlockMath,{math:"\\mathbf{z} = \\boldsymbol{\\mu} + \\boldsymbol{\\sigma} \\odot \\boldsymbol{\\epsilon}, \\quad \\boldsymbol{\\epsilon} \\sim \\mathcal{N}(0, I)"}),e.jsxs("p",{children:["Now gradients flow through ",e.jsx(t.InlineMath,{math:"\\boldsymbol{\\mu}"})," and ",e.jsx(t.InlineMath,{math:"\\boldsymbol{\\sigma}"})," directly."]})]}),e.jsx(u,{title:"VAE in PyTorch",code:`import torch
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
print(f"ELBO loss: {loss.item():.1f}")`}),e.jsx(g,{type:"note",title:"Posterior Collapse",children:e.jsxs("p",{children:["A common failure mode where the decoder ignores ",e.jsx(t.InlineMath,{math:"\\mathbf{z}"})," and the encoder collapses to the prior. Solutions include KL annealing (warming up the KL weight from 0 to 1), free bits (minimum KL per dimension), and cyclical schedules."]})})]})}const ne=Object.freeze(Object.defineProperty({__proto__:null,default:z},Symbol.toStringTag,{value:"Module"}));function S(){const[a,h]=p.useState(1),r=1,i=a;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Beta-VAE Trade-off"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["beta = ",a.toFixed(1),e.jsx("input",{type:"range",min:.1,max:10,step:.1,value:a,onChange:l=>h(parseFloat(l.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("div",{className:"flex gap-4 items-end h-32",children:[e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("div",{className:"w-16 bg-violet-400 rounded-t",style:{height:`${r/Math.max(r,i)*100}px`}}),e.jsxs("span",{className:"text-xs text-gray-500 mt-1",children:["Recon (",r.toFixed(1),")"]})]}),e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("div",{className:"w-16 bg-violet-700 rounded-t",style:{height:`${i/Math.max(r,i)*100}px`}}),e.jsxs("span",{className:"text-xs text-gray-500 mt-1",children:["KL (",i.toFixed(1),")"]})]})]}),e.jsx("p",{className:"text-xs text-gray-500 mt-2",children:a<1?"Low beta: better reconstruction, less disentanglement":a===1?"Standard VAE (beta=1)":"High beta: more disentanglement, blurrier reconstructions"})]})}function A(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Several important VAE variants address limitations of the standard model. Beta-VAE encourages disentangled representations, while VQ-VAE replaces continuous latents with discrete codebooks."}),e.jsxs(f,{title:"Beta-VAE",children:[e.jsxs("p",{children:["Beta-VAE adds a hyperparameter ",e.jsx(t.InlineMath,{math:"\\beta"})," to control the KL weight:"]}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_{\\beta\\text{-VAE}} = \\mathbb{E}_{q}\\left[\\log p(\\mathbf{x}|\\mathbf{z})\\right] - \\beta \\cdot D_{\\text{KL}}\\left(q(\\mathbf{z}|\\mathbf{x}) \\| p(\\mathbf{z})\\right)"}),e.jsxs("p",{className:"mt-2",children:["When ",e.jsx(t.InlineMath,{math:"\\beta > 1"}),", the model is pressured to find a more efficient, disentangled encoding where each latent dimension captures an independent factor of variation."]})]}),e.jsx(S,{}),e.jsxs(f,{title:"VQ-VAE (Vector Quantized VAE)",children:[e.jsxs("p",{children:["VQ-VAE uses a discrete codebook ",e.jsx(t.InlineMath,{math:"\\mathbf{e} \\in \\mathbb{R}^{K \\times D}"}),". The encoder output is quantized:"]}),e.jsx(t.BlockMath,{math:"z_q = \\mathbf{e}_k, \\quad k = \\arg\\min_j \\| f_\\theta(\\mathbf{x}) - \\mathbf{e}_j \\|"}),e.jsx("p",{className:"mt-2",children:"Training loss combines reconstruction, codebook, and commitment terms:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = \\|\\mathbf{x} - \\hat{\\mathbf{x}}\\|^2 + \\|\\text{sg}[z_e] - e\\|^2 + \\beta\\|z_e - \\text{sg}[e]\\|^2"})]}),e.jsx(u,{title:"VQ-VAE Quantization Layer",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, z_e):
        # z_e: (B, D) encoder output
        distances = torch.cdist(z_e.unsqueeze(0), self.codebook.weight.unsqueeze(0)).squeeze(0)
        indices = distances.argmin(dim=-1)
        z_q = self.codebook(indices)

        # Straight-through estimator: copy gradients from z_q to z_e
        z_q_st = z_e + (z_q - z_e).detach()

        # Losses
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        return z_q_st, vq_loss, indices

vq = VectorQuantizer(num_embeddings=512, embedding_dim=64)
z_e = torch.randn(8, 64)
z_q, loss, idx = vq(z_e)
print(f"Codebook indices: {idx[:4].tolist()}")
print(f"VQ loss: {loss.item():.4f}")`}),e.jsx(b,{title:"VQ-VAE-2 for High-Res Images",children:e.jsx("p",{children:"VQ-VAE-2 uses a hierarchical codebook with two levels: a top-level captures global structure while a bottom-level captures fine details. Combined with a powerful autoregressive prior (PixelSNAIL), it generates diverse, high-fidelity images."})}),e.jsx(j,{title:"Codebook Collapse",children:e.jsx("p",{children:"A common issue where only a fraction of codebook entries are used. Mitigation strategies include exponential moving average updates, codebook reset for dead entries, and entropy-based regularization to encourage uniform codebook usage."})}),e.jsx(g,{type:"note",title:"Impact on Modern Generative AI",children:e.jsx("p",{children:"VQ-VAE forms the backbone of many modern systems: DALL-E uses a VQ-VAE to tokenize images, and latent diffusion models (Stable Diffusion) use a VAE encoder to compress images into a latent space where diffusion operates more efficiently."})})]})}const ie=Object.freeze(Object.defineProperty({__proto__:null,default:A},Symbol.toStringTag,{value:"Module"}));function I(){const[a,h]=p.useState(0),r=Math.max(.1,.7-a*.02+Math.sin(a*.3)*.1),i=Math.max(.3,2.5-a*.05+Math.cos(a*.2)*.15),l=380,n=140,s=30,o=Array.from({length:a+1},(c,m)=>{const x=Math.max(.1,.7-m*.02+Math.sin(m*.3)*.1);return`${s+m*(l-2*s)/40},${n-s-x*(n-2*s)/3}`}).join(" "),d=Array.from({length:a+1},(c,m)=>{const x=Math.max(.3,2.5-m*.05+Math.cos(m*.2)*.15);return`${s+m*(l-2*s)/40},${n-s-x*(n-2*s)/3}`}).join(" ");return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"GAN Training Dynamics"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Step: ",a,e.jsx("input",{type:"range",min:0,max:40,step:1,value:a,onChange:c=>h(parseInt(c.target.value)),className:"w-40 accent-violet-500"}),e.jsxs("span",{className:"text-xs",children:["D loss: ",r.toFixed(2)," | G loss: ",i.toFixed(2)]})]}),e.jsxs("svg",{width:l,height:n,className:"mx-auto block",children:[e.jsx("line",{x1:s,y1:n-s,x2:l-s,y2:n-s,stroke:"#d1d5db",strokeWidth:.5}),a>0&&e.jsx("polyline",{points:o,fill:"none",stroke:"#8b5cf6",strokeWidth:2}),a>0&&e.jsx("polyline",{points:d,fill:"none",stroke:"#f97316",strokeWidth:2})]}),e.jsxs("div",{className:"flex justify-center gap-4 text-xs mt-1",children:[e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-violet-500"})," D loss"]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-orange-500"})," G loss"]})]})]})}function D(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Generative Adversarial Networks pit two networks against each other: a generator that creates fake data and a discriminator that distinguishes real from fake. This adversarial game drives both networks to improve, ultimately producing realistic samples."}),e.jsxs(f,{title:"GAN Minimax Objective",children:[e.jsx(t.BlockMath,{math:"\\min_G \\max_D \\; \\mathbb{E}_{\\mathbf{x} \\sim p_{\\text{data}}}[\\log D(\\mathbf{x})] + \\mathbb{E}_{\\mathbf{z} \\sim p_z}[\\log(1 - D(G(\\mathbf{z})))]"}),e.jsxs("p",{className:"mt-2",children:[e.jsx(t.InlineMath,{math:"G"})," maps noise ",e.jsx(t.InlineMath,{math:"\\mathbf{z}"})," to data space;",e.jsx(t.InlineMath,{math:"D"})," outputs the probability that its input is real."]})]}),e.jsxs(_,{title:"Optimal Discriminator",id:"optimal-discriminator",children:[e.jsxs("p",{children:["For a fixed generator ",e.jsx(t.InlineMath,{math:"G"}),", the optimal discriminator is:"]}),e.jsx(t.BlockMath,{math:"D^*(\\mathbf{x}) = \\frac{p_{\\text{data}}(\\mathbf{x})}{p_{\\text{data}}(\\mathbf{x}) + p_G(\\mathbf{x})}"}),e.jsxs("p",{className:"mt-2",children:["Substituting back, the generator minimizes the Jensen-Shannon divergence",e.jsx(t.InlineMath,{math:"\\text{JSD}(p_{\\text{data}} \\| p_G)"}),"."]})]}),e.jsx(I,{}),e.jsx(u,{title:"Simple GAN in PyTorch",code:`import torch
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
print(f"D loss: {d_loss.item():.3f}, G loss: {g_loss.item():.3f}")`}),e.jsx(b,{title:"Non-Saturating Loss",children:e.jsxs("p",{children:["In practice, instead of ",e.jsx(t.InlineMath,{math:"\\log(1 - D(G(\\mathbf{z})))"}),", the generator maximizes ",e.jsx(t.InlineMath,{math:"\\log D(G(\\mathbf{z}))"}),". This provides stronger gradients early in training when ",e.jsx(t.InlineMath,{math:"D"})," easily rejects fakes."]})}),e.jsx(g,{type:"note",title:"Training Instability",children:e.jsx("p",{children:"GANs are notoriously hard to train. Common issues include mode collapse (generator produces limited variety), training oscillation, and vanishing gradients when the discriminator becomes too strong. Techniques like spectral normalization, gradient penalty, and careful learning rate scheduling help stabilize training."})})]})}const re=Object.freeze(Object.defineProperty({__proto__:null,default:D},Symbol.toStringTag,{value:"Module"}));function L(){const[a,h]=p.useState(2),r=360,i=120,l=r/2,n=l-a*30,s=l+a*30,o=a>.5?Math.log(2).toFixed(3):"0.000",d=(a*.5).toFixed(3);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Wasserstein vs JS Divergence"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Distribution separation: ",a.toFixed(1),e.jsx("input",{type:"range",min:0,max:4,step:.1,value:a,onChange:c=>h(parseFloat(c.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("svg",{width:r,height:i,className:"mx-auto block",children:[e.jsx("ellipse",{cx:n,cy:i/2,rx:40,ry:30,fill:"#8b5cf6",opacity:.3}),e.jsx("ellipse",{cx:s,cy:i/2,rx:40,ry:30,fill:"#f97316",opacity:.3}),e.jsx("text",{x:n,y:i/2+4,textAnchor:"middle",className:"text-[10px] fill-violet-700",children:"P_r"}),e.jsx("text",{x:s,y:i/2+4,textAnchor:"middle",className:"text-[10px] fill-orange-700",children:"P_g"})]}),e.jsxs("div",{className:"flex justify-center gap-6 text-xs text-gray-600 mt-1",children:[e.jsxs("span",{children:["JS: ",o," ",a>.5?"(saturated)":""]}),e.jsxs("span",{className:"text-violet-600 font-semibold",children:["Wasserstein: ",d," (smooth gradient)"]})]})]})}function T(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"DCGAN introduced architectural guidelines for stable convolutional GANs, while WGAN replaced the JS divergence with the Wasserstein distance, providing meaningful gradients even when distributions do not overlap."}),e.jsxs(f,{title:"DCGAN Architecture Guidelines",children:[e.jsx("p",{children:"Key design principles for stable convolutional GANs:"}),e.jsxs("ul",{className:"list-disc ml-5 mt-2 space-y-1",children:[e.jsx("li",{children:"Replace pooling with strided convolutions (D) and transposed convolutions (G)"}),e.jsx("li",{children:"Use batch normalization in both G and D (except D input and G output)"}),e.jsx("li",{children:"Remove fully connected layers (use global average pooling)"}),e.jsx("li",{children:"G uses ReLU (output: Tanh); D uses LeakyReLU throughout"})]})]}),e.jsxs(f,{title:"Wasserstein Distance (Earth Mover's)",children:[e.jsx(t.BlockMath,{math:"W(p_r, p_g) = \\inf_{\\gamma \\in \\Pi(p_r, p_g)} \\mathbb{E}_{(x,y) \\sim \\gamma}\\left[\\|x - y\\|\\right]"}),e.jsx("p",{className:"mt-2",children:"The WGAN critic (not a discriminator) objective with Kantorovich-Rubinstein duality:"}),e.jsx(t.BlockMath,{math:"\\max_{\\|D\\|_L \\leq 1} \\; \\mathbb{E}_{x \\sim p_r}[D(x)] - \\mathbb{E}_{x \\sim p_g}[D(x)]"})]}),e.jsx(L,{}),e.jsxs(_,{title:"Gradient Penalty (WGAN-GP)",id:"wgan-gp",children:[e.jsx("p",{children:"Instead of weight clipping, WGAN-GP enforces the Lipschitz constraint via a gradient penalty:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_{\\text{GP}} = \\lambda \\, \\mathbb{E}_{\\hat{x}}\\left[\\left(\\|\\nabla_{\\hat{x}} D(\\hat{x})\\|_2 - 1\\right)^2\\right]"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"\\hat{x} = \\epsilon x + (1 - \\epsilon) G(z)"})," with ",e.jsx(t.InlineMath,{math:"\\epsilon \\sim U(0,1)"}),"."]})]}),e.jsx(u,{title:"WGAN-GP Training Step",code:`import torch
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
print("Generator minimizes -E[D(G(z))]")`}),e.jsx(j,{title:"Do Not Use Batch Norm with WGAN-GP",children:e.jsx("p",{children:"Batch normalization creates dependencies between samples in a mini-batch, which violates the per-sample gradient penalty assumption. Use layer normalization or instance normalization in the critic instead."})}),e.jsx(g,{type:"note",title:"Spectral Normalization (SN-GAN)",children:e.jsxs("p",{children:["An alternative to gradient penalty: normalize each weight matrix ",e.jsx(t.InlineMath,{math:"W"})," by its spectral norm ",e.jsx(t.InlineMath,{math:"\\sigma(W)"})," to enforce Lipschitz continuity. Simpler and more computationally efficient than WGAN-GP."]})})]})}const le=Object.freeze(Object.defineProperty({__proto__:null,default:T},Symbol.toStringTag,{value:"Module"}));function F(){const[a,h]=p.useState(4),r=["4x4","8x8","16x16","32x32","64x64","128x128","256x256","512x512","1024x1024"];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Style Mixing Crossover"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Crossover at layer: ",a,e.jsx("input",{type:"range",min:1,max:8,step:1,value:a,onChange:i=>h(parseInt(i.target.value)),className:"w-40 accent-violet-500"})]}),e.jsx("div",{className:"flex gap-1 items-end justify-center",children:r.map((i,l)=>{const n=l<a;return e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("div",{className:`rounded ${n?"bg-violet-500":"bg-orange-400"}`,style:{width:24,height:8+l*6}}),e.jsx("span",{className:"text-[8px] text-gray-400 mt-1",children:i})]},l)})}),e.jsxs("div",{className:"flex justify-center gap-4 text-xs mt-2",children:[e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-2 bg-violet-500 rounded"})," Style A (coarse)"]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-2 bg-orange-400 rounded"})," Style B (fine)"]})]}),e.jsx("p",{className:"text-xs text-gray-500 text-center mt-1",children:a<=3?"Low crossover: A controls pose/shape, B controls colors/details":a<=6?"Mid crossover: A controls structure, B controls fine features":"High crossover: A controls almost everything, B only affects finest details"})]})}function q(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"StyleGAN revolutionized high-resolution image synthesis by introducing a style-based generator architecture that provides unprecedented control over the generation process at different levels of detail."}),e.jsxs(f,{title:"StyleGAN Architecture",children:[e.jsx("p",{children:"Key innovations of the style-based generator:"}),e.jsxs("ul",{className:"list-disc ml-5 mt-2 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Mapping network"}),": 8-layer MLP transforms ",e.jsx(t.InlineMath,{math:"\\mathbf{z} \\in \\mathcal{Z}"})," to ",e.jsx(t.InlineMath,{math:"\\mathbf{w} \\in \\mathcal{W}"})]}),e.jsxs("li",{children:[e.jsx("strong",{children:"AdaIN"}),": Style ",e.jsx(t.InlineMath,{math:"\\mathbf{w}"})," modulates features via adaptive instance normalization"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Constant input"}),": Synthesis starts from a learned constant, not from ",e.jsx(t.InlineMath,{math:"\\mathbf{z}"})]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Noise injection"}),": Per-pixel noise adds stochastic variation (hair, freckles)"]})]}),e.jsx(t.BlockMath,{math:"\\text{AdaIN}(\\mathbf{x}_i, \\mathbf{y}) = y_{s,i}\\frac{\\mathbf{x}_i - \\mu(\\mathbf{x}_i)}{\\sigma(\\mathbf{x}_i)} + y_{b,i}"})]}),e.jsx(F,{}),e.jsx(b,{title:"Progressive Growing to StyleGAN3",children:e.jsxs("p",{children:[e.jsx("strong",{children:"ProGAN"}),": Progressively grows resolution during training (4x4 to 1024x1024).",e.jsx("strong",{children:"StyleGAN2"}),": Removes progressive growing, fixes water droplet artifacts with weight demodulation. ",e.jsx("strong",{children:"StyleGAN3"}),": Achieves alias-free generation with continuous equivariance to translation and rotation."]})}),e.jsx(u,{title:"Simplified StyleGAN Mapping Network + AdaIN",code:`import torch
import torch.nn as nn

class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.extend([nn.Linear(z_dim if i == 0 else w_dim, w_dim), nn.LeakyReLU(0.2)])
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class AdaIN(nn.Module):
    def __init__(self, channels, w_dim=512):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.style = nn.Linear(w_dim, channels * 2)

    def forward(self, x, w):
        style = self.style(w).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = style.chunk(2, dim=1)
        return gamma * self.norm(x) + beta

mapping = MappingNetwork()
adain = AdaIN(channels=256)

z = torch.randn(4, 512)
w = mapping(z)
feat = torch.randn(4, 256, 16, 16)
styled = adain(feat, w)
print(f"z: {z.shape} -> w: {w.shape}")
print(f"Styled features: {styled.shape}")`}),e.jsx(g,{type:"note",title:"The W and W+ Latent Spaces",children:e.jsxs("p",{children:["The intermediate latent space ",e.jsx(t.InlineMath,{math:"\\mathcal{W}"})," is less entangled than",e.jsx(t.InlineMath,{math:"\\mathcal{Z}"}),", enabling more meaningful interpolations. The extended",e.jsx(t.InlineMath,{math:"\\mathcal{W}^+"})," space uses different ",e.jsx(t.InlineMath,{math:"\\mathbf{w}"})," vectors per layer, enabling GAN inversion: finding latents that reconstruct real images for editing."]})})]})}const oe=Object.freeze(Object.defineProperty({__proto__:null,default:q},Symbol.toStringTag,{value:"Module"}));function G(){const[a,h]=p.useState(1.5),[r,i]=p.useState(.5),l=Math.log(Math.abs(a)).toFixed(3),n=380,s=120,o=Array.from({length:80},(c,m)=>{const x=-3+m*.075,y=Math.exp(-.5*x*x)/Math.sqrt(2*Math.PI);return{x,y}}),d=(c,m,x)=>`${x+c*25+90},${s-15-m*200}`;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Change of Variables"}),e.jsxs("div",{className:"flex gap-4 mb-3 flex-wrap",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["scale: ",a.toFixed(1),e.jsx("input",{type:"range",min:.3,max:3,step:.1,value:a,onChange:c=>h(parseFloat(c.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["shift: ",r.toFixed(1),e.jsx("input",{type:"range",min:-2,max:2,step:.1,value:r,onChange:c=>i(parseFloat(c.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("span",{className:"text-xs text-violet-600",children:["log|det J| = ",l]})]}),e.jsxs("svg",{width:n,height:s,className:"mx-auto block",children:[e.jsx("path",{d:o.map((c,m)=>`${m===0?"M":"L"}${d(c.x,c.y,0)}`).join(" "),fill:"none",stroke:"#8b5cf6",strokeWidth:2}),e.jsx("path",{d:o.map((c,m)=>{const x=c.x*a+r,y=c.y/Math.abs(a);return`${m===0?"M":"L"}${d(x,y,0)}`}).join(" "),fill:"none",stroke:"#f97316",strokeWidth:2}),e.jsx("text",{x:50,y:12,className:"text-[10px] fill-violet-500",children:"z ~ N(0,1)"}),e.jsx("text",{x:250,y:12,className:"text-[10px] fill-orange-500",children:"x = scale*z + shift"})]})]})}function E(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Normalizing flows transform a simple base distribution (e.g., Gaussian) into a complex data distribution through a chain of invertible transformations, allowing exact likelihood computation."}),e.jsxs(_,{title:"Change of Variables Formula",id:"change-of-variables",children:[e.jsxs("p",{children:["If ",e.jsx(t.InlineMath,{math:"f: \\mathbb{R}^d \\to \\mathbb{R}^d"})," is an invertible, differentiable map and ",e.jsx(t.InlineMath,{math:"\\mathbf{x} = f(\\mathbf{z})"}),":"]}),e.jsx(t.BlockMath,{math:"\\log p_X(\\mathbf{x}) = \\log p_Z(f^{-1}(\\mathbf{x})) - \\log\\left|\\det \\frac{\\partial f}{\\partial \\mathbf{z}}\\right|"}),e.jsx("p",{className:"mt-2",children:"The Jacobian determinant accounts for volume change under the transformation."})]}),e.jsxs(f,{title:"Normalizing Flow",children:[e.jsxs("p",{children:["A normalizing flow composes ",e.jsx(t.InlineMath,{math:"K"})," invertible transformations:"]}),e.jsx(t.BlockMath,{math:"\\mathbf{x} = f_K \\circ f_{K-1} \\circ \\cdots \\circ f_1(\\mathbf{z}_0), \\quad \\mathbf{z}_0 \\sim p_0(\\mathbf{z})"}),e.jsx("p",{className:"mt-2",children:"Log-likelihood decomposes as:"}),e.jsx(t.BlockMath,{math:"\\log p(\\mathbf{x}) = \\log p_0(\\mathbf{z}_0) - \\sum_{k=1}^{K} \\log\\left|\\det J_{f_k}\\right|"})]}),e.jsx(G,{}),e.jsxs(b,{title:"Planar Flow",children:[e.jsx("p",{children:"A simple flow layer with a single hyperplane:"}),e.jsx(t.BlockMath,{math:"f(\\mathbf{z}) = \\mathbf{z} + \\mathbf{u} \\cdot h(\\mathbf{w}^\\top \\mathbf{z} + b)"}),e.jsxs("p",{className:"mt-2",children:["The Jacobian determinant is ",e.jsx(t.InlineMath,{math:"1 + \\mathbf{u}^\\top h'(\\mathbf{w}^\\top \\mathbf{z} + b) \\mathbf{w}"}),", computable in ",e.jsx(t.InlineMath,{math:"O(d)"})," time."]})]}),e.jsx(u,{title:"Simple Planar Flow in PyTorch",code:`import torch
import torch.nn as nn

class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim))
        self.u = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        linear = z @ self.w + self.b          # (B,)
        h = torch.tanh(linear)                # (B,)
        f_z = z + self.u.unsqueeze(0) * h.unsqueeze(1)  # (B, D)
        # Log-det Jacobian
        h_prime = 1 - h ** 2                  # (B,)
        log_det = torch.log(torch.abs(1 + h_prime * (self.u @ self.w)) + 1e-8)
        return f_z, log_det

# Stack multiple planar flows
z = torch.randn(64, 2)  # 2D base distribution
log_prob_z = -0.5 * z.pow(2).sum(-1)  # log N(0,I)
total_log_det = 0
for _ in range(8):
    flow = PlanarFlow(dim=2)
    z, log_det = flow(z)
    total_log_det += log_det
log_prob_x = log_prob_z - total_log_det
print(f"Output shape: {z.shape}, mean log p(x): {log_prob_x.mean():.3f}")`}),e.jsx(g,{type:"note",title:"Key Trade-off: Expressiveness vs Efficiency",children:e.jsxs("p",{children:["The Jacobian determinant for a general ",e.jsx(t.InlineMath,{math:"d \\times d"})," matrix costs ",e.jsx(t.InlineMath,{math:"O(d^3)"}),". Practical flows use architectures with triangular Jacobians (coupling layers, autoregressive flows) for ",e.jsx(t.InlineMath,{math:"O(d)"})," computation."]})})]})}const de=Object.freeze(Object.defineProperty({__proto__:null,default:E},Symbol.toStringTag,{value:"Module"}));function C(){const[a,h]=p.useState(4),r=8;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Affine Coupling Layer"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Split position: ",a,"/",r,e.jsx("input",{type:"range",min:1,max:r-1,step:1,value:a,onChange:i=>h(parseInt(i.target.value)),className:"w-32 accent-violet-500"})]}),e.jsx("div",{className:"flex gap-1 justify-center mb-2",children:Array.from({length:r},(i,l)=>e.jsxs("div",{className:`w-8 h-8 rounded flex items-center justify-center text-xs text-white font-mono ${l<a?"bg-violet-500":"bg-orange-400"}`,children:["z",l+1]},l))}),e.jsxs("div",{className:"text-center text-xs text-gray-500",children:[e.jsxs("span",{className:"text-violet-600",children:["Identity path (",a,"d)"]})," | ",e.jsxs("span",{className:"text-orange-600",children:["Transformed path (",r-a,"d) via s,t networks"]})]}),e.jsxs("p",{className:"text-xs text-gray-400 text-center mt-1",children:["Jacobian is triangular with diagonal [1,...,1, exp(s_","{","d+1","}","), ..., exp(s_D)]"]})]})}function B(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Coupling layers split the input and transform one part conditioned on the other, yielding triangular Jacobians with cheap determinants. RealNVP and Glow build powerful flows from these simple building blocks."}),e.jsxs(f,{title:"Affine Coupling Layer (RealNVP)",children:[e.jsxs("p",{children:["Split input ",e.jsx(t.InlineMath,{math:"\\mathbf{z} = [\\mathbf{z}_{1:d}, \\mathbf{z}_{d+1:D}]"}),":"]}),e.jsx(t.BlockMath,{math:"\\mathbf{y}_{1:d} = \\mathbf{z}_{1:d}"}),e.jsx(t.BlockMath,{math:"\\mathbf{y}_{d+1:D} = \\mathbf{z}_{d+1:D} \\odot \\exp(s(\\mathbf{z}_{1:d})) + t(\\mathbf{z}_{1:d})"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"s, t"})," are arbitrary neural networks. The Jacobian determinant is simply:"]}),e.jsx(t.BlockMath,{math:"\\log|\\det J| = \\sum_{j=d+1}^{D} s_j(\\mathbf{z}_{1:d})"})]}),e.jsx(C,{}),e.jsxs(b,{title:"Glow: Generative Flow with 1x1 Convolutions",children:[e.jsx("p",{children:"Glow extends RealNVP with three innovations per step: (1) actnorm (data-dependent initialization), (2) invertible 1x1 convolution for channel permutation (replacing fixed shuffling), and (3) affine coupling layers. The 1x1 conv has Jacobian:"}),e.jsx(t.BlockMath,{math:"\\log|\\det J| = h \\cdot w \\cdot \\log|\\det \\mathbf{W}|"}),e.jsxs("p",{className:"mt-1",children:["where ",e.jsx(t.InlineMath,{math:"h, w"})," are spatial dimensions and ",e.jsx(t.InlineMath,{math:"\\mathbf{W}"})," is the weight matrix."]})]}),e.jsx(u,{title:"RealNVP Coupling Layer",code:`import torch
import torch.nn as nn

class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        half = dim // 2
        self.net = nn.Sequential(
            nn.Linear(half, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, (dim - half) * 2),  # s and t
        )
        self.half = half

    def forward(self, z):
        z1, z2 = z[:, :self.half], z[:, self.half:]
        st = self.net(z1)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s)  # clamp scale for stability
        y2 = z2 * torch.exp(s) + t
        y = torch.cat([z1, y2], dim=-1)
        log_det = s.sum(dim=-1)
        return y, log_det

    def inverse(self, y):
        y1, y2 = y[:, :self.half], y[:, self.half:]
        st = self.net(y1)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s)
        z2 = (y2 - t) * torch.exp(-s)
        return torch.cat([y1, z2], dim=-1)

layer = AffineCoupling(dim=16)
z = torch.randn(8, 16)
y, log_det = layer(z)
z_recon = layer.inverse(y)
print(f"Reconstruction error: {(z - z_recon).abs().max():.2e}")
print(f"Log-det Jacobian: {log_det[:3].tolist()}")`}),e.jsx(j,{title:"Alternating Splits Are Essential",children:e.jsx("p",{children:"A single coupling layer leaves half the dimensions unchanged. Flows must alternate which dimensions are identity vs transformed. Without this, the model cannot learn arbitrary distributions — the unchanged dimensions remain exactly Gaussian."})}),e.jsx(g,{type:"note",title:"Masked Autoregressive Flows (MAF)",children:e.jsxs("p",{children:["MAF uses autoregressive conditioning: each ",e.jsx(t.InlineMath,{math:"x_i"})," depends on all previous ",e.jsx(t.InlineMath,{math:"x_{'{<i}'}"}),". This is more expressive than coupling layers but sampling requires ",e.jsx(t.InlineMath,{math:"D"})," sequential passes. Inverse Autoregressive Flow (IAF) reverses this trade-off: fast sampling, slow density evaluation."]})})]})}const ce=Object.freeze(Object.defineProperty({__proto__:null,default:B},Symbol.toStringTag,{value:"Module"}));function P(){const[a,h]=p.useState(10),r=360,i=160,l=r/2,n=i/2,s=Array.from({length:a+1},(o,d)=>{const c=d/a,m=l+60*Math.cos(c*Math.PI*1.2)*(1-.3*c),x=n-50*Math.sin(c*Math.PI*1.5)*(.3+.7*c);return{x:m,y:x}});return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Neural ODE Trajectory"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Integration steps: ",a,e.jsx("input",{type:"range",min:3,max:30,step:1,value:a,onChange:o=>h(parseInt(o.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("svg",{width:r,height:i,className:"mx-auto block",children:[s.map((o,d)=>d>0&&e.jsx("line",{x1:s[d-1].x,y1:s[d-1].y,x2:o.x,y2:o.y,stroke:"#8b5cf6",strokeWidth:1.5,opacity:.6},d)),s.map((o,d)=>e.jsx("circle",{cx:o.x,cy:o.y,r:d===0||d===a?5:2.5,fill:d===0?"#8b5cf6":d===a?"#f97316":"#a78bfa"},d)),e.jsx("text",{x:s[0].x+8,y:s[0].y-5,className:"text-[10px] fill-violet-600",children:"z(0)"}),e.jsx("text",{x:s[a].x+8,y:s[a].y-5,className:"text-[10px] fill-orange-600",children:"z(1)=x"})]})]})}function O(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Continuous normalizing flows (CNFs) parameterize transformations as solutions to ordinary differential equations, replacing discrete flow steps with a continuous-time dynamics governed by a neural network."}),e.jsxs(f,{title:"Neural ODE",children:[e.jsx("p",{children:"A neural ODE defines the dynamics of a hidden state via:"}),e.jsx(t.BlockMath,{math:"\\frac{d\\mathbf{z}(t)}{dt} = f_\\theta(\\mathbf{z}(t), t)"}),e.jsxs("p",{className:"mt-2",children:["The output is obtained by integrating from ",e.jsx(t.InlineMath,{math:"t_0"})," to ",e.jsx(t.InlineMath,{math:"t_1"}),":"]}),e.jsx(t.BlockMath,{math:"\\mathbf{z}(t_1) = \\mathbf{z}(t_0) + \\int_{t_0}^{t_1} f_\\theta(\\mathbf{z}(t), t)\\,dt"})]}),e.jsxs(_,{title:"Instantaneous Change of Variables",id:"inst-change-vars",children:[e.jsx("p",{children:"For a continuous normalizing flow, the log-density evolves as:"}),e.jsx(t.BlockMath,{math:"\\frac{\\partial \\log p(\\mathbf{z}(t))}{\\partial t} = -\\text{tr}\\left(\\frac{\\partial f_\\theta}{\\partial \\mathbf{z}(t)}\\right)"}),e.jsxs("p",{className:"mt-2",children:["This avoids computing full Jacobian determinants. The trace can be estimated stochastically via the Hutchinson trace estimator: ",e.jsx(t.InlineMath,{math:"\\text{tr}(A) = \\mathbb{E}_{\\epsilon}[\\epsilon^\\top A \\epsilon]"}),"."]})]}),e.jsx(P,{}),e.jsx(u,{title:"Neural ODE with torchdiffeq",code:`import torch
import torch.nn as nn
# from torchdiffeq import odeint  # pip install torchdiffeq

class ODEFunc(nn.Module):
    def __init__(self, dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t, z):
        # Concatenate time to input
        t_expand = t.expand(z.shape[0], 1)
        return self.net(torch.cat([z, t_expand], dim=-1))

# Simple Euler integration (in practice, use adaptive ODE solvers)
def euler_integrate(func, z0, t_span, steps=20):
    dt = (t_span[1] - t_span[0]) / steps
    z = z0
    t = t_span[0]
    for _ in range(steps):
        z = z + dt * func(torch.tensor([t]), z)
        t += dt
    return z

func = ODEFunc(dim=2)
z0 = torch.randn(16, 2)
z1 = euler_integrate(func, z0, t_span=(0.0, 1.0), steps=20)
print(f"z(0): {z0.shape} -> z(1): {z1.shape}")
print(f"Moved distance: {(z1 - z0).norm(dim=-1).mean():.3f}")`}),e.jsx(b,{title:"FFJORD: Free-Form Jacobian of Reversible Dynamics",children:e.jsxs("p",{children:["FFJORD combines the CNF with the Hutchinson trace estimator for unbiased, scalable log-likelihood computation. It avoids all architectural restrictions (coupling layers, autoregressive structure) — the vector field ",e.jsx(t.InlineMath,{math:"f_\\theta"})," can be any neural network."]})}),e.jsx(g,{type:"note",title:"From CNFs to Flow Matching",children:e.jsx("p",{children:"Training CNFs via maximum likelihood requires solving an ODE at every training step, which is expensive. Flow matching (covered in Chapter 5) provides a simulation-free alternative: directly regressing the vector field against a target, dramatically reducing training cost."})})]})}const he=Object.freeze(Object.defineProperty({__proto__:null,default:O},Symbol.toStringTag,{value:"Module"}));function V(){const[a,h]=p.useState(500),r=1e3,i=Math.exp(-1e-4*a-.02*(a/r)*(a/r)*r*.5),l=(1-i).toFixed(3),n=i.toFixed(3),s=360,o=80,d=300;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Forward Noising Process"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["t = ",a," / ",r,e.jsx("input",{type:"range",min:0,max:r,step:10,value:a,onChange:c=>h(parseInt(c.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("svg",{width:s,height:o,className:"mx-auto block",children:[e.jsx("rect",{x:30,y:20,width:d*i,height:24,rx:4,fill:"#8b5cf6"}),e.jsx("rect",{x:30+d*i,y:20,width:d*(1-i),height:24,rx:4,fill:"#f97316",opacity:.6}),e.jsxs("text",{x:30+d*i/2,y:36,textAnchor:"middle",className:"text-[10px] fill-white font-semibold",children:["signal (",n,")"]}),1-i>.15&&e.jsxs("text",{x:30+d*i+d*(1-i)/2,y:36,textAnchor:"middle",className:"text-[10px] fill-white font-semibold",children:["noise (",l,")"]}),e.jsx("text",{x:30,y:60,className:"text-[9px] fill-gray-500",children:"clean image"}),e.jsx("text",{x:d+10,y:60,textAnchor:"end",className:"text-[9px] fill-gray-500",children:"pure noise"})]})]})}function W(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Denoising Diffusion Probabilistic Models (DDPM) generate data by learning to reverse a gradual noising process. Starting from pure Gaussian noise, the model iteratively denoises to produce high-quality samples."}),e.jsxs(f,{title:"Forward Process (Noising)",children:[e.jsxs("p",{children:["Gradually add Gaussian noise over ",e.jsx(t.InlineMath,{math:"T"})," steps with schedule ",e.jsx(t.InlineMath,{math:"\\beta_1, \\ldots, \\beta_T"}),":"]}),e.jsx(t.BlockMath,{math:"q(\\mathbf{x}_t | \\mathbf{x}_{t-1}) = \\mathcal{N}(\\mathbf{x}_t; \\sqrt{1-\\beta_t}\\,\\mathbf{x}_{t-1},\\; \\beta_t \\mathbf{I})"}),e.jsxs("p",{className:"mt-2",children:["The closed-form for any timestep with ",e.jsx(t.InlineMath,{math:"\\bar{\\alpha}_t = \\prod_{s=1}^{t}(1-\\beta_s)"}),":"]}),e.jsx(t.BlockMath,{math:"q(\\mathbf{x}_t | \\mathbf{x}_0) = \\mathcal{N}(\\mathbf{x}_t; \\sqrt{\\bar{\\alpha}_t}\\,\\mathbf{x}_0,\\; (1-\\bar{\\alpha}_t)\\mathbf{I})"})]}),e.jsx(V,{}),e.jsxs(_,{title:"DDPM Training Objective",id:"ddpm-loss",children:[e.jsx("p",{children:"The simplified training loss reduces to predicting the noise added at each step:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_{\\text{simple}} = \\mathbb{E}_{t, \\mathbf{x}_0, \\boldsymbol{\\epsilon}}\\left[\\|\\boldsymbol{\\epsilon} - \\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, t)\\|^2\\right]"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"\\mathbf{x}_t = \\sqrt{\\bar{\\alpha}_t}\\,\\mathbf{x}_0 + \\sqrt{1-\\bar{\\alpha}_t}\\,\\boldsymbol{\\epsilon}"})," and ",e.jsx(t.InlineMath,{math:"\\boldsymbol{\\epsilon} \\sim \\mathcal{N}(0, \\mathbf{I})"}),"."]})]}),e.jsx(u,{title:"DDPM Training Loop Core",code:`import torch
import torch.nn as nn

T = 1000
# Linear beta schedule
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

def q_sample(x0, t, noise=None):
    """Forward process: add noise to x0 at timestep t."""
    if noise is None:
        noise = torch.randn_like(x0)
    ab_t = alpha_bar[t].view(-1, 1, 1, 1)  # for image shapes
    return torch.sqrt(ab_t) * x0 + torch.sqrt(1 - ab_t) * noise

def training_step(model, x0, optimizer):
    B = x0.shape[0]
    t = torch.randint(0, T, (B,))
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise)

    predicted_noise = model(x_t, t)
    loss = nn.MSELoss()(predicted_noise, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# The model (U-Net) predicts the noise epsilon
# Sampling: iteratively denoise from x_T ~ N(0,I) to x_0
print(f"alpha_bar at t=0: {alpha_bar[0]:.4f}")
print(f"alpha_bar at t=500: {alpha_bar[500]:.4f}")
print(f"alpha_bar at t=999: {alpha_bar[999]:.6f}")`}),e.jsxs(b,{title:"Sampling (Reverse Process)",children:[e.jsxs("p",{children:["Starting from ",e.jsx(t.InlineMath,{math:"\\mathbf{x}_T \\sim \\mathcal{N}(0, \\mathbf{I})"}),", iterate:"]}),e.jsx(t.BlockMath,{math:"\\mathbf{x}_{t-1} = \\frac{1}{\\sqrt{\\alpha_t}}\\left(\\mathbf{x}_t - \\frac{\\beta_t}{\\sqrt{1-\\bar{\\alpha}_t}}\\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, t)\\right) + \\sigma_t \\mathbf{z}"}),e.jsxs("p",{className:"mt-1",children:["This requires ",e.jsx(t.InlineMath,{math:"T"})," forward passes (typically 1000), making sampling slow."]})]}),e.jsx(g,{type:"note",title:"DDIM: Faster Sampling",children:e.jsx("p",{children:"DDIM (Denoising Diffusion Implicit Models) uses a non-Markovian reverse process that allows skipping steps, reducing sampling from 1000 to as few as 20-50 steps with minimal quality loss. The key insight is that the same trained model can be sampled with different schedules."})})]})}const me=Object.freeze(Object.defineProperty({__proto__:null,default:W},Symbol.toStringTag,{value:"Module"}));function R(){const[a,h]=p.useState(.5),r=300,i=200,l=r/2,n=i/2,s=[];for(let o=30;o<r;o+=40)for(let d=30;d<i;d+=40){const c=l-o,m=n-d,x=Math.sqrt(c*c+m*m)+1,y=Math.min(15,200/(x*a+10)),v=c/x*y,w=m/x*y;s.push({x:o,y:d,dx:v,dy:w})}return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Score Field Visualization"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Noise level (sigma): ",a.toFixed(2),e.jsx("input",{type:"range",min:.1,max:2,step:.05,value:a,onChange:o=>h(parseFloat(o.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("svg",{width:r,height:i,className:"mx-auto block",children:[e.jsx("circle",{cx:l,cy:n,r:8,fill:"#8b5cf6",opacity:.3}),s.map((o,d)=>e.jsx("line",{x1:o.x,y1:o.y,x2:o.x+o.dx,y2:o.y+o.dy,stroke:"#8b5cf6",strokeWidth:1.5,markerEnd:"url(#arrowhead)"},d)),e.jsx("defs",{children:e.jsx("marker",{id:"arrowhead",markerWidth:"6",markerHeight:"4",refX:"5",refY:"2",orient:"auto",children:e.jsx("polygon",{points:"0 0, 6 2, 0 4",fill:"#8b5cf6"})})}),e.jsx("text",{x:l,y:n+22,textAnchor:"middle",className:"text-[10px] fill-violet-600",children:"data mode"})]}),e.jsxs("p",{className:"text-xs text-gray-500 text-center mt-1",children:["Arrows show the score (gradient of log-density) pointing toward high-density regions.",a<.5?" Low noise: sharp arrows near the mode.":" High noise: smoother, more spread field."]})]})}function U(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Score-based generative models learn the gradient of the log-probability (the score function) and generate samples by following these gradients via Langevin dynamics. This perspective unifies diffusion models with stochastic differential equations."}),e.jsxs(f,{title:"Score Function",children:[e.jsxs("p",{children:["The score of a distribution ",e.jsx(t.InlineMath,{math:"p(\\mathbf{x})"})," is the gradient of its log-density:"]}),e.jsx(t.BlockMath,{math:"\\mathbf{s}(\\mathbf{x}) = \\nabla_{\\mathbf{x}} \\log p(\\mathbf{x})"}),e.jsxs("p",{className:"mt-2",children:["A score network ",e.jsx(t.InlineMath,{math:"\\mathbf{s}_\\theta(\\mathbf{x}, \\sigma)"})," is trained to approximate the score of the noise-perturbed distribution ",e.jsx(t.InlineMath,{math:"p_\\sigma(\\mathbf{x})"}),"."]})]}),e.jsx(R,{}),e.jsxs(_,{title:"Denoising Score Matching",id:"dsm",children:[e.jsxs("p",{children:["Instead of directly matching the intractable true score, we match the score of noisy data ",e.jsx(t.InlineMath,{math:"q_\\sigma(\\tilde{\\mathbf{x}}|\\mathbf{x}) = \\mathcal{N}(\\tilde{\\mathbf{x}}; \\mathbf{x}, \\sigma^2 I)"}),":"]}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_{\\text{DSM}} = \\mathbb{E}_{\\mathbf{x}, \\tilde{\\mathbf{x}}}\\left[\\|\\mathbf{s}_\\theta(\\tilde{\\mathbf{x}}, \\sigma) - \\nabla_{\\tilde{\\mathbf{x}}} \\log q_\\sigma(\\tilde{\\mathbf{x}}|\\mathbf{x})\\|^2\\right]"}),e.jsxs("p",{className:"mt-2",children:["Since ",e.jsx(t.InlineMath,{math:"\\nabla_{\\tilde{\\mathbf{x}}} \\log q_\\sigma = -(\\tilde{\\mathbf{x}} - \\mathbf{x})/\\sigma^2"}),", this is equivalent to noise prediction (the DDPM objective)."]})]}),e.jsxs(b,{title:"Langevin Dynamics Sampling",children:[e.jsx("p",{children:"Given the score, generate samples via annealed Langevin dynamics:"}),e.jsx(t.BlockMath,{math:"\\mathbf{x}_{i+1} = \\mathbf{x}_i + \\frac{\\eta}{2}\\,\\mathbf{s}_\\theta(\\mathbf{x}_i, \\sigma) + \\sqrt{\\eta}\\,\\mathbf{z}, \\quad \\mathbf{z} \\sim \\mathcal{N}(0, I)"}),e.jsxs("p",{className:"mt-1",children:["The noise levels ",e.jsx(t.InlineMath,{math:"\\sigma"})," are annealed from large to small during sampling."]})]}),e.jsx(u,{title:"Score Matching and Langevin Sampling",code:`import torch
import torch.nn as nn

class ScoreNet(nn.Module):
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, sigma):
        sigma_input = sigma.expand(x.shape[0], 1)
        return self.net(torch.cat([x, sigma_input], dim=-1))

# Denoising score matching loss
def dsm_loss(model, x, sigma=0.5):
    noise = torch.randn_like(x)
    x_noisy = x + sigma * noise
    score_pred = model(x_noisy, torch.tensor([sigma]))
    target = -noise / sigma  # true score of Gaussian perturbation
    return ((score_pred - target) ** 2).sum(dim=-1).mean()

# Langevin dynamics sampling
@torch.no_grad()
def langevin_sample(model, shape, sigmas, steps_per_sigma=100, lr=0.01):
    x = torch.randn(shape)
    for sigma in sigmas:
        for _ in range(steps_per_sigma):
            score = model(x, torch.tensor([sigma]))
            x = x + (lr / 2) * score + torch.sqrt(torch.tensor(lr)) * torch.randn_like(x)
    return x

model = ScoreNet(dim=2)
x_data = torch.randn(256, 2) * 0.5 + torch.tensor([2.0, 2.0])
loss = dsm_loss(model, x_data)
print(f"DSM loss: {loss.item():.4f}")`}),e.jsx(g,{type:"note",title:"SDE Framework: Unifying Diffusion and Score Models",children:e.jsxs("p",{children:["Song et al. showed that both DDPM and score-based models are discretizations of a continuous-time SDE: ",e.jsx(t.InlineMath,{math:"d\\mathbf{x} = f(\\mathbf{x},t)\\,dt + g(t)\\,d\\mathbf{w}"}),". The reverse SDE uses the score: ",e.jsx(t.InlineMath,{math:"d\\mathbf{x} = [f - g^2 \\nabla_x \\log p_t]\\,dt + g\\,d\\bar{\\mathbf{w}}"}),". This unification enables flexible solver choices (ODE for deterministic, SDE for stochastic sampling)."]})})]})}const xe=Object.freeze(Object.defineProperty({__proto__:null,default:U},Symbol.toStringTag,{value:"Module"}));function $(){const[a,h]=p.useState(7.5),r=Math.max(0,1-(a-1)*.08),i=Math.min(1,.3+a*.08),l=a>12?Math.min(1,(a-12)*.1):0;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Classifier-Free Guidance Scale"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["w = ",a.toFixed(1),e.jsx("input",{type:"range",min:1,max:20,step:.5,value:a,onChange:n=>h(parseFloat(n.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("div",{className:"flex gap-4 justify-center items-end h-24",children:[e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("div",{className:"w-14 bg-violet-400 rounded-t transition-all",style:{height:`${i*80}px`}}),e.jsx("span",{className:"text-xs text-gray-500 mt-1",children:"Quality"})]}),e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("div",{className:"w-14 bg-violet-600 rounded-t transition-all",style:{height:`${r*80}px`}}),e.jsx("span",{className:"text-xs text-gray-500 mt-1",children:"Diversity"})]}),e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("div",{className:"w-14 bg-red-400 rounded-t transition-all",style:{height:`${l*80}px`}}),e.jsx("span",{className:"text-xs text-gray-500 mt-1",children:"Artifacts"})]})]}),e.jsx("p",{className:"text-xs text-gray-500 text-center mt-2",children:a<3?"Low guidance: diverse but may not match the condition well":a<=10?"Sweet spot: good balance of quality and condition adherence":"High guidance: oversaturated, artifacts may appear"})]})}function J(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Classifier-free guidance (CFG) enables conditional generation without a separate classifier, trading off sample diversity for stronger adherence to the conditioning signal. It has become the standard approach in text-to-image models like Stable Diffusion and DALL-E."}),e.jsxs(f,{title:"Classifier-Free Guidance",children:[e.jsxs("p",{children:["During training, randomly drop the condition ",e.jsx(t.InlineMath,{math:"c"})," (replace with null) with probability ",e.jsx(t.InlineMath,{math:"p_{\\text{uncond}}"}),". At inference, combine conditional and unconditional predictions:"]}),e.jsx(t.BlockMath,{math:"\\tilde{\\boldsymbol{\\epsilon}}_\\theta(\\mathbf{x}_t, c) = (1 + w)\\,\\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, c) - w\\,\\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, \\varnothing)"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"w"})," is the guidance scale. Equivalently in score form:"]}),e.jsx(t.BlockMath,{math:"\\tilde{\\nabla} \\log p(\\mathbf{x}_t | c) = \\nabla \\log p(\\mathbf{x}_t) + (1+w)\\left(\\nabla \\log p(c | \\mathbf{x}_t)\\right)"})]}),e.jsx($,{}),e.jsxs(b,{title:"Classifier Guidance (Original Approach)",children:[e.jsxs("p",{children:["The original approach by Dhariwal & Nichol uses a separate classifier ",e.jsx(t.InlineMath,{math:"p_\\phi(c|\\mathbf{x}_t)"})," trained on noisy data:"]}),e.jsx(t.BlockMath,{math:"\\tilde{\\boldsymbol{\\epsilon}} = \\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, t) - s \\cdot \\sigma_t \\nabla_{\\mathbf{x}_t} \\log p_\\phi(c|\\mathbf{x}_t)"}),e.jsx("p",{className:"mt-1",children:"CFG eliminates the need for this external classifier, simplifying the pipeline."})]}),e.jsx(u,{title:"Classifier-Free Guidance Sampling",code:`import torch

def cfg_sample_step(model, x_t, t, condition, w=7.5, null_cond=None):
    """Single CFG denoising step."""
    # Conditional prediction
    eps_cond = model(x_t, t, condition)
    # Unconditional prediction (condition dropped)
    eps_uncond = model(x_t, t, null_cond)
    # Guided prediction
    eps_guided = (1 + w) * eps_cond - w * eps_uncond
    return eps_guided

# Training with random condition dropout
def training_step(model, x0, condition, p_uncond=0.1):
    t = torch.randint(0, 1000, (x0.shape[0],))
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise)  # forward noising

    # Randomly drop condition
    mask = torch.rand(x0.shape[0]) < p_uncond
    cond_input = condition.clone()
    cond_input[mask] = 0  # null condition (e.g., zero embedding)

    eps_pred = model(x_t, t, cond_input)
    loss = ((eps_pred - noise) ** 2).mean()
    return loss

# Typical guidance scales:
# w=1.0: minimal guidance
# w=7.5: default for Stable Diffusion
# w=15+: very strong guidance (risk of artifacts)
print("CFG: two forward passes per step (conditional + unconditional)")
print("Doubles inference cost but dramatically improves condition adherence")`}),e.jsx(j,{title:"Guidance Scale Pitfalls",children:e.jsxs("p",{children:["Very high guidance scales (",e.jsx(t.InlineMath,{math:"w > 15"}),") can cause color saturation, loss of fine detail, and unrealistic artifacts. Dynamic guidance (varying ",e.jsx(t.InlineMath,{math:"w"})," across timesteps) and guidance rescaling can mitigate these issues."]})}),e.jsx(g,{type:"note",title:"Beyond Text Conditioning",children:e.jsx("p",{children:"CFG works with any conditioning signal: text embeddings (Stable Diffusion), class labels (ImageNet generation), spatial maps (ControlNet), or even image references (IP-Adapter). The same principle applies: train with dropout, guide at inference."})})]})}const pe=Object.freeze(Object.defineProperty({__proto__:null,default:J},Symbol.toStringTag,{value:"Module"}));function H(){const[a,h]=p.useState(.5),r=340,i=160,l=[{x0:40,y0:30,x1:280,y1:40},{x0:60,y0:80,x1:300,y1:90},{x0:30,y0:130,x1:260,y1:120},{x0:70,y0:50,x1:290,y1:60}];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Optimal Transport Paths"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["t = ",a.toFixed(2),e.jsx("input",{type:"range",min:0,max:1,step:.02,value:a,onChange:n=>h(parseFloat(n.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("svg",{width:r,height:i,className:"mx-auto block",children:[l.map((n,s)=>{const o=n.x0+(n.x1-n.x0)*a,d=n.y0+(n.y1-n.y0)*a;return e.jsxs("g",{children:[e.jsx("line",{x1:n.x0,y1:n.y0,x2:n.x1,y2:n.y1,stroke:"#d1d5db",strokeWidth:.8,strokeDasharray:"3,3"}),e.jsx("circle",{cx:n.x0,cy:n.y0,r:4,fill:"#8b5cf6",opacity:.4}),e.jsx("circle",{cx:n.x1,cy:n.y1,r:4,fill:"#f97316",opacity:.4}),e.jsx("circle",{cx:o,cy:d,r:5,fill:"#8b5cf6"})]},s)}),e.jsx("text",{x:20,y:i-5,className:"text-[10px] fill-violet-500",children:"noise (t=0)"}),e.jsx("text",{x:r-80,y:i-5,className:"text-[10px] fill-orange-500",children:"data (t=1)"})]})]})}function K(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Flow matching learns a vector field that transports a simple distribution to the data distribution. When paths are straight (via optimal transport), training is more efficient and sampling requires fewer steps than diffusion models."}),e.jsxs(f,{title:"Flow Matching Objective",children:[e.jsxs("p",{children:["Learn a time-dependent vector field ",e.jsx(t.InlineMath,{math:"v_\\theta(\\mathbf{x}, t)"})," that generates a probability path from noise to data:"]}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_{\\text{FM}} = \\mathbb{E}_{t, q(\\mathbf{x}_1)}\\left[\\|v_\\theta(\\mathbf{x}_t, t) - u_t(\\mathbf{x}_t | \\mathbf{x}_1)\\|^2\\right]"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"u_t"})," is the target vector field and ",e.jsx(t.InlineMath,{math:"\\mathbf{x}_t"})," interpolates between noise and data."]})]}),e.jsxs(_,{title:"Conditional OT Path",id:"cot-path",children:[e.jsx("p",{children:"The optimal transport conditional path is a straight line:"}),e.jsx(t.BlockMath,{math:"\\mathbf{x}_t = (1 - t)\\mathbf{x}_0 + t\\mathbf{x}_1, \\quad \\mathbf{x}_0 \\sim \\mathcal{N}(0, I),\\; \\mathbf{x}_1 \\sim q(\\mathbf{x})"}),e.jsx("p",{className:"mt-2",children:"The target vector field is constant along each path:"}),e.jsx(t.BlockMath,{math:"u_t(\\mathbf{x}_t | \\mathbf{x}_1) = \\mathbf{x}_1 - \\mathbf{x}_0"})]}),e.jsx(H,{}),e.jsx(b,{title:"Flow Matching vs Diffusion",children:e.jsx("p",{children:"Diffusion models use curved, stochastic paths (adding/removing noise gradually). Flow matching with OT uses straight paths from noise to data. Benefits: (1) simulation-free training (no ODE solving), (2) straighter trajectories need fewer integration steps at inference, (3) simpler implementation."})}),e.jsx(u,{title:"Flow Matching Training",code:`import torch
import torch.nn as nn

class VectorField(nn.Module):
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t.view(-1, 1).expand(-1, 1)], dim=-1))

def flow_matching_loss(model, x1):
    """OT conditional flow matching loss."""
    B = x1.shape[0]
    t = torch.rand(B)
    x0 = torch.randn_like(x1)  # noise samples

    # Straight-line interpolation
    x_t = (1 - t.view(-1, 1)) * x0 + t.view(-1, 1) * x1

    # Target: direction from noise to data
    target = x1 - x0

    # Predict vector field
    v_pred = model(x_t, t)
    return ((v_pred - target) ** 2).mean()

# Sampling: integrate the learned vector field
@torch.no_grad()
def sample(model, shape, steps=50):
    x = torch.randn(shape)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((shape[0],), i * dt)
        x = x + model(x, t) * dt  # Euler integration
    return x

model = VectorField(dim=2)
data = torch.randn(256, 2) + 3  # shifted Gaussian
loss = flow_matching_loss(model, data)
print(f"Flow matching loss: {loss.item():.4f}")`}),e.jsx(g,{type:"note",title:"Stable Diffusion 3 Uses Flow Matching",children:e.jsx("p",{children:"Modern text-to-image models like Stable Diffusion 3 have moved from DDPM-style diffusion to rectified flow matching, benefiting from straighter sampling trajectories and better training stability. The core insight: straight paths are easier to learn and faster to sample."})})]})}const fe=Object.freeze(Object.defineProperty({__proto__:null,default:K},Symbol.toStringTag,{value:"Module"}));function Q(){const[a,h]=p.useState(0),r=340,i=150,l=Math.max(0,1-a*.35),n=[{x0:30,y0:130,x1:310,y1:30},{x0:50,y0:100,x1:280,y1:50},{x0:40,y0:60,x1:300,y1:110}];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Rectification Iterations"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Reflow iteration: ",a,e.jsx("input",{type:"range",min:0,max:3,step:1,value:a,onChange:s=>h(parseInt(s.target.value)),className:"w-32 accent-violet-500"}),e.jsxs("span",{className:"text-xs text-violet-600",children:["straightness: ",(1-l).toFixed(0)==="1"?"1.00":(1-l).toFixed(2)]})]}),e.jsx("svg",{width:r,height:i,className:"mx-auto block",children:n.map((s,o)=>{const d=(s.x0+s.x1)/2+l*(40-o*30),c=(s.y0+s.y1)/2+l*(20*(o-1));return e.jsxs("g",{children:[e.jsx("path",{d:`M${s.x0},${s.y0} Q${d},${c} ${s.x1},${s.y1}`,fill:"none",stroke:"#8b5cf6",strokeWidth:2}),e.jsx("circle",{cx:s.x0,cy:s.y0,r:4,fill:"#8b5cf6"}),e.jsx("circle",{cx:s.x1,cy:s.y1,r:4,fill:"#f97316"})]},o)})}),e.jsx("p",{className:"text-xs text-gray-500 text-center mt-1",children:a===0?"Initial: curved trajectories from diffusion training":a===1?"After 1 reflow: noticeably straighter":a===2?"After 2 reflows: nearly straight":"After 3 reflows: almost perfectly straight (1-step generation possible)"})]})}function X(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Rectified flows iteratively straighten the transport paths between noise and data distributions. Straighter paths enable accurate generation with fewer integration steps, approaching single-step generation."}),e.jsxs(f,{title:"Rectified Flow (Reflow)",children:[e.jsxs("p",{children:["Starting with a learned flow ",e.jsx(t.InlineMath,{math:"v_\\theta"}),", generate coupled pairs ",e.jsx(t.InlineMath,{math:"(\\mathbf{x}_0, \\mathbf{x}_1)"})," by running the ODE, then retrain on straight-line interpolants:"]}),e.jsx(t.BlockMath,{math:"\\mathbf{x}_t = (1-t)\\mathbf{x}_0 + t\\mathbf{x}_1, \\quad u_t = \\mathbf{x}_1 - \\mathbf{x}_0"}),e.jsx("p",{className:"mt-2",children:"Each reflow iteration produces straighter trajectories, reducing truncation error."})]}),e.jsx(Q,{}),e.jsxs(_,{title:"Straightness Bound",id:"straightness",children:[e.jsx("p",{children:"The transport cost (path curvature) is non-increasing with each reflow iteration:"}),e.jsx(t.BlockMath,{math:"\\mathbb{E}\\left[\\int_0^1 \\|v_{\\theta}^{(k+1)}(\\mathbf{x}_t, t) - (\\mathbf{x}_1 - \\mathbf{x}_0)\\|^2 dt\\right] \\leq \\mathbb{E}\\left[\\int_0^1 \\|v_{\\theta}^{(k)}(\\mathbf{x}_t, t) - (\\mathbf{x}_1 - \\mathbf{x}_0)\\|^2 dt\\right]"}),e.jsx("p",{className:"mt-2",children:"In the limit, trajectories become straight lines and one-step generation is exact."})]}),e.jsxs(b,{title:"Distillation for One-Step Generation",children:[e.jsxs("p",{children:["After rectification, further distill into a single-step model. The student network learns to predict ",e.jsx(t.InlineMath,{math:"\\mathbf{x}_1"})," directly from ",e.jsx(t.InlineMath,{math:"\\mathbf{x}_0"}),":"]}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_{\\text{distill}} = \\|\\text{Student}(\\mathbf{x}_0) - \\text{ODE}(\\mathbf{x}_0; v_\\theta)\\|^2"}),e.jsx("p",{children:"This gives single-step generation with quality approaching the multi-step teacher."})]}),e.jsx(u,{title:"Rectified Flow: Reflow Procedure",code:`import torch
import torch.nn as nn

class FlowModel(nn.Module):
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, t.unsqueeze(-1)], dim=-1))

@torch.no_grad()
def generate_pairs(model, n=1000, steps=100):
    """Generate (x0, x1) pairs by running the ODE."""
    x0 = torch.randn(n, 2)
    x = x0.clone()
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), i * dt)
        x = x + model(x, t) * dt
    x1 = x
    return x0, x1  # coupled noise-data pairs

def reflow_loss(model, x0, x1):
    """Train on straight-line interpolants of coupled pairs."""
    B = x0.shape[0]
    t = torch.rand(B)
    x_t = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
    target = x1 - x0  # straight-line direction
    v_pred = model(x_t, t)
    return ((v_pred - target) ** 2).mean()

model = FlowModel()
# Reflow: generate pairs -> retrain -> repeat
# x0, x1 = generate_pairs(model)
# loss = reflow_loss(new_model, x0, x1)
print("Reflow procedure:")
print("1. Train initial flow matching model")
print("2. Generate coupled (noise, data) pairs via ODE")
print("3. Retrain on straight-line interpolants")
print("4. Repeat 1-2 times for near-straight trajectories")`}),e.jsx(g,{type:"note",title:"InstaFlow and Practical Applications",children:e.jsx("p",{children:"InstaFlow applies rectified flows to Stable Diffusion, achieving one-step text-to-image generation. The combination of flow matching + reflow + distillation has become a leading paradigm for fast generative models, powering systems like Stable Diffusion 3 and FLUX."})})]})}const ge=Object.freeze(Object.defineProperty({__proto__:null,default:X},Symbol.toStringTag,{value:"Module"}));function Z(){const[a,h]=p.useState(1),r=360,i=120,l=Array.from({length:6},(n,s)=>({x:30+s*60,y:60+30*Math.sin(s*.8)*(1-s/6),t:(5-s)/5}));return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Consistency Model: Any Point Maps to Origin"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Sampling steps: ",a,e.jsx("input",{type:"range",min:1,max:4,step:1,value:a,onChange:n=>h(parseInt(n.target.value)),className:"w-32 accent-violet-500"})]}),e.jsxs("svg",{width:r,height:i,className:"mx-auto block",children:[l.map((n,s)=>s>0&&e.jsx("line",{x1:l[s-1].x,y1:l[s-1].y,x2:n.x,y2:n.y,stroke:"#d1d5db",strokeWidth:1,strokeDasharray:"3,3"},`l${s}`)),l.map((n,s)=>{const o=s===0||(a>=4?!0:s>=6-a-1);return e.jsxs("g",{children:[e.jsx("circle",{cx:n.x,cy:n.y,r:o?5:3,fill:s===5?"#f97316":"#8b5cf6",opacity:o?1:.3}),o&&s<5&&e.jsx("line",{x1:n.x,y1:n.y,x2:l[5].x,y2:l[5].y,stroke:"#8b5cf6",strokeWidth:1.5,opacity:.4})]},s)}),e.jsx("text",{x:l[0].x,y:15,textAnchor:"middle",className:"text-[9px] fill-gray-500",children:"t=T (noise)"}),e.jsx("text",{x:l[5].x,y:15,textAnchor:"middle",className:"text-[9px] fill-orange-500",children:"t=0 (data)"})]}),e.jsxs("p",{className:"text-xs text-gray-500 text-center mt-1",children:["All points on the same trajectory map to the same output. ",a===1?"Single step: direct jump from any t to data.":`${a} steps: multi-step refinement.`]})]})}function Y(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Consistency models learn to map any point on a diffusion trajectory directly to the trajectory's origin (clean data), enabling one-step generation without iterative denoising."}),e.jsxs(f,{title:"Consistency Function",children:[e.jsxs("p",{children:["A consistency function ",e.jsx(t.InlineMath,{math:"f_\\theta"})," satisfies the self-consistency property:"]}),e.jsx(t.BlockMath,{math:"f_\\theta(\\mathbf{x}_t, t) = f_\\theta(\\mathbf{x}_{t'}, t') \\quad \\forall\\, t, t' \\in [\\epsilon, T]"}),e.jsxs("p",{className:"mt-2",children:["For any two points on the same PF-ODE trajectory, the model produces the same output. The boundary condition is ",e.jsx(t.InlineMath,{math:"f_\\theta(\\mathbf{x}_\\epsilon, \\epsilon) = \\mathbf{x}_\\epsilon \\approx \\mathbf{x}_0"}),"."]})]}),e.jsx(Z,{}),e.jsxs(_,{title:"Consistency Training Loss",id:"ct-loss",children:[e.jsxs("p",{children:["Enforce consistency between adjacent timesteps using the target network ",e.jsx(t.InlineMath,{math:"\\theta^{-}"})," (EMA):"]}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_{\\text{CT}} = \\mathbb{E}\\left[d\\left(f_\\theta(\\mathbf{x}_{t_{n+1}}, t_{n+1}),\\; f_{\\theta^{-}}(\\hat{\\mathbf{x}}_{t_n}, t_n)\\right)\\right]"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"\\hat{\\mathbf{x}}_{t_n}"})," is obtained by one step of a numerical ODE solver from ",e.jsx(t.InlineMath,{math:"\\mathbf{x}_{t_{n+1}}"}),", and ",e.jsx(t.InlineMath,{math:"d"})," is a distance metric (e.g., LPIPS)."]})]}),e.jsx(b,{title:"Consistency Distillation vs Training",children:e.jsxs("p",{children:[e.jsx("strong",{children:"Consistency distillation"})," (CD) requires a pre-trained diffusion model to generate ODE trajectories. ",e.jsx("strong",{children:"Consistency training"})," (CT) trains from scratch by estimating the ODE step with a single denoiser evaluation, removing the dependency on a teacher model."]})}),e.jsx(u,{title:"Consistency Model Pseudocode",code:`import torch
import torch.nn as nn

class ConsistencyModel(nn.Module):
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        self.eps = 0.002  # boundary epsilon

    def forward(self, x, t):
        # Skip connection enforces boundary condition: f(x, eps) = x
        t_input = t.unsqueeze(-1)
        c_skip = self.eps / (t_input ** 2 + self.eps ** 2).sqrt()
        c_out = (t_input - self.eps) / (t_input ** 2 + self.eps ** 2).sqrt()
        return c_skip * x + c_out * self.net(torch.cat([x, t_input], dim=-1))

def consistency_loss(model, target_model, x0, noise_schedule):
    B = x0.shape[0]
    # Sample adjacent timestep pairs
    n = torch.randint(0, len(noise_schedule) - 1, (B,))
    t_next = noise_schedule[n + 1]
    t_curr = noise_schedule[n]

    noise = torch.randn_like(x0)
    x_next = x0 + t_next.unsqueeze(-1) * noise

    # One ODE step estimate (using pre-trained denoiser or self)
    x_curr = x0 + t_curr.unsqueeze(-1) * noise  # simplified

    pred = model(x_next, t_next)
    with torch.no_grad():
        target = target_model(x_curr, t_curr)
    return ((pred - target) ** 2).mean()

model = ConsistencyModel()
x = torch.randn(8, 2)
t = torch.ones(8) * 0.5
out = model(x, t)
print(f"One-step generation: input {x.shape} -> output {out.shape}")`}),e.jsx(g,{type:"note",title:"Improved Consistency Training (iCT)",children:e.jsx("p",{children:"Improved consistency training removes the need for a pre-trained diffusion model entirely, using adaptive schedules for the number of discretization steps and the EMA decay rate. iCT achieves state-of-the-art FID for single-step generation on ImageNet, making it a compelling alternative to multi-step diffusion."})})]})}const ue=Object.freeze(Object.defineProperty({__proto__:null,default:Y},Symbol.toStringTag,{value:"Module"}));export{ne as a,ie as b,re as c,le as d,oe as e,de as f,ce as g,he as h,me as i,xe as j,pe as k,fe as l,ge as m,ue as n,se as s};

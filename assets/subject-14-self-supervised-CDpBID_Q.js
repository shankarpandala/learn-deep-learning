import{j as e,r as m}from"./vendor-DpISuAX6.js";import{r as t}from"./vendor-katex-CbWCYdth.js";import{D as x,E as g,P as u,N as p,T as f,W as b}from"./subject-01-foundations-D0A1VJsr.js";function _(){const[s,d]=m.useState("rotation"),r={rotation:{label:"Rotation Prediction",desc:"Predict which of 4 rotations (0, 90, 180, 270) was applied. Forces understanding of object orientation and canonical poses.",classes:4},jigsaw:{label:"Jigsaw Puzzle",desc:"Predict the permutation of shuffled image patches. Learns spatial relationships between parts.",classes:"30-1000"},colorization:{label:"Colorization",desc:"Predict color channels from grayscale input. Requires semantic understanding (sky is blue, grass is green).",classes:"continuous"},inpainting:{label:"Inpainting",desc:"Reconstruct missing image regions. Forces understanding of context and object structure.",classes:"continuous"}},a=r[s];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-2 text-base font-bold text-gray-800 dark:text-gray-200",children:"Pretext Tasks"}),e.jsx("div",{className:"flex gap-2 mb-3 flex-wrap",children:Object.entries(r).map(([l,o])=>e.jsx("button",{onClick:()=>d(l),className:`px-3 py-1 rounded-full text-xs font-medium transition-colors ${s===l?"bg-violet-500 text-white":"bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400"}`,children:o.label},l))}),e.jsxs("div",{className:"bg-gray-50 dark:bg-gray-800 rounded-lg p-3",children:[e.jsx("p",{className:"text-sm text-gray-700 dark:text-gray-300",children:a.desc}),e.jsxs("p",{className:"text-xs text-violet-600 mt-1",children:["Output classes: ",a.classes]})]})]})}function v(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Self-supervised learning creates supervision signals from the data itself, eliminating the need for manual labels. Pretext tasks exploit the inherent structure of images to train representations that transfer to downstream tasks."}),e.jsxs(x,{title:"Pretext Task",children:[e.jsxs("p",{children:["A pretext task is a surrogate objective where labels are derived automatically from the input. The model learns representations ",e.jsx(t.InlineMath,{math:"f_\\theta(\\mathbf{x})"})," useful for downstream tasks by solving the pretext task:"]}),e.jsx(t.BlockMath,{math:"\\min_\\theta \\mathbb{E}_{\\mathbf{x} \\sim \\mathcal{D}}\\left[\\mathcal{L}(g_\\phi(f_\\theta(T(\\mathbf{x}))), y_T)\\right]"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"T"})," is a transformation, ",e.jsx(t.InlineMath,{math:"y_T"})," is the pseudo-label, and",e.jsx(t.InlineMath,{math:"g_\\phi"})," is a task-specific head (discarded after pre-training)."]})]}),e.jsx(_,{}),e.jsx(g,{title:"Rotation Prediction (RotNet)",children:e.jsxs("p",{children:["Apply one of four rotations ",e.jsx(t.InlineMath,{math:"r \\in \\{0°, 90°, 180°, 270°\\}"})," and train a classifier to predict which rotation was applied. This forces the network to understand object semantics — recognizing that a dog is upside-down requires knowing what a dog looks like right-side up."]})}),e.jsx(u,{title:"Rotation Prediction Pretext Task",code:`import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class RotNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone  # e.g., ResNet without final FC
        self.rotation_head = nn.Linear(512, 4)  # 4 rotation classes

    def forward(self, x):
        features = self.backbone(x)
        return self.rotation_head(features)

def create_rotation_batch(images):
    """Create self-supervised batch with 4 rotations per image."""
    rotated, labels = [], []
    for img in images:
        for r, angle in enumerate([0, 90, 180, 270]):
            rotated.append(TF.rotate(img, angle))
            labels.append(r)
    return torch.stack(rotated), torch.tensor(labels)

# Training
# images = next(dataloader)  # unlabeled images
# x_rot, y_rot = create_rotation_batch(images)
# logits = model(x_rot)
# loss = nn.CrossEntropyLoss()(logits, y_rot)

# After pre-training: discard rotation_head, fine-tune backbone
print("RotNet: 4-class classification on rotation angle")
print("Key insight: semantic understanding is needed to detect orientation")`}),e.jsx(p,{type:"note",title:"Jigsaw Puzzles (Noroozi & Favaro)",children:e.jsx("p",{children:"Split an image into a 3x3 grid of patches, shuffle them, and train the network to predict the permutation. From 9! = 362,880 possible permutations, a subset of ~1000 maximally different permutations is selected. This teaches spatial reasoning and part-whole relationships."})}),e.jsx(p,{type:"note",title:"Limitations of Pretext Tasks",children:e.jsx("p",{children:"Hand-designed pretext tasks have a fundamental limitation: the learned features may be biased toward solving the specific proxy task rather than learning general representations. Contrastive and masked modeling approaches (next sections) largely supersede these methods."})})]})}const U=Object.freeze(Object.defineProperty({__proto__:null,default:v},Symbol.toStringTag,{value:"Module"}));function y(){const[s,d]=m.useState(.5),r=8,a=r*r,l=Math.round(a*s),[o]=m.useState(()=>Array.from({length:a},()=>Math.random())),c=o.map((i,n)=>{const h=o.slice().sort()[l-1]||0;return i<=h});return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Predictive Masking"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Mask ratio: ",(s*100).toFixed(0),"%",e.jsx("input",{type:"range",min:.1,max:.9,step:.05,value:s,onChange:i=>d(parseFloat(i.target.value)),className:"w-40 accent-violet-500"}),e.jsxs("span",{className:"text-xs",children:["(",l,"/",a," patches masked)"]})]}),e.jsx("div",{className:"flex justify-center",children:e.jsx("div",{className:"grid gap-0.5",style:{gridTemplateColumns:`repeat(${r}, 1fr)`},children:Array.from({length:a},(i,n)=>e.jsx("div",{className:`w-5 h-5 rounded-sm ${c[n]?"bg-gray-300 dark:bg-gray-600":"bg-violet-400"}`},n))})}),e.jsxs("div",{className:"flex justify-center gap-4 text-xs mt-2",children:[e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-3 bg-violet-400 rounded-sm"})," Visible"]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-3 bg-gray-300 rounded-sm"})," Masked (predict)"]})]})]})}function j(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Predictive self-supervision trains models to predict missing or future parts of the input. This paradigm, inspired by language model pre-training (predicting next tokens), has become the dominant approach in self-supervised visual representation learning."}),e.jsxs(x,{title:"Predictive Self-Supervised Learning",children:[e.jsxs("p",{children:["Given input ",e.jsx(t.InlineMath,{math:"\\mathbf{x}"}),", partition into visible context ",e.jsx(t.InlineMath,{math:"\\mathbf{x}_v"})," and target ",e.jsx(t.InlineMath,{math:"\\mathbf{x}_t"}),":"]}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = \\mathbb{E}\\left[d\\left(g_\\phi(f_\\theta(\\mathbf{x}_v)),\\; \\mathbf{x}_t\\right)\\right]"}),e.jsxs("p",{className:"mt-2",children:["The distance ",e.jsx(t.InlineMath,{math:"d"})," can operate in pixel space (MSE), token space (cross-entropy), or representation space (feature regression)."]})]}),e.jsx(y,{}),e.jsxs(f,{title:"Information-Theoretic Perspective",id:"info-theory",children:[e.jsx("p",{children:"Predictive learning maximizes a lower bound on the mutual information between visible and target parts:"}),e.jsx(t.BlockMath,{math:"I(\\mathbf{x}_v; \\mathbf{x}_t) \\geq -\\mathcal{L}_{\\text{pred}} + H(\\mathbf{x}_t)"}),e.jsx("p",{className:"mt-2",children:"Better prediction (lower loss) implies more information captured about the data structure."})]}),e.jsx(g,{title:"Context Prediction (Doersch et al., 2015)",children:e.jsx("p",{children:"One of the earliest visual predictive tasks: given a center patch, predict the relative position (1 of 8 neighbors) of a second patch. This teaches the network about spatial layouts and object parts — for example, an eye patch is typically above a mouth patch."})}),e.jsx(u,{title:"Relative Position Prediction",code:`import torch
import torch.nn as nn

class RelativePositionNet(nn.Module):
    def __init__(self, backbone_dim=512):
        super().__init__()
        # Two patch embeddings are concatenated
        self.position_head = nn.Sequential(
            nn.Linear(backbone_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 8),  # 8 relative positions
        )

    def forward(self, patch_center, patch_neighbor):
        return self.position_head(torch.cat([patch_center, patch_neighbor], dim=-1))

def extract_patch_pairs(images, patch_size=64, gap=32):
    """Extract center patch and one of 8 neighboring patches."""
    B, C, H, W = images.shape
    # Center patch coordinates
    cy = H // 2 - patch_size // 2
    cx = W // 2 - patch_size // 2
    center = images[:, :, cy:cy+patch_size, cx:cx+patch_size]

    # Random neighbor direction (0-7: TL, T, TR, L, R, BL, B, BR)
    direction = torch.randint(0, 8, (B,))
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    neighbors, labels = [], direction
    for i in range(B):
        dy, dx = offsets[direction[i]]
        ny = cy + dy * (patch_size + gap)
        nx = cx + dx * (patch_size + gap)
        ny, nx = max(0, ny), max(0, nx)
        neighbors.append(images[i, :, ny:ny+patch_size, nx:nx+patch_size])

    return center, torch.stack(neighbors), labels

print("Relative position: 8-class classification")
print("Forces learning spatial relationships between object parts")`}),e.jsx(p,{type:"note",title:"From Pixel Prediction to Feature Prediction",children:e.jsxs("p",{children:["Predicting raw pixels encourages learning low-level statistics (textures, edges) rather than high-level semantics. Modern approaches predict in ",e.jsx("em",{children:"feature space"})," instead: MAE predicts normalized pixel patches, BEiT predicts discrete visual tokens, and data2vec predicts teacher network representations. This shift dramatically improves downstream performance."]})})]})}const Y=Object.freeze(Object.defineProperty({__proto__:null,default:j},Symbol.toStringTag,{value:"Module"}));function k(){const[s,d]=m.useState("none"),r={none:{spread:5,label:"No prevention",desc:"All representations collapse to a single point."},negatives:{spread:60,label:"Negative pairs",desc:"Contrastive loss pushes negatives apart."},momentum:{spread:55,label:"Momentum encoder",desc:"Slow-moving target prevents rapid collapse."},variance:{spread:58,label:"Variance regularization",desc:"Explicit loss term maintains variance."}},a=r[s],l=Array.from({length:20},(o,c)=>{const i=c/20*Math.PI*2,n=a.spread*(.5+.5*Math.sin(c*1.7));return{x:150+n*Math.cos(i),y:75+n*Math.sin(i)*.6}});return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-2 text-base font-bold text-gray-800 dark:text-gray-200",children:"Representational Collapse"}),e.jsx("div",{className:"flex gap-2 mb-3 flex-wrap",children:Object.entries(r).map(([o,c])=>e.jsx("button",{onClick:()=>d(o),className:`px-3 py-1 rounded-full text-xs font-medium transition-colors ${s===o?"bg-violet-500 text-white":"bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400"}`,children:c.label},o))}),e.jsx("svg",{width:300,height:150,className:"mx-auto block",children:l.map((o,c)=>e.jsx("circle",{cx:o.x,cy:o.y,r:4,fill:"#8b5cf6",opacity:.7},c))}),e.jsx("p",{className:"text-xs text-gray-500 text-center mt-1",children:a.desc})]})}function N(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"The central challenge in self-supervised learning is representational collapse: the model learns to map all inputs to the same representation, achieving zero loss trivially. Multiple strategies have been developed to prevent this failure mode."}),e.jsxs(x,{title:"Representational Collapse",children:[e.jsx("p",{children:"Collapse occurs when the encoder produces constant or low-rank representations:"}),e.jsx(t.BlockMath,{math:"f_\\theta(\\mathbf{x}) \\approx \\mathbf{c} \\quad \\forall\\, \\mathbf{x} \\in \\mathcal{X}"}),e.jsxs("p",{className:"mt-2",children:["This is a trivial solution to many self-supervised objectives. For instance, with a similarity loss ",e.jsx(t.InlineMath,{math:"\\mathcal{L} = -\\text{sim}(f(\\mathbf{x}), f(\\mathbf{x}'))"}),", constant output gives perfect similarity with zero learning."]})]}),e.jsx(k,{}),e.jsx(b,{title:"Dimensional Collapse",children:e.jsxs("p",{children:["Even without full collapse, ",e.jsx("strong",{children:"dimensional collapse"})," can occur: representations occupy a low-dimensional subspace of the embedding space. The effective rank of the representation matrix drops, wasting capacity. This is harder to detect than full collapse."]})}),e.jsx(g,{title:"Four Strategies to Prevent Collapse",children:e.jsxs("p",{className:"space-y-2",children:[e.jsx("strong",{children:"1. Contrastive (negative pairs):"})," Push apart representations of different images. SimCLR, MoCo use large sets of negatives.",e.jsx("br",{}),e.jsx("strong",{children:"2. Momentum encoder:"})," BYOL, MoCo use a slowly-updated target network, preventing the representations from changing too rapidly.",e.jsx("br",{}),e.jsx("strong",{children:"3. Variance/covariance regularization:"})," VICReg explicitly penalizes low variance and high covariance between embedding dimensions.",e.jsx("br",{}),e.jsx("strong",{children:"4. Centering + sharpening:"})," DINO centers the teacher output and sharpens it, preventing any single dimension from dominating."]})}),e.jsx(u,{title:"VICReg Loss: Variance-Invariance-Covariance",code:`import torch
import torch.nn.functional as F

def vicreg_loss(z1, z2, lam=25.0, mu=25.0, nu=1.0):
    """VICReg: Variance-Invariance-Covariance Regularization."""
    B, D = z1.shape

    # Invariance: MSE between positive pairs
    inv_loss = F.mse_loss(z1, z2)

    # Variance: std of each dimension should be >= 1
    std_z1 = z1.std(dim=0)
    std_z2 = z2.std(dim=0)
    var_loss = (F.relu(1 - std_z1).mean() + F.relu(1 - std_z2).mean())

    # Covariance: off-diagonal elements should be 0
    z1_centered = z1 - z1.mean(dim=0)
    z2_centered = z2 - z2.mean(dim=0)
    cov1 = (z1_centered.T @ z1_centered) / (B - 1)
    cov2 = (z2_centered.T @ z2_centered) / (B - 1)
    # Zero out diagonal, penalize off-diagonal
    mask = ~torch.eye(D, dtype=bool)
    cov_loss = (cov1[mask].pow(2).mean() + cov2[mask].pow(2).mean())

    return lam * inv_loss + mu * var_loss + nu * cov_loss

z1 = torch.randn(64, 128)
z2 = z1 + 0.1 * torch.randn_like(z1)  # positive pairs
loss = vicreg_loss(z1, z2)
print(f"VICReg loss: {loss.item():.3f}")
print(f"Embedding std: {z1.std(dim=0).mean():.3f} (target >= 1.0)")`}),e.jsx(p,{type:"note",title:"Barlow Twins: Redundancy Reduction",children:e.jsxs("p",{children:["Barlow Twins takes a complementary approach: the cross-correlation matrix between two augmented views should be the identity matrix. This simultaneously prevents collapse (diagonal elements = 1) and redundancy (off-diagonal = 0):",e.jsx(t.InlineMath,{math:"\\mathcal{L} = \\sum_i (1 - C_{ii})^2 + \\lambda \\sum_{i \\neq j} C_{ij}^2"}),"."]})})]})}const H=Object.freeze(Object.defineProperty({__proto__:null,default:N},Symbol.toStringTag,{value:"Module"}));function w(){const[s,d]=m.useState(.5),r=[-.5,0,.3,.7,.9,1],a=r.map(i=>Math.exp(i/s)),l=a.reduce((i,n)=>i+n,0),o=a.map(i=>i/l),c=Math.max(...o);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Temperature Effect on NT-Xent"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["tau = ",s.toFixed(2),e.jsx("input",{type:"range",min:.05,max:2,step:.05,value:s,onChange:i=>d(parseFloat(i.target.value)),className:"w-40 accent-violet-500"})]}),e.jsx("div",{className:"flex gap-2 items-end justify-center h-28",children:r.map((i,n)=>e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("div",{className:`w-10 rounded-t transition-all ${n===r.length-1?"bg-violet-500":"bg-violet-300"}`,style:{height:`${o[n]/c*90}px`}}),e.jsxs("span",{className:"text-[9px] text-gray-500 mt-1",children:["sim=",i]}),e.jsx("span",{className:"text-[9px] text-violet-600",children:o[n].toFixed(3)})]},n))}),e.jsx("p",{className:"text-xs text-gray-500 text-center mt-1",children:s<.2?"Very low tau: only the highest similarity matters (hard)":s<.7?"Moderate tau: good discrimination between similarities":"High tau: similarities become nearly uniform (too soft)"})]})}function M(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"SimCLR (Simple Contrastive Learning of Representations) is a foundational contrastive learning framework. It learns representations by maximizing agreement between differently augmented views of the same image while pushing apart representations of different images."}),e.jsxs(x,{title:"NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)",children:[e.jsxs("p",{children:["For a positive pair ",e.jsx(t.InlineMath,{math:"(i, j)"})," within a batch of ",e.jsx(t.InlineMath,{math:"2N"})," augmented samples:"]}),e.jsx(t.BlockMath,{math:"\\ell_{i,j} = -\\log \\frac{\\exp(\\text{sim}(\\mathbf{z}_i, \\mathbf{z}_j) / \\tau)}{\\sum_{k=1}^{2N} \\mathbb{1}_{k \\neq i}\\, \\exp(\\text{sim}(\\mathbf{z}_i, \\mathbf{z}_k) / \\tau)}"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"\\text{sim}(\\mathbf{u}, \\mathbf{v}) = \\mathbf{u}^\\top \\mathbf{v} / (\\|\\mathbf{u}\\|\\|\\mathbf{v}\\|)"})," is cosine similarity and ",e.jsx(t.InlineMath,{math:"\\tau"})," is the temperature."]})]}),e.jsx(w,{}),e.jsxs(f,{title:"SimCLR Framework",id:"simclr-framework",children:[e.jsx("p",{children:"The four components of SimCLR:"}),e.jsxs("ol",{className:"list-decimal ml-5 mt-2 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Augmentations"})," ",e.jsx(t.InlineMath,{math:"T"}),": Random crop + color jitter + Gaussian blur"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Encoder"})," ",e.jsx(t.InlineMath,{math:"f"}),": ResNet backbone producing ",e.jsx(t.InlineMath,{math:"\\mathbf{h} = f(\\tilde{\\mathbf{x}})"})]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Projection head"})," ",e.jsx(t.InlineMath,{math:"g"}),": MLP mapping ",e.jsx(t.InlineMath,{math:"\\mathbf{z} = g(\\mathbf{h})"})," (discarded after training)"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"NT-Xent loss"}),": Contrastive objective on the projected representations"]})]})]}),e.jsx(u,{title:"SimCLR Loss Implementation",code:`import torch
import torch.nn.functional as F

def simclr_loss(z1, z2, temperature=0.5):
    """NT-Xent loss for SimCLR.
    z1, z2: (B, D) embeddings of two augmented views.
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    z = F.normalize(z, dim=-1)

    # Cosine similarity matrix (2B x 2B)
    sim = z @ z.T / temperature

    # Mask out self-similarity
    mask = ~torch.eye(2 * B, dtype=bool, device=z.device)
    sim = sim.masked_fill(~mask, -1e9)

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(B, 2 * B),
        torch.arange(0, B)
    ], dim=0).to(z.device)

    loss = F.cross_entropy(sim, labels)
    return loss

# Example usage
z1 = F.normalize(torch.randn(256, 128), dim=-1)
z2 = F.normalize(torch.randn(256, 128), dim=-1)
loss = simclr_loss(z1, z2, temperature=0.5)
print(f"SimCLR loss (B=256): {loss.item():.3f}")
print(f"Random baseline: {torch.log(torch.tensor(2*256-1.0)):.3f}")`}),e.jsx(g,{title:"Why Large Batches Matter",children:e.jsx("p",{children:"SimCLR uses negatives from within the batch. Larger batches provide more negatives, creating a harder contrastive task. SimCLR v1 used batch size 4096 (8192 augmented views); performance degrades significantly below 256. This is a key limitation addressed by MoCo."})}),e.jsx(p,{type:"note",title:"The Projection Head Is Critical",children:e.jsxs("p",{children:["Representations before the projection head (",e.jsx(t.InlineMath,{math:"\\mathbf{h}"}),") transfer better than those after it (",e.jsx(t.InlineMath,{math:"\\mathbf{z}"}),"). The projection head discards information useful for downstream tasks but irrelevant to the contrastive objective (e.g., color information lost to augmentation). Always evaluate on ",e.jsx(t.InlineMath,{math:"\\mathbf{h}"}),", not ",e.jsx(t.InlineMath,{math:"\\mathbf{z}"}),"."]})})]})}const Q=Object.freeze(Object.defineProperty({__proto__:null,default:M},Symbol.toStringTag,{value:"Module"}));function z(){const[s,d]=m.useState(.999),[r,a]=m.useState(65536),l=Math.log(.5)/Math.log(s);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"MoCo: Momentum & Queue"}),e.jsxs("div",{className:"flex gap-4 mb-3 flex-wrap",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["momentum: ",s.toFixed(3),e.jsx("input",{type:"range",min:.9,max:.9999,step:.001,value:s,onChange:o=>d(parseFloat(o.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["queue: ",r.toLocaleString(),e.jsx("input",{type:"range",min:4096,max:131072,step:4096,value:r,onChange:o=>a(parseInt(o.target.value)),className:"w-28 accent-violet-500"})]})]}),e.jsxs("div",{className:"flex gap-8 justify-center items-center",children:[e.jsxs("div",{className:"text-center",children:[e.jsxs("div",{className:"w-16 h-16 rounded-lg bg-violet-500 flex items-center justify-center text-white text-xs font-bold",children:["Query",e.jsx("br",{}),"Encoder"]}),e.jsx("p",{className:"text-[9px] text-gray-500 mt-1",children:"gradient update"})]}),e.jsx("div",{className:"text-violet-400 text-lg",children:"→"}),e.jsxs("div",{className:"text-center",children:[e.jsxs("div",{className:"w-16 h-16 rounded-lg bg-violet-300 flex items-center justify-center text-white text-xs font-bold",children:["Key",e.jsx("br",{}),"Encoder"]}),e.jsxs("p",{className:"text-[9px] text-gray-500 mt-1",children:["m=",s.toFixed(3)]})]}),e.jsx("div",{className:"text-violet-400 text-lg",children:"→"}),e.jsxs("div",{className:"text-center",children:[e.jsxs("div",{className:"w-20 h-16 rounded-lg bg-orange-300 flex items-center justify-center text-white text-xs font-bold",children:["Queue",e.jsx("br",{}),(r/1024).toFixed(0),"K keys"]}),e.jsx("p",{className:"text-[9px] text-gray-500 mt-1",children:"FIFO negatives"})]})]}),e.jsxs("p",{className:"text-xs text-gray-500 text-center mt-2",children:["Half-life: ~",l.toFixed(0)," steps | Effective negatives: ",r.toLocaleString()," (vs batch size for SimCLR)"]})]})}function T(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Momentum Contrast (MoCo) decouples the number of negatives from batch size by maintaining a queue of encoded keys. A momentum-updated encoder ensures the keys are consistent, enabling contrastive learning with standard batch sizes."}),e.jsxs(x,{title:"MoCo Framework",children:[e.jsxs("p",{children:["MoCo maintains a query encoder ",e.jsx(t.InlineMath,{math:"f_q"})," and a momentum key encoder ",e.jsx(t.InlineMath,{math:"f_k"}),":"]}),e.jsx(t.BlockMath,{math:"\\theta_k \\leftarrow m \\cdot \\theta_k + (1 - m) \\cdot \\theta_q"}),e.jsx("p",{className:"mt-2",children:"The InfoNCE loss with queue negatives:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_q = -\\log \\frac{\\exp(\\mathbf{q} \\cdot \\mathbf{k}_+ / \\tau)}{\\exp(\\mathbf{q} \\cdot \\mathbf{k}_+ / \\tau) + \\sum_{i=0}^{K} \\exp(\\mathbf{q} \\cdot \\mathbf{k}_i / \\tau)}"}),e.jsxs("p",{className:"mt-1",children:["where ",e.jsx(t.InlineMath,{math:"\\mathbf{k}_+"})," is the positive key and ",e.jsx(t.InlineMath,{math:"\\mathbf{k}_i"})," are queue negatives."]})]}),e.jsx(z,{}),e.jsx(g,{title:"MoCo v1 to v3 Evolution",children:e.jsxs("p",{children:[e.jsx("strong",{children:"MoCo v1"}),": Queue + momentum encoder with ResNet.",e.jsx("strong",{children:"MoCo v2"}),": Adds SimCLR's MLP projection head and stronger augmentations.",e.jsx("strong",{children:"MoCo v3"}),": Adapts to Vision Transformers, removes the queue (uses batch negatives like SimCLR but with momentum encoder for stability), and adds a prediction head."]})}),e.jsx(u,{title:"MoCo v2 Core Implementation",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class MoCo(nn.Module):
    def __init__(self, encoder, dim=128, K=65536, m=0.999, tau=0.2):
        super().__init__()
        self.K, self.m, self.tau = K, m, tau

        self.encoder_q = encoder  # query encoder (gradient)
        self.encoder_k = type(encoder)()  # key encoder (momentum)

        # Initialize key encoder = query encoder
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False

        # Queue of negative keys
        self.register_buffer("queue", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self):
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = self.m * p_k.data + (1 - self.m) * p_q.data

    @torch.no_grad()
    def enqueue(self, keys):
        B = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + B] = keys.T
        self.queue_ptr[0] = (ptr + B) % self.K

    def forward(self, x_q, x_k):
        q = F.normalize(self.encoder_q(x_q), dim=-1)

        with torch.no_grad():
            self.momentum_update()
            k = F.normalize(self.encoder_k(x_k), dim=-1)

        # Positive logits: (B, 1)
        l_pos = torch.einsum('bd,bd->b', q, k).unsqueeze(-1) / self.tau
        # Negative logits: (B, K)
        l_neg = torch.einsum('bd,dk->bk', q, self.queue.clone().detach()) / self.tau

        logits = torch.cat([l_pos, l_neg], dim=-1)
        labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)

        self.enqueue(k)
        return F.cross_entropy(logits, labels)

print("MoCo: 65K negatives with batch size 256")
print("Key insight: momentum encoder keeps queue keys consistent")`}),e.jsx(p,{type:"note",title:"MoCo vs SimCLR Trade-offs",children:e.jsx("p",{children:"SimCLR requires large batch sizes (4096+) and large GPU memory. MoCo achieves comparable results with batch size 256 by maintaining a large queue. However, MoCo's momentum encoder adds complexity. In practice, MoCo v3 and DINO (momentum-based) have proven more effective for Vision Transformers than pure SimCLR-style approaches."})})]})}const G=Object.freeze(Object.defineProperty({__proto__:null,default:T},Symbol.toStringTag,{value:"Module"}));function I(){const[s,d]=m.useState("byol"),r={byol:{name:"BYOL",negatives:"No",momentum:"Yes",predictor:"Yes",key:"Asymmetric architecture + momentum prevents collapse"},simsiam:{name:"SimSiam",negatives:"No",momentum:"No",predictor:"Yes",key:"Stop-gradient alone prevents collapse (surprisingly)"},vicreg:{name:"VICReg",negatives:"No",momentum:"No",predictor:"No",key:"Variance + covariance regularization prevents collapse"},barlow:{name:"Barlow Twins",negatives:"No",momentum:"No",predictor:"No",key:"Cross-correlation matrix should equal identity"}},a=r[s];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-2 text-base font-bold text-gray-800 dark:text-gray-200",children:"Non-Contrastive Methods"}),e.jsx("div",{className:"flex gap-2 mb-3 flex-wrap",children:Object.entries(r).map(([l,o])=>e.jsx("button",{onClick:()=>d(l),className:`px-3 py-1 rounded-full text-xs font-medium transition-colors ${s===l?"bg-violet-500 text-white":"bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400"}`,children:o.name},l))}),e.jsxs("div",{className:"grid grid-cols-3 gap-2 text-xs bg-gray-50 dark:bg-gray-800 rounded-lg p-3",children:[e.jsxs("div",{children:[e.jsx("span",{className:"text-gray-500",children:"Negatives:"})," ",e.jsx("span",{className:`font-medium ${a.negatives==="No"?"text-violet-600":""}`,children:a.negatives})]}),e.jsxs("div",{children:[e.jsx("span",{className:"text-gray-500",children:"Momentum:"})," ",e.jsx("span",{className:"font-medium",children:a.momentum})]}),e.jsxs("div",{children:[e.jsx("span",{className:"text-gray-500",children:"Predictor:"})," ",e.jsx("span",{className:"font-medium",children:a.predictor})]})]}),e.jsx("p",{className:"text-xs text-violet-600 mt-2 text-center font-medium",children:a.key})]})}function C(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Non-contrastive methods learn representations without negative pairs. BYOL showed that a momentum encoder with an asymmetric predictor suffices, while VICReg uses explicit regularization of the embedding statistics."}),e.jsxs(x,{title:"BYOL (Bootstrap Your Own Latent)",children:[e.jsxs("p",{children:["BYOL uses an online network ",e.jsx(t.InlineMath,{math:"(\\theta)"})," with a predictor and a target network",e.jsx(t.InlineMath,{math:"(\\xi)"})," updated via momentum:"]}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_{\\text{BYOL}} = \\|\\bar{q}_\\theta(\\mathbf{z}_\\theta) - \\bar{\\mathbf{z}}_\\xi\\|^2"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"\\bar{\\cdot}"})," denotes L2 normalization, ",e.jsx(t.InlineMath,{math:"q_\\theta"})," is the predictor, and the loss is symmetrized over both views. No negatives needed."]})]}),e.jsx(I,{}),e.jsxs(g,{title:"SimSiam: Simplicity Is All You Need",children:[e.jsx("p",{children:"SimSiam removes even the momentum encoder. The key insight is that stop-gradient on the target branch creates an implicit moving average:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = -\\frac{1}{2}\\left[\\text{sim}(p_1, \\text{sg}(z_2)) + \\text{sim}(p_2, \\text{sg}(z_1))\\right]"}),e.jsxs("p",{className:"mt-1",children:["where ",e.jsx(t.InlineMath,{math:"\\text{sg}"})," is stop-gradient. Without it, instant collapse occurs."]})]}),e.jsx(u,{title:"BYOL Implementation",code:`import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class BYOL(nn.Module):
    def __init__(self, encoder, proj_dim=256, pred_dim=128, momentum=0.996):
        super().__init__()
        self.momentum = momentum

        # Online network
        self.encoder = encoder
        feat_dim = 512  # encoder output dim
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim), nn.BatchNorm1d(pred_dim), nn.ReLU(),
            nn.Linear(pred_dim, proj_dim),
        )

        # Target network (no gradients)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_projector = copy.deepcopy(self.projector)
        for p in list(self.target_encoder.parameters()) + list(self.target_projector.parameters()):
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self):
        for po, pt in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            pt.data = self.momentum * pt.data + (1 - self.momentum) * po.data
        for po, pt in zip(self.projector.parameters(), self.target_projector.parameters()):
            pt.data = self.momentum * pt.data + (1 - self.momentum) * po.data

    def forward(self, x1, x2):
        # Online predictions
        p1 = self.predictor(self.projector(self.encoder(x1)))
        p2 = self.predictor(self.projector(self.encoder(x2)))

        with torch.no_grad():
            self.update_target()
            t1 = self.target_projector(self.target_encoder(x1))
            t2 = self.target_projector(self.target_encoder(x2))

        loss = (F.cosine_similarity(p1, t2.detach(), dim=-1).mean()
              + F.cosine_similarity(p2, t1.detach(), dim=-1).mean())
        return -loss  # maximize cosine similarity

print("BYOL: No negatives, no large batches needed")
print("Key: predictor + momentum encoder = implicit regularization")`}),e.jsx(b,{title:"Batch Normalization in BYOL",children:e.jsx("p",{children:"Early analysis suggested BYOL's success depended critically on batch normalization in the projector, which implicitly provides negative-like information across the batch. Later work showed that BYOL can work without BN if the predictor is properly initialized and the learning rate is carefully tuned."})}),e.jsx(p,{type:"note",title:"Which Method to Choose?",children:e.jsx("p",{children:"For Vision Transformers: DINO/DINOv2 (momentum + self-distillation) dominates. For CNNs with limited compute: VICReg or Barlow Twins (simple, no momentum). For large-scale with TPUs: SimCLR v2 remains competitive with large batches. All methods achieve similar linear probe accuracy on ImageNet (~75% with ResNet-50)."})})]})}const J=Object.freeze(Object.defineProperty({__proto__:null,default:C},Symbol.toStringTag,{value:"Module"}));function S(){const[s,d]=m.useState(.75),r=14,a=r*r,l=Math.round(a*s),[o]=m.useState(()=>{const i=Array.from({length:a},(n,h)=>h);for(let n=i.length-1;n>0;n--){const h=Math.floor(Math.random()*(n+1));[i[n],i[h]]=[i[h],i[n]]}return i}),c=new Set(o.slice(0,l));return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"MAE: Masked Image Patches"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Mask ratio: ",(s*100).toFixed(0),"%",e.jsx("input",{type:"range",min:.5,max:.9,step:.05,value:s,onChange:i=>d(parseFloat(i.target.value)),className:"w-40 accent-violet-500"}),e.jsxs("span",{className:"text-xs",children:["(",a-l," visible / ",a," total)"]})]}),e.jsx("div",{className:"flex justify-center",children:e.jsx("div",{className:"grid gap-px",style:{gridTemplateColumns:`repeat(${r}, 1fr)`},children:Array.from({length:a},(i,n)=>e.jsx("div",{className:`w-3.5 h-3.5 ${c.has(n)?"bg-gray-200 dark:bg-gray-700":"bg-violet-400"}`},n))})}),e.jsxs("p",{className:"text-xs text-gray-500 text-center mt-2",children:["Only ",((1-s)*100).toFixed(0),"% of patches go through the encoder (huge compute savings)"]})]})}function B(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Masked Autoencoders (MAE) adapt the masked language modeling paradigm (BERT) to vision. By masking a very high proportion of image patches (75%) and reconstructing them, MAE learns powerful visual representations with remarkable compute efficiency."}),e.jsxs(x,{title:"MAE Architecture",children:[e.jsx("p",{children:"MAE consists of an asymmetric encoder-decoder:"}),e.jsxs("ul",{className:"list-disc ml-5 mt-2 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Encoder"})," (ViT): processes only ",e.jsx("em",{children:"visible"})," patches (~25%), making it very fast"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Decoder"})," (lightweight): processes full set of tokens (visible + mask tokens)"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Reconstruction target"}),": per-patch normalized pixel values"]})]}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_{\\text{MAE}} = \\frac{1}{|\\mathcal{M}|}\\sum_{i \\in \\mathcal{M}} \\| \\hat{\\mathbf{p}}_i - \\mathbf{p}_i \\|^2"}),e.jsxs("p",{className:"mt-1",children:["Loss computed only on masked patches ",e.jsx(t.InlineMath,{math:"\\mathcal{M}"}),"."]})]}),e.jsx(S,{}),e.jsxs(f,{title:"Why 75% Masking Works",id:"high-masking",children:[e.jsx("p",{children:"Images have high spatial redundancy: neighboring patches are highly correlated. Low masking ratios allow the model to interpolate from nearby visible patches without learning semantics. At 75%, the task becomes truly challenging:"}),e.jsx(t.BlockMath,{math:"I(\\mathbf{x}_{\\text{visible}}; \\mathbf{x}_{\\text{masked}}) \\ll I(\\mathbf{x}; \\mathbf{x})"}),e.jsx("p",{className:"mt-2",children:"The visible patches provide insufficient local information, forcing holistic understanding. This contrasts with BERT's 15% masking, since text tokens carry more information per token."})]}),e.jsx(u,{title:"MAE Core: Masking and Reconstruction",code:`import torch
import torch.nn as nn

class MAEEncoder(nn.Module):
    def __init__(self, num_patches=196, embed_dim=768, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_embed = nn.Linear(768, embed_dim)  # patch projection
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        # Transformer blocks would go here

    def random_masking(self, x):
        B, N, D = x.shape
        keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_keep = ids_shuffle[:, :keep]

        # Keep only visible patches
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        return x_visible, ids_keep, ids_shuffle

    def forward(self, x):
        # x: (B, num_patches, patch_dim)
        x = self.patch_embed(x) + self.pos_embed
        x_visible, ids_keep, ids_shuffle = self.random_masking(x)
        # Pass x_visible through transformer (not shown)
        return x_visible, ids_keep, ids_shuffle

# Efficiency: encoder processes only 25% of patches
encoder = MAEEncoder(num_patches=196, mask_ratio=0.75)
patches = torch.randn(4, 196, 768)
visible, keep_ids, _ = encoder(patches)
print(f"Input: {patches.shape[1]} patches")
print(f"Encoder processes: {visible.shape[1]} patches ({visible.shape[1]/196*100:.0f}%)")
print(f"3-4x faster than processing all patches!")`}),e.jsx(g,{title:"MAE Pre-training Results",children:e.jsx("p",{children:"MAE pre-trained ViT-Large achieves 85.9% top-1 on ImageNet with fine-tuning, surpassing supervised pre-training. The decoder is discarded after pre-training, and only the encoder is used for downstream tasks. MAE is particularly effective for larger models (ViT-Huge: 86.9%) where labeled data is insufficient."})}),e.jsx(p,{type:"note",title:"Pixel vs Feature Reconstruction",children:e.jsx("p",{children:"MAE reconstructs normalized pixels, which works surprisingly well despite the concern that pixel prediction emphasizes low-level details. The high masking ratio forces semantic understanding regardless. BEiT (next section) reconstructs discrete visual tokens instead, providing a different inductive bias."})})]})}const X=Object.freeze(Object.defineProperty({__proto__:null,default:B},Symbol.toStringTag,{value:"Module"}));function L(){const[s,d]=m.useState(8192),r=8,a=r*r,l=Array.from({length:a},(c,i)=>Math.floor(Math.sin(i*7.3+2.1)*s/2+s/2)%s),o=new Set([5,12,18,23,30,37,42,48,51,55,60]);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Visual Tokenization"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Codebook size: ",s.toLocaleString(),e.jsx("input",{type:"range",min:512,max:16384,step:512,value:s,onChange:c=>d(parseInt(c.target.value)),className:"w-40 accent-violet-500"})]}),e.jsx("div",{className:"flex justify-center",children:e.jsx("div",{className:"grid gap-0.5",style:{gridTemplateColumns:`repeat(${r}, 1fr)`},children:Array.from({length:a},(c,i)=>{const n=o.has(i),h=l[i]/s*270;return e.jsx("div",{className:"w-7 h-7 rounded-sm flex items-center justify-center text-[7px] text-white font-mono",style:{backgroundColor:n?"#9ca3af":`hsl(${h}, 60%, 55%)`},children:n?"?":l[i]},i)})})}),e.jsx("p",{className:"text-xs text-gray-500 text-center mt-2",children:"Each patch is mapped to a discrete token ID. Masked patches (gray) must be predicted."})]})}function O(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"BEiT (Bidirectional Encoder representation from Image Transformers) adapts BERT-style pre-training to vision by predicting discrete visual tokens for masked patches, rather than raw pixels. This provides a higher-level reconstruction target."}),e.jsxs(x,{title:"BEiT Pre-training",children:[e.jsx("p",{children:"BEiT uses a two-stage approach:"}),e.jsxs("ol",{className:"list-decimal ml-5 mt-2 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Stage 1"}),": Train a discrete VAE (dVAE) tokenizer to map image patches to visual tokens ",e.jsx(t.InlineMath,{math:"v_i \\in \\{1, \\ldots, V\\}"})]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Stage 2"}),": Mask patches and predict their token IDs via cross-entropy"]})]}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_{\\text{BEiT}} = -\\sum_{i \\in \\mathcal{M}} \\log p_\\theta(v_i | \\mathbf{x}_{\\setminus \\mathcal{M}})"}),e.jsxs("p",{className:"mt-1",children:["where ",e.jsx(t.InlineMath,{math:"v_i"})," is the visual token for patch ",e.jsx(t.InlineMath,{math:"i"})," and",e.jsx(t.InlineMath,{math:"\\mathcal{M}"})," is the set of masked positions."]})]}),e.jsx(L,{}),e.jsxs(g,{title:"BEiT v2: Semantic Visual Tokens",children:[e.jsx("p",{children:"BEiT v2 replaces the dVAE tokenizer with a vector-quantized knowledge distillation (VQ-KD) tokenizer trained using a CLIP teacher. This produces semantically meaningful tokens where similar visual concepts share the same token ID, significantly improving the pre-training signal."}),e.jsx(t.BlockMath,{math:"\\text{BEiT v2 tokenizer: } v_i = \\text{VQ}(\\text{CLIP}_{\\text{visual}}(\\text{patch}_i))"})]}),e.jsx(u,{title:"BEiT-style Masked Image Modeling",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class BEiTPretraining(nn.Module):
    def __init__(self, num_patches=196, embed_dim=768, vocab_size=8192):
        super().__init__()
        self.vocab_size = vocab_size
        self.patch_embed = nn.Linear(768, embed_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        # Transformer encoder (simplified)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=12, dim_feedforward=3072, batch_first=True),
            num_layers=2,  # use 12+ in practice
        )
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, patches, mask, target_tokens):
        # patches: (B, N, D), mask: (B, N) bool, target_tokens: (B, N) long
        x = self.patch_embed(patches) + self.pos_embed

        # Replace masked patches with learnable mask token
        mask_expanded = mask.unsqueeze(-1).expand_as(x)
        x = torch.where(mask_expanded, self.mask_token.expand_as(x), x)

        x = self.transformer(x)

        # Predict tokens only for masked positions
        logits = self.head(x)  # (B, N, vocab_size)
        masked_logits = logits[mask]  # (num_masked, vocab_size)
        masked_targets = target_tokens[mask]  # (num_masked,)

        loss = F.cross_entropy(masked_logits, masked_targets)
        return loss

model = BEiTPretraining(num_patches=196, vocab_size=8192)
patches = torch.randn(4, 196, 768)
mask = torch.rand(4, 196) > 0.6  # ~40% masking
tokens = torch.randint(0, 8192, (4, 196))
loss = model(patches, mask, tokens)
print(f"BEiT loss: {loss.item():.3f}")
print(f"Random baseline: {torch.log(torch.tensor(8192.0)):.3f}")`}),e.jsx(b,{title:"Tokenizer Quality Matters",children:e.jsx("p",{children:"BEiT's performance depends heavily on the quality of the visual tokenizer. A poor tokenizer produces noisy tokens that make the prediction task ill-defined. BEiT v2's CLIP-based tokenizer outperforms BEiT v1's dVAE by providing semantically richer targets."})}),e.jsx(p,{type:"note",title:"BEiT vs MAE",children:e.jsx("p",{children:"BEiT predicts discrete tokens (classification); MAE predicts pixels (regression). BEiT typically shows stronger linear probe performance (the representations are more semantic), while MAE excels at fine-tuning. BEiT requires pre-training a tokenizer; MAE needs no extra components. Both achieve similar final fine-tuning accuracy."})})]})}const Z=Object.freeze(Object.defineProperty({__proto__:null,default:O},Symbol.toStringTag,{value:"Module"}));function E(){const[s,d]=m.useState("vision"),r={vision:{input:"Image patches",masking:"40-75% random patches",target:"Teacher ViT features",examples:"ImageNet, COCO"},speech:{input:"Audio waveform frames",masking:"Contiguous spans of frames",target:"Teacher wav2vec features",examples:"LibriSpeech"},text:{input:"Subword tokens",masking:"15% random tokens (BERT-style)",target:"Teacher Transformer features",examples:"Books, Wikipedia"}},a=r[s];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-2 text-base font-bold text-gray-800 dark:text-gray-200",children:"data2vec: Unified Across Modalities"}),e.jsx("div",{className:"flex gap-2 mb-3",children:Object.keys(r).map(l=>e.jsx("button",{onClick:()=>d(l),className:`px-3 py-1 rounded-full text-xs font-medium capitalize transition-colors ${s===l?"bg-violet-500 text-white":"bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400"}`,children:l},l))}),e.jsxs("div",{className:"grid grid-cols-2 gap-2 text-xs bg-gray-50 dark:bg-gray-800 rounded-lg p-3",children:[e.jsxs("div",{children:[e.jsx("span",{className:"text-gray-500",children:"Input:"})," ",e.jsx("span",{className:"font-medium",children:a.input})]}),e.jsxs("div",{children:[e.jsx("span",{className:"text-gray-500",children:"Masking:"})," ",e.jsx("span",{className:"font-medium",children:a.masking})]}),e.jsxs("div",{children:[e.jsx("span",{className:"text-gray-500",children:"Target:"})," ",e.jsx("span",{className:"font-medium text-violet-600",children:a.target})]}),e.jsxs("div",{children:[e.jsx("span",{className:"text-gray-500",children:"Data:"})," ",e.jsx("span",{className:"font-medium",children:a.examples})]})]}),e.jsx("p",{className:"text-xs text-violet-600 text-center mt-2 font-medium",children:"Same algorithm, same objective, same architecture backbone across all modalities"})]})}function P(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"data2vec provides a unified self-supervised learning framework across vision, speech, and text. Instead of predicting modality-specific targets (pixels, tokens, waveforms), it predicts contextualized representations from a teacher network."}),e.jsxs(x,{title:"data2vec Objective",children:[e.jsx("p",{children:"A student model with masked input predicts the representations from an EMA teacher that sees the full unmasked input:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = \\frac{1}{|\\mathcal{M}|}\\sum_{i \\in \\mathcal{M}} \\left\\| f_\\theta^{\\text{student}}(\\tilde{\\mathbf{x}})_i - \\bar{f}_\\xi^{\\text{teacher}}(\\mathbf{x})_i \\right\\|^2"}),e.jsxs("p",{className:"mt-2",children:["The teacher target ",e.jsx(t.InlineMath,{math:"\\bar{f}_\\xi"})," is the average of the top ",e.jsx(t.InlineMath,{math:"K"})," transformer layers, followed by instance normalization. The teacher is updated via EMA:",e.jsx(t.InlineMath,{math:"\\xi \\leftarrow \\tau \\xi + (1-\\tau)\\theta"}),"."]})]}),e.jsx(E,{}),e.jsxs(f,{title:"Why Predict Representations?",id:"pred-repr",children:[e.jsx("p",{children:"Teacher representations capture contextual information from the full input:"}),e.jsx(t.BlockMath,{math:"\\mathbf{y}_i = \\text{Normalize}\\left(\\frac{1}{K}\\sum_{l=L-K+1}^{L} \\mathbf{h}_i^{(l)}\\right)"}),e.jsx("p",{className:"mt-2",children:"Unlike pixel/token targets, these representations are: (1) inherently high-level and semantic, (2) context-dependent (same patch has different targets in different images), and (3) modality-agnostic in their loss formulation."})]}),e.jsx(g,{title:"data2vec 2.0: Efficiency Improvements",children:e.jsx("p",{children:"data2vec 2.0 introduces several efficiency improvements: (1) the teacher processes each sample only once and caches targets, (2) multi-mask training applies multiple different masks to the same teacher encoding, and (3) convolutional decoders replace transformer decoders. This yields 2-16x speedups over data2vec 1.0."})}),e.jsx(u,{title:"data2vec Core: Teacher Targets and Student Loss",code:`import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Data2Vec(nn.Module):
    def __init__(self, encoder, num_layers=12, top_k=8, tau=0.999):
        super().__init__()
        self.student = encoder
        self.teacher = copy.deepcopy(encoder)
        self.top_k = top_k
        self.tau = tau

        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self):
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data = self.tau * pt.data + (1 - self.tau) * ps.data

    @torch.no_grad()
    def get_teacher_targets(self, x):
        # Get representations from top-K layers of teacher
        # (Simplified: in practice, hook into intermediate layers)
        teacher_out = self.teacher(x)  # assume returns list of layer outputs
        # Average top-K layers
        top_k_layers = teacher_out[-self.top_k:]
        target = torch.stack(top_k_layers).mean(dim=0)
        # Instance normalization
        target = F.layer_norm(target, target.shape[-1:])
        return target

    def forward(self, x, mask):
        self.update_teacher()
        targets = self.get_teacher_targets(x)  # (B, N, D)

        # Student sees masked input
        student_out = self.student(x, mask=mask)  # (B, N, D)

        # Loss only on masked positions
        loss = F.smooth_l1_loss(
            student_out[mask],
            targets[mask].detach(),
        )
        return loss

print("data2vec: predict teacher representations, not raw inputs")
print("Unified framework: same code for vision, speech, and text")
print("Key: EMA teacher + top-K layer averaging + instance norm")`}),e.jsx(p,{type:"note",title:"I-JEPA: Predicting in Representation Space",children:e.jsx("p",{children:"I-JEPA (Image Joint Embedding Predictive Architecture) by LeCun et al. extends the idea of predicting representations: a predictor network maps context patch embeddings to predict target patch embeddings, without pixel-level reconstruction. This avoids the bias toward low-level features and produces representations that excel at semantic tasks."})})]})}const ee=Object.freeze(Object.defineProperty({__proto__:null,default:P},Symbol.toStringTag,{value:"Module"}));function F(){const[s,d]=m.useState(.9),r=8,a=[.6,.1,.05,.02,.02,.01,.01,.19],l=a.map(n=>n*(1-s)+1/r*s),o=l.reduce((n,h)=>n+h,0),c=l.map(n=>n/o),i=Math.max(...c);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Centering + Sharpening Effect"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Centering strength: ",s.toFixed(2),e.jsx("input",{type:"range",min:0,max:.99,step:.01,value:s,onChange:n=>d(parseFloat(n.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("div",{className:"flex gap-3 justify-center",children:[e.jsxs("div",{children:[e.jsx("p",{className:"text-[10px] text-gray-500 text-center mb-1",children:"Without centering"}),e.jsx("div",{className:"flex gap-1 items-end h-20",children:a.map((n,h)=>e.jsx("div",{className:"w-5 bg-gray-300 rounded-t",style:{height:`${n/.6*70}px`}},h))})]}),e.jsxs("div",{children:[e.jsx("p",{className:"text-[10px] text-violet-500 text-center mb-1",children:"With centering"}),e.jsx("div",{className:"flex gap-1 items-end h-20",children:c.map((n,h)=>e.jsx("div",{className:"w-5 bg-violet-400 rounded-t",style:{height:`${n/i*70}px`}},h))})]})]}),e.jsx("p",{className:"text-xs text-gray-500 text-center mt-1",children:"Centering prevents one dimension from dominating (collapse to uniform = prevented)"})]})}function R(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"DINO (Self-Distillation with No Labels) learns visual features through self-distillation between student and teacher networks. Its attention maps exhibit remarkable emergent properties, capturing object boundaries without any segmentation supervision."}),e.jsxs(x,{title:"DINO Framework",children:[e.jsx("p",{children:"Student and teacher networks produce probability distributions via softmax with temperatures:"}),e.jsx(t.BlockMath,{math:"P_s(x)^{(i)} = \\frac{\\exp(g_{\\theta_s}(x)^{(i)} / \\tau_s)}{\\sum_k \\exp(g_{\\theta_s}(x)^{(k)} / \\tau_s)}"}),e.jsx("p",{className:"mt-2",children:"The loss minimizes cross-entropy between student and (centered, sharpened) teacher:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = -\\sum_{\\substack{x \\in \\{x_1^g, x_2^g\\} \\\\ x' \\neq x}} \\sum_i P_t(x)^{(i)} \\log P_s(x')^{(i)}"}),e.jsxs("p",{className:"mt-1",children:["Teacher uses lower temperature (",e.jsx(t.InlineMath,{math:"\\tau_t = 0.04"}),") for sharper outputs; student uses higher (",e.jsx(t.InlineMath,{math:"\\tau_s = 0.1"}),")."]})]}),e.jsx(F,{}),e.jsxs(f,{title:"Multi-Crop Strategy",id:"multi-crop",children:[e.jsx("p",{children:"DINO uses asymmetric crops to create a local-to-global correspondence:"}),e.jsxs("ul",{className:"list-disc ml-5 mt-2 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"2 global views"})," (224x224, covering >50% of image): processed by both student and teacher"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"N local views"})," (96x96, covering <50%): processed only by student"]})]}),e.jsx("p",{className:"mt-2",children:"The student must predict the teacher's global view output from local crops, encouraging learning of semantic features that generalize across spatial scales."})]}),e.jsx(u,{title:"DINO Loss with Centering",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        # student_output: list of (B, D) for each crop
        # teacher_output: list of (B, D) for global crops only
        s_out = [s / self.student_temp for s in student_output]
        t_out = [(t - self.center) / self.teacher_temp for t in teacher_output]

        # Softmax
        s_probs = [F.log_softmax(s, dim=-1) for s in s_out]
        t_probs = [F.softmax(t, dim=-1).detach() for t in t_out]

        # Cross-entropy: each student crop vs each teacher global crop
        total_loss = 0
        n_loss_terms = 0
        for t in t_probs:
            for s in s_probs:
                loss = -torch.sum(t * s, dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1

        # Update center with EMA
        with torch.no_grad():
            batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True)
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        return total_loss / n_loss_terms

criterion = DINOLoss(out_dim=256)
student_out = [torch.randn(32, 256) for _ in range(8)]  # 2 global + 6 local
teacher_out = [torch.randn(32, 256) for _ in range(2)]   # 2 global only
loss = criterion(student_out, teacher_out)
print(f"DINO loss: {loss.item():.3f}")`}),e.jsx(g,{title:"Emergent Object Segmentation",children:e.jsx("p",{children:"DINO ViT attention maps spontaneously learn to segment objects without any segmentation labels. The [CLS] token's self-attention in the last layer highlights foreground objects with sharp boundaries. This emergent property makes DINO features excellent for dense prediction tasks like semantic segmentation and object detection."})}),e.jsx(p,{type:"note",title:"Why Self-Distillation Works",children:e.jsx("p",{children:"The combination of centering (prevents collapse to uniform), sharpening (prevents collapse to one-hot), momentum teacher (provides stable targets), and multi-crop (creates difficulty asymmetry) creates a self-reinforcing learning signal. The teacher slowly improves, providing increasingly informative targets for the student."})})]})}const te=Object.freeze(Object.defineProperty({__proto__:null,default:R},Symbol.toStringTag,{value:"Module"}));function A(){const[s,d]=m.useState(2),a=[{name:"ViT-S/14",params:21,linear:79,knn:77.2},{name:"ViT-B/14",params:86,linear:82.1,knn:80.1},{name:"ViT-L/14",params:300,linear:83.5,knn:82},{name:"ViT-g/14",params:1100,linear:83.9,knn:82.8}][s],l=84;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"DINOv2 Scaling Behavior"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Model: ",a.name," (",a.params,"M params)",e.jsx("input",{type:"range",min:0,max:3,step:1,value:s,onChange:o=>d(parseInt(o.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("div",{className:"flex gap-4 justify-center items-end h-28",children:[e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("div",{className:"w-14 bg-violet-500 rounded-t transition-all",style:{height:`${a.linear/l*90}px`}}),e.jsx("span",{className:"text-xs text-gray-500 mt-1",children:"Linear"}),e.jsxs("span",{className:"text-xs text-violet-600 font-semibold",children:[a.linear,"%"]})]}),e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("div",{className:"w-14 bg-violet-300 rounded-t transition-all",style:{height:`${a.knn/l*90}px`}}),e.jsx("span",{className:"text-xs text-gray-500 mt-1",children:"k-NN"}),e.jsxs("span",{className:"text-xs text-violet-600 font-semibold",children:[a.knn,"%"]})]})]}),e.jsx("p",{className:"text-xs text-gray-500 text-center mt-1",children:"ImageNet top-1 accuracy (no fine-tuning)"})]})}function D(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"DINOv2 scales self-supervised learning to produce all-purpose visual features that work across tasks without fine-tuning. By combining DINO self-distillation with iBOT masked modeling and careful data curation, DINOv2 creates foundation-level visual representations."}),e.jsxs(x,{title:"DINOv2: Combined Objective",children:[e.jsx("p",{children:"DINOv2 combines two self-supervised losses:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_{\\text{DINOv2}} = \\mathcal{L}_{\\text{DINO}}(\\text{[CLS]}) + \\mathcal{L}_{\\text{iBOT}}(\\text{patch tokens})"}),e.jsxs("p",{className:"mt-2",children:[e.jsx(t.InlineMath,{math:"\\mathcal{L}_{\\text{DINO}}"}),": Self-distillation on [CLS] token (global features).",e.jsx(t.InlineMath,{math:"\\mathcal{L}_{\\text{iBOT}}"}),": Masked image modeling on patch tokens (local features). The combination yields features strong for both image-level and dense prediction tasks."]})]}),e.jsx(A,{}),e.jsx(g,{title:"Data Curation Pipeline",children:e.jsx("p",{children:"DINOv2 curates a 142M image dataset (LVD-142M) through: (1) web crawling to collect candidate images, (2) deduplication using copy detection, (3) self-supervised retrieval to select images similar to curated datasets (ImageNet). This automated pipeline avoids manual annotation while ensuring data quality and diversity."})}),e.jsx(u,{title:"Using DINOv2 Features for Downstream Tasks",code:`import torch
import torch.nn as nn

# Load pre-trained DINOv2 (using torch.hub)
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

class DINOv2Wrapper:
    """Demonstrates DINOv2 feature extraction patterns."""
    def __init__(self, embed_dim=1024):
        self.embed_dim = embed_dim

    def extract_features(self, images):
        """Returns CLS token and patch tokens."""
        # In practice: features = model.forward_features(images)
        B = images.shape[0]
        cls_token = torch.randn(B, self.embed_dim)
        patch_tokens = torch.randn(B, 256, self.embed_dim)  # 16x16 patches
        return cls_token, patch_tokens

# Usage patterns (no fine-tuning needed!)
wrapper = DINOv2Wrapper()
images = torch.randn(4, 3, 224, 224)
cls_feat, patch_feat = wrapper.extract_features(images)

# 1. Image classification: linear probe on CLS token
classifier = nn.Linear(1024, 1000)
logits = classifier(cls_feat)

# 2. Semantic segmentation: linear probe on patch tokens
seg_head = nn.Linear(1024, 21)  # 21 classes
seg_map = seg_head(patch_feat)  # (B, 256, 21)

# 3. k-NN classification (no training at all!)
# Just compute cosine similarity to labeled reference features

print(f"CLS features: {cls_feat.shape} (for classification)")
print(f"Patch features: {patch_feat.shape} (for dense prediction)")
print("All from a single frozen backbone — no fine-tuning!")`}),e.jsx(b,{title:"Compute Requirements",children:e.jsx("p",{children:"DINOv2 ViT-g was trained on 142M images for 625K iterations on 140 A100 GPUs. Training from scratch is impractical for most researchers. However, the released pre-trained models serve as powerful frozen feature extractors for a wide range of tasks."})}),e.jsx(p,{type:"note",title:"DINOv2 as a Visual Foundation Model",children:e.jsx("p",{children:"DINOv2 features rival or exceed supervised features (including CLIP) on many tasks without any task-specific training: depth estimation, semantic segmentation, image retrieval, and classification. This makes DINOv2 arguably the strongest general-purpose visual feature extractor as of its release."})})]})}const ae=Object.freeze(Object.defineProperty({__proto__:null,default:D},Symbol.toStringTag,{value:"Module"}));function q(){const[s,d]=m.useState("classification"),r={classification:{name:"ImageNet Classification",supervised:76.5,simclr:71.7,dino:77.3,dinov2:83.5,metric:"Top-1 Acc (%)"},segmentation:{name:"ADE20K Segmentation",supervised:45.1,simclr:39.2,dino:44.6,dinov2:49,metric:"mIoU"},detection:{name:"COCO Detection",supervised:38.2,simclr:35.8,dino:39.1,dinov2:42.5,metric:"AP50"},retrieval:{name:"Image Retrieval",supervised:70.1,simclr:65.4,dino:73.8,dinov2:78.2,metric:"mAP"}},a=r[s],l=Math.max(a.supervised,a.simclr,a.dino,a.dinov2);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-2 text-base font-bold text-gray-800 dark:text-gray-200",children:"Transfer Learning Benchmark"}),e.jsx("div",{className:"flex gap-2 mb-3 flex-wrap",children:Object.entries(r).map(([o,c])=>e.jsx("button",{onClick:()=>d(o),className:`px-3 py-1 rounded-full text-xs font-medium transition-colors ${s===o?"bg-violet-500 text-white":"bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400"}`,children:c.name.split(" ").slice(-1)[0]},o))}),e.jsx("div",{className:"flex gap-3 justify-center items-end h-28",children:[{label:"Supervised",val:a.supervised,color:"bg-gray-400"},{label:"SimCLR",val:a.simclr,color:"bg-violet-300"},{label:"DINO",val:a.dino,color:"bg-violet-400"},{label:"DINOv2",val:a.dinov2,color:"bg-violet-600"}].map(({label:o,val:c,color:i})=>e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("div",{className:`w-12 ${i} rounded-t transition-all`,style:{height:`${c/l*85}px`}}),e.jsx("span",{className:"text-[9px] text-gray-500 mt-1",children:o}),e.jsx("span",{className:"text-[9px] font-semibold",children:c})]},o))}),e.jsxs("p",{className:"text-xs text-gray-500 text-center mt-1",children:[a.name," — ",a.metric," (linear probe, ViT-L)"]})]})}function V(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"The ultimate test of self-supervised representations is their transfer performance. Understanding how to evaluate, align, and effectively use pre-trained features for downstream tasks is crucial for practical applications."}),e.jsxs(x,{title:"Evaluation Protocols",children:[e.jsx("p",{children:"Standard protocols for evaluating self-supervised features:"}),e.jsxs("ul",{className:"list-disc ml-5 mt-2 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Linear probe"}),": Train a single linear layer on frozen features"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"k-NN evaluation"}),": Nearest-neighbor classification with no training"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Fine-tuning"}),": Update all parameters on downstream task"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Few-shot"}),": Linear probe with limited labeled data (1%, 10%)"]})]}),e.jsx(t.BlockMath,{math:"\\text{Linear probe: } \\min_W \\mathcal{L}(W f_\\theta(\\mathbf{x}), y), \\quad \\theta \\text{ frozen}"})]}),e.jsx(q,{}),e.jsxs(f,{title:"Feature Alignment and Uniformity",id:"alignment-uniformity",children:[e.jsx("p",{children:"Good representation quality requires two properties on the unit hypersphere:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_{\\text{align}} = \\mathbb{E}_{(x,x^+)}\\left[\\|f(x) - f(x^+)\\|^2\\right]"}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_{\\text{uniform}} = \\log \\mathbb{E}_{(x,y)}\\left[e^{-2\\|f(x) - f(y)\\|^2}\\right]"}),e.jsxs("p",{className:"mt-2",children:[e.jsx("strong",{children:"Alignment"}),": positive pairs should be close. ",e.jsx("strong",{children:"Uniformity"}),": features should be uniformly distributed (high entropy). These two metrics predict downstream performance."]})]}),e.jsx(g,{title:"When to Fine-tune vs Freeze",children:e.jsxs("p",{children:[e.jsx("strong",{children:"Freeze features"})," when: target domain is similar to pre-training data, labeled data is very limited, or you need fast iteration. ",e.jsx("strong",{children:"Fine-tune"})," when: target domain differs significantly (medical, satellite imagery), sufficient labeled data exists, or maximum performance is needed. A middle ground: fine-tune only the last few layers."]})}),e.jsx(u,{title:"Feature Evaluation: Linear Probe and k-NN",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearProbe(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, features):
        return self.linear(features.detach())  # detach = frozen backbone

def knn_evaluate(train_features, train_labels, test_features, test_labels, k=20):
    """k-NN classification with no training."""
    train_features = F.normalize(train_features, dim=-1)
    test_features = F.normalize(test_features, dim=-1)

    # Cosine similarity
    sim = test_features @ train_features.T  # (N_test, N_train)
    topk_sim, topk_idx = sim.topk(k, dim=-1)

    # Weighted voting
    topk_labels = train_labels[topk_idx]  # (N_test, k)
    predictions = topk_labels.mode(dim=-1).values

    accuracy = (predictions == test_labels).float().mean()
    return accuracy.item()

# Alignment and uniformity metrics
def alignment(x, y):
    return (x - y).norm(dim=-1).pow(2).mean()

def uniformity(x, t=2):
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()

features = F.normalize(torch.randn(256, 768), dim=-1)
pos_pairs = features + 0.1 * torch.randn_like(features)
pos_pairs = F.normalize(pos_pairs, dim=-1)
print(f"Alignment: {alignment(features, pos_pairs):.4f} (lower = better)")
print(f"Uniformity: {uniformity(features):.4f} (lower = better)")`}),e.jsx(p,{type:"note",title:"Self-Supervised Features in Practice",children:e.jsx("p",{children:"Self-supervised pre-training has become the default initialization for many vision applications. DINOv2 features serve as drop-in replacements for ImageNet-supervised features across classification, detection, segmentation, and retrieval. The key practical insight: invest in the best available pre-trained backbone and adapt minimally to your task."})})]})}const se=Object.freeze(Object.defineProperty({__proto__:null,default:V},Symbol.toStringTag,{value:"Module"}));export{Y as a,H as b,Q as c,G as d,J as e,X as f,Z as g,ee as h,te as i,ae as j,se as k,U as s};

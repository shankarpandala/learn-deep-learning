import{j as e,r as d}from"./vendor-DpISuAX6.js";import{r as t}from"./vendor-katex-CbWCYdth.js";import{D as u,E as p,P as f,N as h,W as v,T as k}from"./subject-01-foundations-D0A1VJsr.js";function N(){const[a,l]=d.useState(.07),[i,n]=d.useState(4),s=Array.from({length:i},(r,m)=>Array.from({length:i},(j,g)=>{const b=m===g?.8+Math.random()*.15:Math.random()*.3;return Math.exp(b/a)})),o=s.map(r=>r.reduce((m,j)=>m+j,0));return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"CLIP Contrastive Similarity Matrix"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3 flex-wrap",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Temperature: ",a.toFixed(2),e.jsx("input",{type:"range",min:.01,max:.5,step:.01,value:a,onChange:r=>l(parseFloat(r.target.value)),className:"w-32 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Batch size: ",i,e.jsx("input",{type:"range",min:2,max:8,step:1,value:i,onChange:r=>n(parseInt(r.target.value)),className:"w-24 accent-violet-500"})]})]}),e.jsx("div",{className:"overflow-x-auto",children:e.jsxs("table",{className:"mx-auto text-xs",children:[e.jsx("thead",{children:e.jsxs("tr",{children:[e.jsx("th",{className:"p-1 text-gray-500",children:"img\\txt"}),Array.from({length:i},(r,m)=>e.jsxs("th",{className:"p-1 text-violet-600 dark:text-violet-400",children:["T",m]},m))]})}),e.jsx("tbody",{children:s.map((r,m)=>e.jsxs("tr",{children:[e.jsxs("td",{className:"p-1 font-medium text-violet-600 dark:text-violet-400",children:["I",m]}),r.map((j,g)=>{const b=j/o[m],_=m===g?`rgba(139, 92, 246, ${Math.min(b*1.5,.6)})`:`rgba(156, 163, 175, ${b*.5})`;return e.jsx("td",{className:"p-1 text-center rounded",style:{backgroundColor:_},children:b.toFixed(2)},g)})]},m))})]})}),e.jsx("p",{className:"mt-2 text-xs text-gray-500 text-center",children:"Diagonal entries are matched image-text pairs (should be high probability)"})]})}function w(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"CLIP (Contrastive Language-Image Pre-training) by OpenAI learns a shared embedding space for images and text using contrastive learning on 400M image-text pairs from the internet. It enables powerful zero-shot transfer to downstream vision tasks."}),e.jsxs(u,{title:"CLIP Contrastive Objective",children:[e.jsxs("p",{children:["Given a batch of ",e.jsx(t.InlineMath,{math:"N"})," image-text pairs, CLIP maximizes the cosine similarity of matching pairs while minimizing it for non-matching pairs:"]}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = -\\frac{1}{2N}\\sum_{i=1}^{N}\\left[\\log\\frac{\\exp(\\text{sim}(I_i, T_i)/\\tau)}{\\sum_{j=1}^{N}\\exp(\\text{sim}(I_i, T_j)/\\tau)} + \\log\\frac{\\exp(\\text{sim}(T_i, I_i)/\\tau)}{\\sum_{j=1}^{N}\\exp(\\text{sim}(T_i, I_j)/\\tau)}\\right]"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"\\tau"})," is a learned temperature parameter and ",e.jsx(t.InlineMath,{math:"\\text{sim}(a,b) = \\frac{a \\cdot b}{\\|a\\|\\|b\\|}"}),"."]})]}),e.jsx(N,{}),e.jsxs(p,{title:"Why Contrastive Learning Works at Scale",children:[e.jsxs("p",{children:["With a batch size of ",e.jsx(t.InlineMath,{math:"N = 32{,}768"}),", each image acts as a positive pair with its text and a negative pair with ",e.jsx(t.InlineMath,{math:"32{,}767"})," other texts. This gives:"]}),e.jsx(t.BlockMath,{math:"\\text{Negative examples per step} = N^2 - N = 32{,}768^2 - 32{,}768 \\approx 10^9"}),e.jsx("p",{children:"The massive number of negatives drives the model to learn fine-grained distinctions."})]}),e.jsx(f,{title:"CLIP-Style Contrastive Loss",code:`import torch
import torch.nn.functional as F

def clip_loss(image_embeds, text_embeds, temperature=0.07):
    """Symmetric contrastive loss for CLIP."""
    # Normalize embeddings
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # Cosine similarity matrix [N, N]
    logits = image_embeds @ text_embeds.T / temperature

    # Symmetric cross-entropy loss
    N = logits.shape[0]
    labels = torch.arange(N, device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2

# Example: batch of 8 image-text pairs, 512-dim embeddings
img_emb = torch.randn(8, 512)
txt_emb = torch.randn(8, 512)
loss = clip_loss(img_emb, txt_emb)
print(f"CLIP loss: {loss.item():.4f}")`}),e.jsx(h,{type:"note",title:"Dual Encoder Architecture",children:e.jsxs("p",{children:["CLIP uses separate encoders for each modality: a Vision Transformer (ViT) or ResNet for images and a Transformer for text. Both project to a shared ",e.jsx(t.InlineMath,{math:"d"}),"-dimensional space (typically 512 or 768). This dual-encoder design enables efficient retrieval since embeddings can be precomputed and compared with simple dot products."]})}),e.jsx(h,{type:"warning",title:"Training Scale Matters",children:e.jsx("p",{children:"CLIP required 400M image-text pairs and massive compute. Smaller-scale reproductions often underperform significantly, highlighting the critical role of data scale in contrastive vision-language learning."})})]})}const ce=Object.freeze(Object.defineProperty({__proto__:null,default:w},Symbol.toStringTag,{value:"Module"}));function M(){const[a,l]=d.useState(.5),i=-Math.log(Math.exp(a/.07)/(Math.exp(a/.07)+Math.exp(.2/.07))),n=-Math.log(1/(1+Math.exp(-a/.07)))-Math.log(1-1/(1+Math.exp(.2/.07)));return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Softmax vs Sigmoid Loss Comparison"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Positive pair similarity: ",a.toFixed(2),e.jsx("input",{type:"range",min:-1,max:1,step:.01,value:a,onChange:s=>l(parseFloat(s.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("div",{className:"grid grid-cols-2 gap-4 text-sm",children:[e.jsxs("div",{className:"p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20",children:[e.jsx("p",{className:"font-medium text-violet-700 dark:text-violet-300",children:"Softmax (CLIP)"}),e.jsxs("p",{className:"text-gray-600 dark:text-gray-400",children:["Loss: ",i.toFixed(4)]}),e.jsx("p",{className:"text-xs mt-1 text-gray-500",children:"Requires all-to-all comparison within batch"})]}),e.jsxs("div",{className:"p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20",children:[e.jsx("p",{className:"font-medium text-violet-700 dark:text-violet-300",children:"Sigmoid (SigLIP)"}),e.jsxs("p",{className:"text-gray-600 dark:text-gray-400",children:["Loss: ",n.toFixed(4)]}),e.jsx("p",{className:"text-xs mt-1 text-gray-500",children:"Pairwise — no global normalization needed"})]})]})]})}function L(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"OpenCLIP is an open-source reproduction of CLIP trained on public datasets like LAION-2B. SigLIP replaces the softmax-based contrastive loss with a simpler sigmoid loss, removing the need for global batch normalization and enabling better scaling."}),e.jsxs(u,{title:"SigLIP Sigmoid Loss",children:[e.jsx("p",{children:"Instead of softmax over the full batch, SigLIP applies a binary sigmoid loss to each pair independently:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = -\\frac{1}{N^2}\\sum_{i,j}\\left[y_{ij}\\log\\sigma(s_{ij}/\\tau) + (1-y_{ij})\\log(1-\\sigma(s_{ij}/\\tau))\\right]"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"y_{ij} = \\mathbb{1}[i=j]"})," and ",e.jsx(t.InlineMath,{math:"s_{ij} = \\text{sim}(I_i, T_j)"}),". This eliminates the softmax denominator that requires all-gather across devices."]})]}),e.jsx(M,{}),e.jsxs(p,{title:"OpenCLIP Scaling Results",children:[e.jsx("p",{children:"OpenCLIP trained on LAION-2B with ViT-G/14 achieves:"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsxs("li",{children:["ImageNet zero-shot top-1: ",e.jsx("strong",{children:"80.1%"})," (vs. CLIP's 76.2%)"]}),e.jsx("li",{children:"Training: 34B samples seen, 1024 GPUs"}),e.jsx("li",{children:"Key insight: open data + longer training can match proprietary data"})]})]}),e.jsx(f,{title:"SigLIP-Style Sigmoid Contrastive Loss",code:`import torch
import torch.nn.functional as F

def siglip_loss(image_embeds, text_embeds, temperature=0.1, bias=0.0):
    """Sigmoid contrastive loss (SigLIP).
    No softmax normalization — each pair is independent."""
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # Pairwise similarities [N, N]
    logits = image_embeds @ text_embeds.T / temperature + bias

    # Binary labels: 1 on diagonal, 0 elsewhere
    N = logits.shape[0]
    labels = 2 * torch.eye(N, device=logits.device) - 1  # +1 or -1

    # Sigmoid binary cross-entropy
    loss = -F.logsigmoid(labels * logits).mean()
    return loss

# Compare losses
img = torch.randn(16, 512)
txt = torch.randn(16, 512)
print(f"SigLIP loss: {siglip_loss(img, txt).item():.4f}")`}),e.jsx(h,{type:"note",title:"Advantages of Sigmoid Loss",children:e.jsx("p",{children:"The sigmoid formulation has two practical benefits: (1) it avoids the all-gather communication needed for softmax normalization in distributed training, enabling larger effective batch sizes, and (2) it provides a per-pair learning signal rather than competing within a batch, which empirically improves performance on smaller batches."})}),e.jsx(h,{type:"note",title:"LAION Datasets",children:e.jsx("p",{children:"OpenCLIP was trained on LAION-400M and LAION-2B, large-scale image-text datasets collected from Common Crawl. These datasets enabled the research community to reproduce and extend CLIP-style training without proprietary data. Key processing steps include NSFW filtering, deduplication, and CLIP-based quality scoring to remove low-quality pairs."})}),e.jsxs(p,{title:"SigLIP vs CLIP Performance Comparison",children:[e.jsx("p",{children:"SigLIP with ViT-B/16 on ImageNet zero-shot classification:"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsx("li",{children:"SigLIP: 73.2% top-1 accuracy (sigmoid loss)"}),e.jsx("li",{children:"CLIP: 71.1% top-1 accuracy (softmax loss)"}),e.jsx("li",{children:"Improvement comes from better gradient signal per pair"}),e.jsx("li",{children:"Advantage grows at smaller batch sizes (128-4096)"})]}),e.jsx("p",{className:"mt-2",children:"The sigmoid loss also enables batch sizes up to 1M without communication overhead."})]})]})}const de=Object.freeze(Object.defineProperty({__proto__:null,default:L},Symbol.toStringTag,{value:"Module"}));function T(){const[a,l]=d.useState(0),i=["a photo of a cat","a photo of a dog","a photo of a bird","a photo of a car"],s=[[.92,.15,.08,.02],[.12,.89,.1,.03],[.06,.08,.94,.01],[.03,.02,.01,.97]][a];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Zero-Shot Classification Demo"}),e.jsx("p",{className:"text-sm text-gray-500 mb-3",children:"Select an image type to see similarity scores across text prompts"}),e.jsx("div",{className:"flex gap-2 mb-4 flex-wrap",children:["Cat","Dog","Bird","Car"].map((o,r)=>e.jsxs("button",{onClick:()=>l(r),className:`px-3 py-1 rounded-lg text-sm transition ${a===r?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:[o," Image"]},r))}),e.jsx("div",{className:"space-y-2",children:i.map((o,r)=>e.jsxs("div",{className:"flex items-center gap-3",children:[e.jsxs("span",{className:"text-xs text-gray-500 w-40 truncate",children:['"',o,'"']}),e.jsx("div",{className:"flex-1 h-5 bg-gray-100 dark:bg-gray-800 rounded overflow-hidden",children:e.jsx("div",{className:"h-full bg-violet-500 rounded transition-all duration-300",style:{width:`${s[r]*100}%`}})}),e.jsxs("span",{className:"text-xs font-mono w-12 text-right text-gray-600 dark:text-gray-400",children:[(s[r]*100).toFixed(1),"%"]})]},r))})]})}function S(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"CLIP's shared embedding space enables zero-shot classification — recognizing categories never seen during training by comparing image embeddings with text descriptions of classes. This eliminates the need for labeled training data for new tasks."}),e.jsxs(u,{title:"Zero-Shot Classification with CLIP",children:[e.jsxs("p",{children:["Given an image ",e.jsx(t.InlineMath,{math:"I"})," and candidate class names ",e.jsx(t.InlineMath,{math:"\\{c_1, \\ldots, c_K\\}"}),', form text prompts like "a photo of a [class]". The predicted class is:']}),e.jsx(t.BlockMath,{math:"\\hat{y} = \\arg\\max_{k} \\frac{\\exp(\\text{sim}(f_I(I), f_T(c_k))/\\tau)}{\\sum_{j=1}^{K}\\exp(\\text{sim}(f_I(I), f_T(c_j))/\\tau)}"})]}),e.jsx(T,{}),e.jsxs(p,{title:"Prompt Engineering for Zero-Shot CLIP",children:[e.jsx("p",{children:"The choice of text prompt significantly affects performance. Using ensembles of prompts improves accuracy:"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsxs("li",{children:[e.jsx("code",{children:'"a photo of a dog"'})," — baseline template"]}),e.jsxs("li",{children:[e.jsx("code",{children:'"a centered satellite photo of a dog"'})," — domain-specific"]}),e.jsxs("li",{children:[e.jsx("code",{children:'"a good photo of a dog"'}),", ",e.jsx("code",{children:'"a bad photo of a dog"'})," — quality variations"]}),e.jsxs("li",{children:["Ensemble: average embeddings across ",e.jsx(t.InlineMath,{math:"M"})," prompts per class"]})]}),e.jsx(t.BlockMath,{math:"\\bar{t}_k = \\frac{1}{M}\\sum_{m=1}^{M} f_T(\\text{prompt}_m(c_k))"})]}),e.jsx(f,{title:"Zero-Shot Classification with CLIP",code:`import torch
import torch.nn.functional as F

def zero_shot_classify(image_features, text_features, temperature=0.01):
    """Zero-shot classification using precomputed CLIP features.

    Args:
        image_features: [N, D] normalized image embeddings
        text_features: [K, D] normalized text embeddings for K classes
    Returns:
        predictions: [N] predicted class indices
        probs: [N, K] class probabilities
    """
    # Cosine similarity -> softmax probabilities
    logits = image_features @ text_features.T / temperature
    probs = F.softmax(logits, dim=-1)
    predictions = probs.argmax(dim=-1)
    return predictions, probs

# Example: 100 images, 10 classes, 512-dim features
image_feats = F.normalize(torch.randn(100, 512), dim=-1)
text_feats = F.normalize(torch.randn(10, 512), dim=-1)

preds, probs = zero_shot_classify(image_feats, text_feats)
print(f"Predictions shape: {preds.shape}")
print(f"Top class probabilities: {probs.max(dim=-1).values[:5]}")`}),e.jsx(h,{type:"note",title:"Image-Text Retrieval",children:e.jsx("p",{children:"The same embedding space supports bidirectional retrieval: given an image, find the most relevant texts (image-to-text retrieval), or given a text query, find matching images (text-to-image retrieval). This is the backbone of many visual search systems."})}),e.jsx(h,{type:"warning",title:"Distribution Shift Limitations",children:e.jsx("p",{children:"While CLIP is robust to many distribution shifts, it can still fail on specialized domains (medical imaging, satellite imagery) where internet-scraped training data provides poor coverage. Fine-tuning or domain-specific adapters are often needed."})})]})}const me=Object.freeze(Object.defineProperty({__proto__:null,default:S},Symbol.toStringTag,{value:"Module"}));function C(){const[a,l]=d.useState(64),[i,n]=d.useState(256),s=(a/i*100).toFixed(1);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Perceiver Resampler Compression"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3 flex-wrap",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Visual tokens: ",i,e.jsx("input",{type:"range",min:64,max:576,step:64,value:i,onChange:o=>n(parseInt(o.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Latent queries: ",a,e.jsx("input",{type:"range",min:8,max:128,step:8,value:a,onChange:o=>l(parseInt(o.target.value)),className:"w-28 accent-violet-500"})]})]}),e.jsxs("div",{className:"flex items-center gap-4",children:[e.jsxs("div",{className:"flex-1 h-6 bg-violet-100 dark:bg-violet-900/30 rounded overflow-hidden relative",children:[e.jsx("div",{className:"h-full bg-violet-400",style:{width:"100%"}}),e.jsxs("span",{className:"absolute inset-0 flex items-center justify-center text-xs font-medium",children:[i," visual tokens"]})]}),e.jsx("span",{className:"text-gray-400 text-lg",children:"→"}),e.jsxs("div",{className:"h-6 bg-violet-500 rounded flex items-center justify-center text-xs text-white font-medium px-2",style:{width:`${Math.max(s,15)}%`,minWidth:"80px"},children:[a," latents"]})]}),e.jsxs("p",{className:"mt-2 text-xs text-gray-500 text-center",children:["Compression ratio: ",s,"% — reduces compute for cross-attention in LM layers"]})]})}function z(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Flamingo by DeepMind interleaves visual and text tokens, enabling few-shot multimodal learning. It uses a frozen vision encoder and a frozen language model, connected by lightweight cross-attention layers and a Perceiver Resampler."}),e.jsxs(u,{title:"Perceiver Resampler",children:[e.jsx("p",{children:"The Perceiver Resampler compresses variable-length visual features into a fixed number of latent tokens using cross-attention:"}),e.jsx(t.BlockMath,{math:"z = \\text{CrossAttn}(Q=\\text{latents}, K=V=\\text{visual\\_tokens})"}),e.jsxs("p",{className:"mt-2",children:["where latents are ",e.jsx(t.InlineMath,{math:"N_q"})," learned queries (typically 64) and visual tokens come from the frozen vision encoder. This produces a fixed-size representation regardless of image resolution."]})]}),e.jsx(C,{}),e.jsxs(p,{title:"Flamingo Few-Shot Performance",children:[e.jsx("p",{children:"With just 4 image-text examples as context (4-shot), Flamingo-80B achieves:"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsx("li",{children:"VQAv2: 67.6% (vs. 56.3% zero-shot)"}),e.jsx("li",{children:"COCO captioning: CIDEr 113.4"}),e.jsx("li",{children:"OK-VQA: 57.8% (requires external knowledge)"})]}),e.jsx("p",{className:"mt-2",children:"Key insight: frozen pretrained models + lightweight adapters = strong few-shot multimodal learning."})]}),e.jsx(f,{title:"Gated Cross-Attention (Flamingo-Style)",code:`import torch
import torch.nn as nn

class GatedCrossAttention(nn.Module):
    """Cross-attention layer inserted into frozen LM (Flamingo)."""
    def __init__(self, dim=768, num_heads=12):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))  # tanh gate, init at 0
        self.norm = nn.LayerNorm(dim)

    def forward(self, text_hidden, visual_tokens):
        # Cross-attend from text to visual tokens
        residual = text_hidden
        x = self.norm(text_hidden)
        attn_out, _ = self.cross_attn(query=x, key=visual_tokens, value=visual_tokens)
        # Gated residual — gate starts at 0 so LM behavior is preserved initially
        return residual + torch.tanh(self.gate) * attn_out

layer = GatedCrossAttention()
text_h = torch.randn(2, 128, 768)    # [batch, seq_len, dim]
vis_tok = torch.randn(2, 64, 768)    # [batch, num_latents, dim]
out = layer(text_h, vis_tok)
print(f"Output shape: {out.shape}")   # [2, 128, 768]
print(f"Initial gate value: {torch.tanh(layer.gate).item():.4f}")`}),e.jsx(h,{type:"note",title:"Interleaved Image-Text Input",children:e.jsx("p",{children:'Flamingo can process sequences with multiple images interleaved with text, such as "Image1: [img] This is a cat. Image2: [img] This is a ___". This interleaving enables in-context learning for multimodal tasks, similar to how GPT-3 does few-shot learning with text-only examples.'})}),e.jsx(h,{type:"warning",title:"Frozen vs Fine-Tuned Components",children:e.jsx("p",{children:"Flamingo keeps both the vision encoder and LLM frozen, training only the cross-attention layers and Perceiver Resampler (~1.5% of total parameters). This preserves the strong pretrained representations while being data-efficient. However, the frozen backbone limits adaptation to domains far from the pretraining distribution. Follow-up work like IDEFICS explores selective unfreezing for domain adaptation. OpenFlamingo provides an open-source reproduction achieving competitive results with public datasets (LAION-2B, MMC4)."})})]})}const he=Object.freeze(Object.defineProperty({__proto__:null,default:z},Symbol.toStringTag,{value:"Module"}));function I(){const[a,l]=d.useState(0),i=[{name:"Stage 1: Alignment Pre-training",frozen:["Vision Encoder","LLM"],trained:["Projection"],data:"558K image-caption pairs",desc:"Train only the projection layer to align visual features with the LLM input space."},{name:"Stage 2: Visual Instruction Tuning",frozen:["Vision Encoder"],trained:["Projection","LLM"],data:"665K instruction-following data",desc:"Fine-tune the LLM and projection on multimodal instruction data."}],n=i[a];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"LLaVA Two-Stage Training"}),e.jsx("div",{className:"flex gap-2 mb-4",children:i.map((s,o)=>e.jsxs("button",{onClick:()=>l(o),className:`px-3 py-1 rounded-lg text-sm transition ${a===o?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:["Stage ",o+1]},o))}),e.jsx("p",{className:"text-sm font-medium text-gray-700 dark:text-gray-300 mb-2",children:n.name}),e.jsxs("div",{className:"flex gap-3 mb-2 flex-wrap",children:[n.frozen.map(s=>e.jsxs("span",{className:"px-2 py-1 rounded text-xs bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400",children:["❄ ",s," (frozen)"]},s)),n.trained.map(s=>e.jsxs("span",{className:"px-2 py-1 rounded text-xs bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300",children:["✎ ",s," (trained)"]},s))]}),e.jsxs("p",{className:"text-xs text-gray-500",children:["Data: ",n.data]}),e.jsx("p",{className:"text-sm text-gray-600 dark:text-gray-400 mt-2",children:n.desc})]})}function P(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"LLaVA (Large Language-and-Vision Assistant) connects a CLIP vision encoder to a language model via a simple linear projection, then fine-tunes on visual instruction-following data. Its simplicity and effectiveness made it a foundational architecture for open-source multimodal LLMs."}),e.jsxs(u,{title:"LLaVA Architecture",children:[e.jsx("p",{children:"LLaVA processes an image through a vision encoder and projects visual tokens into the LLM embedding space:"}),e.jsx(t.BlockMath,{math:"H_v = W \\cdot f_{\\text{CLIP}}(I), \\quad H_v \\in \\mathbb{R}^{N_v \\times d}"}),e.jsxs("p",{className:"mt-2",children:["The visual tokens ",e.jsx(t.InlineMath,{math:"H_v"})," are prepended to the text token embeddings and fed to the LLM as a unified sequence. The projection ",e.jsx(t.InlineMath,{math:"W"})," is a learned linear layer (or MLP in LLaVA-1.5)."]})]}),e.jsx(I,{}),e.jsx(f,{title:"LLaVA-Style Visual Projection",code:`import torch
import torch.nn as nn

class LLaVAProjection(nn.Module):
    """MLP projection from vision to language space (LLaVA-1.5)."""
    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, vision_features):
        return self.proj(vision_features)

def prepare_multimodal_input(vision_features, text_embeds, projector):
    """Concatenate projected visual tokens with text embeddings."""
    visual_tokens = projector(vision_features)  # [B, N_v, D_llm]
    # Prepend visual tokens to text sequence
    combined = torch.cat([visual_tokens, text_embeds], dim=1)
    return combined

proj = LLaVAProjection()
vis = torch.randn(1, 576, 1024)    # 576 patches from ViT-L/14@336px
txt = torch.randn(1, 128, 4096)    # text embeddings
combined = prepare_multimodal_input(vis, txt, proj)
print(f"Combined sequence: {combined.shape}")  # [1, 704, 4096]`}),e.jsxs(p,{title:"Visual Instruction Data Generation",children:[e.jsx("p",{children:"LLaVA's key innovation was using GPT-4 to generate visual instruction-following data:"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsx("li",{children:"158K unique images from COCO with bounding boxes and captions"}),e.jsx("li",{children:"GPT-4 generates conversations, descriptions, and reasoning about images"}),e.jsx("li",{children:"Three types: conversation (58K), detailed description (23K), complex reasoning (77K)"})]})]}),e.jsx(v,{title:"Data Quality vs Quantity",children:e.jsx("p",{children:"LLaVA demonstrates that high-quality instruction data matters more than quantity. 665K carefully curated examples outperform millions of noisy web-scraped pairs. Data quality and diversity of instruction types are critical for generalization."})}),e.jsx(h,{type:"note",title:"LLaVA-1.5 and Beyond",children:e.jsx("p",{children:"LLaVA-1.5 improved on the original with three changes: (1) replacing the linear projection with a two-layer MLP, (2) using higher resolution input (336px vs 224px), and (3) adding academic VQA data to the instruction mix. These simple changes pushed LLaVA-1.5 13B to match or exceed models trained on orders of magnitude more data (InstructBLIP, Qwen-VL). The architecture's simplicity became its strength — easy to iterate on and scale. LLaVA-NeXT further extends this with dynamic high-resolution input and stronger base LLMs, achieving near-GPT-4V performance on several benchmarks. The LLaVA family demonstrates that simple architectures with careful data curation can compete with far more complex systems."})})]})}const pe=Object.freeze(Object.defineProperty({__proto__:null,default:P},Symbol.toStringTag,{value:"Module"}));function A(){const[a,l]=d.useState("text"),i={text:{tokens:"~500 tokens",encoder:"SentencePiece tokenizer",color:"violet"},image:{tokens:"~256 tokens",encoder:"ViT patches + resampler",color:"violet"},audio:{tokens:"~128 tokens",encoder:"Whisper encoder + projection",color:"violet"},video:{tokens:"~1024 tokens",encoder:"Per-frame ViT + temporal sampling",color:"violet"}},n=i[a];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Unified Multimodal Tokenization"}),e.jsx("div",{className:"flex gap-2 mb-4 flex-wrap",children:Object.keys(i).map(s=>e.jsx("button",{onClick:()=>l(s),className:`px-3 py-1 rounded-lg text-sm capitalize transition ${a===s?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:s},s))}),e.jsxs("div",{className:"p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20 space-y-1 text-sm",children:[e.jsxs("p",{children:[e.jsx("strong",{children:"Modality:"})," ",a]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Encoder:"})," ",n.encoder]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Typical sequence length:"})," ",n.tokens]}),e.jsx("p",{className:"text-xs text-gray-500 mt-2",children:"All modalities are projected to the same transformer embedding dimension"})]})]})}function F(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Models like Gemini, GPT-4V, and Claude natively process multiple modalities within a single transformer architecture. Rather than bolting separate encoders together, these models are trained from the ground up on interleaved multimodal data."}),e.jsxs(u,{title:"Native Multimodal Architecture",children:[e.jsx("p",{children:"A unified model processes all modalities as token sequences in a shared transformer:"}),e.jsx(t.BlockMath,{math:"h = \\text{Transformer}(\\text{concat}(E_{\\text{text}}(x_t), E_{\\text{image}}(x_i), E_{\\text{audio}}(x_a), \\ldots))"}),e.jsxs("p",{className:"mt-2",children:["Each modality has its own encoder ",e.jsx(t.InlineMath,{math:"E_m"})," that produces tokens in a shared embedding space of dimension ",e.jsx(t.InlineMath,{math:"d"}),". The transformer attends across all tokens regardless of modality."]})]}),e.jsx(A,{}),e.jsxs(p,{title:"Gemini Architecture Insights",children:[e.jsx("p",{children:"Gemini (Google DeepMind) is natively multimodal from pre-training:"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsx("li",{children:"Trained on interleaved text, image, audio, and video from the start"}),e.jsx("li",{children:"Uses SentencePiece for text, ViT-style patches for images"}),e.jsx("li",{children:"Gemini Ultra: estimated 1.56T parameters (MoE architecture)"}),e.jsx("li",{children:"Can generate both text and images as output"})]})]}),e.jsx(f,{title:"Unified Multimodal Sequence Construction",code:`import torch
import torch.nn as nn

class UnifiedMultimodalEncoder(nn.Module):
    """Simplified unified encoder for multiple modalities."""
    def __init__(self, d_model=768):
        super().__init__()
        self.text_embed = nn.Embedding(32000, d_model)
        self.image_proj = nn.Linear(1024, d_model)  # from ViT
        self.audio_proj = nn.Linear(512, d_model)    # from audio encoder
        self.modality_embed = nn.Embedding(3, d_model)  # text=0, image=1, audio=2

    def forward(self, text_ids=None, image_feats=None, audio_feats=None):
        tokens = []
        if image_feats is not None:
            img_tok = self.image_proj(image_feats) + self.modality_embed(
                torch.ones(image_feats.shape[:2], dtype=torch.long, device=image_feats.device))
            tokens.append(img_tok)
        if text_ids is not None:
            txt_tok = self.text_embed(text_ids) + self.modality_embed(
                torch.zeros_like(text_ids))
            tokens.append(txt_tok)
        if audio_feats is not None:
            aud_tok = self.audio_proj(audio_feats) + self.modality_embed(
                2 * torch.ones(audio_feats.shape[:2], dtype=torch.long, device=audio_feats.device))
            tokens.append(aud_tok)
        return torch.cat(tokens, dim=1)  # [B, total_tokens, D]

enc = UnifiedMultimodalEncoder()
text = torch.randint(0, 32000, (1, 64))
image = torch.randn(1, 256, 1024)
combined = enc(text_ids=text, image_feats=image)
print(f"Combined multimodal sequence: {combined.shape}")  # [1, 320, 768]`}),e.jsx(h,{type:"note",title:"Early vs Late Fusion",children:e.jsxs("p",{children:["Unified models use ",e.jsx("strong",{children:"early fusion"})," — modalities interact from the first transformer layer. This contrasts with late fusion approaches (like CLIP) where modalities are encoded independently and only interact at the final embedding. Early fusion enables richer cross-modal reasoning but requires training on paired multimodal data."]})}),e.jsx(h,{type:"warning",title:"Challenges of Unified Models",children:e.jsx("p",{children:"Natively multimodal training introduces unique challenges: (1) balancing loss across modalities — text tends to dominate due to higher data volume, (2) tokenization inconsistencies between modalities, (3) evaluation is harder since capabilities span many benchmarks. Despite these challenges, the trend is clearly toward unified architectures that can seamlessly reason across modalities in a single forward pass."})})]})}const xe=Object.freeze(Object.defineProperty({__proto__:null,default:F},Symbol.toStringTag,{value:"Module"}));function B(){const[a,l]=d.useState(8192),[i,n]=d.useState(32),s=i*i,o=s*Math.log2(a);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"dVAE Image Tokenization"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3 flex-wrap",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Codebook size: ",a.toLocaleString(),e.jsx("input",{type:"range",min:512,max:16384,step:512,value:a,onChange:r=>l(parseInt(r.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Grid: ",i,"x",i,e.jsx("input",{type:"range",min:8,max:64,step:8,value:i,onChange:r=>n(parseInt(r.target.value)),className:"w-28 accent-violet-500"})]})]}),e.jsxs("div",{className:"grid grid-cols-3 gap-3 text-sm text-center",children:[e.jsxs("div",{className:"p-2 rounded bg-violet-50 dark:bg-violet-900/20",children:[e.jsx("p",{className:"text-violet-700 dark:text-violet-300 font-medium",children:"Image Tokens"}),e.jsx("p",{className:"text-lg font-bold",children:s})]}),e.jsxs("div",{className:"p-2 rounded bg-violet-50 dark:bg-violet-900/20",children:[e.jsx("p",{className:"text-violet-700 dark:text-violet-300 font-medium",children:"Codebook Size"}),e.jsx("p",{className:"text-lg font-bold",children:a.toLocaleString()})]}),e.jsxs("div",{className:"p-2 rounded bg-violet-50 dark:bg-violet-900/20",children:[e.jsx("p",{className:"text-violet-700 dark:text-violet-300 font-medium",children:"Bits/Image"}),e.jsxs("p",{className:"text-lg font-bold",children:[(o/1e3).toFixed(1),"K"]})]})]})]})}function E(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"DALL-E (OpenAI, 2021) pioneered text-to-image generation by treating images as sequences of discrete tokens and generating them autoregressively with a transformer. This approach unified image generation with language modeling."}),e.jsxs(u,{title:"DALL-E Two-Stage Approach",children:[e.jsxs("p",{children:[e.jsx("strong",{children:"Stage 1:"})," Train a discrete VAE (dVAE) to encode images into a grid of discrete tokens:"]}),e.jsx(t.BlockMath,{math:"z = \\arg\\min_{z_k \\in \\mathcal{C}} \\|f_{\\text{enc}}(x)_{ij} - z_k\\|^2, \\quad \\mathcal{C} = \\{z_1, \\ldots, z_K\\}"}),e.jsxs("p",{className:"mt-2",children:[e.jsx("strong",{children:"Stage 2:"})," Train an autoregressive transformer on concatenated text + image tokens:"]}),e.jsx(t.BlockMath,{math:"p(x|y) = \\prod_{i=1}^{N} p(z_i | z_{<i}, y)"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"y"})," is the text caption and ",e.jsx(t.InlineMath,{math:"z_i"})," are image tokens."]})]}),e.jsx(B,{}),e.jsxs(p,{title:"DALL-E Model Scale",children:[e.jsx("p",{children:"The DALL-E transformer uses 12B parameters to model the joint distribution of 256 BPE text tokens and 1024 image tokens (32x32 grid with codebook size 8192):"}),e.jsx(t.BlockMath,{math:"\\text{Sequence length} = 256_{\\text{text}} + 1024_{\\text{image}} = 1280 \\text{ tokens}"}),e.jsx("p",{children:"At inference, text tokens are provided as prefix and image tokens are sampled autoregressively."})]}),e.jsx(f,{title:"Simple Vector Quantization (dVAE Core)",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """Vector quantization layer for image tokenization."""
    def __init__(self, num_codes=8192, code_dim=256):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1/num_codes, 1/num_codes)

    def forward(self, z_e):
        # z_e: [B, D, H, W] -> [B, H, W, D]
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z_e.view(-1, z_e.shape[-1])

        # Find nearest codebook entry
        dists = torch.cdist(flat, self.codebook.weight)
        indices = dists.argmin(dim=-1)
        z_q = self.codebook(indices).view(z_e.shape)

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st.permute(0, 3, 1, 2), indices

vq = VectorQuantizer(num_codes=8192, code_dim=256)
encoded = torch.randn(2, 256, 32, 32)  # encoder output
quantized, tokens = vq(encoded)
print(f"Quantized shape: {quantized.shape}")
print(f"Image tokens: {tokens.shape}, unique codes used: {tokens.unique().numel()}")`}),e.jsx(h,{type:"note",title:"From DALL-E to DALL-E 2",children:e.jsx("p",{children:"DALL-E 2 replaced the autoregressive approach with a diffusion model, generating CLIP image embeddings from text and then decoding to pixels. This produced higher-fidelity images but moved away from the elegant unified token-based approach."})}),e.jsx(h,{type:"note",title:"Modern Image Tokenizers",children:e.jsx("p",{children:"The dVAE in DALL-E has been superseded by improved tokenizers: VQGAN uses adversarial training and perceptual losses for sharper reconstructions, while MAGVIT-v2 achieves near-lossless image compression at high compression ratios. These tokenizers are also used in video generation (VideoGPT, MAGVIT) and unified vision-language models that generate both text and images autoregressively."})})]})}const ge=Object.freeze(Object.defineProperty({__proto__:null,default:E},Symbol.toStringTag,{value:"Module"}));function q(){const[a,l]=d.useState(50),[i,n]=d.useState(7.5),s=Math.max(0,1-a/50),o=1-s;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Latent Diffusion Denoising Process"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3 flex-wrap",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Denoising step: ",a,"/50",e.jsx("input",{type:"range",min:0,max:50,step:1,value:a,onChange:r=>l(parseInt(r.target.value)),className:"w-32 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["CFG scale: ",i.toFixed(1),e.jsx("input",{type:"range",min:1,max:20,step:.5,value:i,onChange:r=>n(parseFloat(r.target.value)),className:"w-28 accent-violet-500"})]})]}),e.jsxs("div",{className:"space-y-2",children:[e.jsxs("div",{className:"flex items-center gap-2",children:[e.jsx("span",{className:"text-xs w-16 text-gray-500",children:"Noise"}),e.jsx("div",{className:"flex-1 h-4 bg-gray-100 dark:bg-gray-800 rounded overflow-hidden",children:e.jsx("div",{className:"h-full bg-gray-400 transition-all duration-200",style:{width:`${s*100}%`}})}),e.jsxs("span",{className:"text-xs w-10 text-right font-mono",children:[(s*100).toFixed(0),"%"]})]}),e.jsxs("div",{className:"flex items-center gap-2",children:[e.jsx("span",{className:"text-xs w-16 text-gray-500",children:"Signal"}),e.jsx("div",{className:"flex-1 h-4 bg-gray-100 dark:bg-gray-800 rounded overflow-hidden",children:e.jsx("div",{className:"h-full bg-violet-500 transition-all duration-200",style:{width:`${o*100}%`}})}),e.jsxs("span",{className:"text-xs w-10 text-right font-mono",children:[(o*100).toFixed(0),"%"]})]})]}),e.jsxs("p",{className:"mt-2 text-xs text-gray-500",children:["CFG scale ",i.toFixed(1),": higher values produce images more aligned with the text prompt but with less diversity."]})]})}function O(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Stable Diffusion performs the diffusion process in a compressed latent space rather than pixel space, dramatically reducing compute. A text encoder (CLIP) provides conditioning, and a VAE decoder converts latents back to high-resolution images."}),e.jsxs(u,{title:"Latent Diffusion Model (LDM)",children:[e.jsxs("p",{children:["The key insight is performing diffusion in latent space ",e.jsx(t.InlineMath,{math:"z = \\mathcal{E}(x)"})," where ",e.jsx(t.InlineMath,{math:"\\mathcal{E}"})," is a pretrained encoder:"]}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_{\\text{LDM}} = \\mathbb{E}_{z_0, \\epsilon, t, c}\\left[\\|\\epsilon - \\epsilon_\\theta(z_t, t, c)\\|^2\\right]"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"z_t = \\sqrt{\\bar{\\alpha}_t}z_0 + \\sqrt{1-\\bar{\\alpha}_t}\\epsilon"})," is the noised latent, ",e.jsx(t.InlineMath,{math:"c = f_\\text{CLIP}(\\text{text})"})," is the text conditioning, and ",e.jsx(t.InlineMath,{math:"\\epsilon_\\theta"})," is a U-Net with cross-attention."]})]}),e.jsx(q,{}),e.jsxs(u,{title:"Classifier-Free Guidance (CFG)",children:[e.jsx("p",{children:"CFG interpolates between conditional and unconditional predictions to strengthen text alignment:"}),e.jsx(t.BlockMath,{math:"\\tilde{\\epsilon}_\\theta(z_t, c) = \\epsilon_\\theta(z_t, \\varnothing) + s \\cdot (\\epsilon_\\theta(z_t, c) - \\epsilon_\\theta(z_t, \\varnothing))"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"s"})," is the guidance scale (typically 7-12). During training, the text condition ",e.jsx(t.InlineMath,{math:"c"})," is randomly dropped (replaced with ",e.jsx(t.InlineMath,{math:"\\varnothing"}),") 10-20% of the time."]})]}),e.jsx(f,{title:"Latent Diffusion Sampling Loop",code:`import torch

def sample_latent_diffusion(unet, scheduler, text_embeds, cfg_scale=7.5,
                            num_steps=50, latent_shape=(1, 4, 64, 64)):
    """Simplified Stable Diffusion sampling loop."""
    device = text_embeds.device
    # Start from pure noise
    latents = torch.randn(latent_shape, device=device)

    # Null text embedding for CFG
    null_embeds = torch.zeros_like(text_embeds)

    scheduler.set_timesteps(num_steps)
    for t in scheduler.timesteps:
        # Duplicate latents for CFG (unconditional + conditional)
        latent_input = torch.cat([latents, latents])
        text_input = torch.cat([null_embeds, text_embeds])

        # Predict noise
        noise_pred = unet(latent_input, t, encoder_hidden_states=text_input)
        noise_uncond, noise_cond = noise_pred.chunk(2)

        # Classifier-free guidance
        noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

        # Denoise one step
        latents = scheduler.step(noise_pred, t, latents)

    return latents  # Decode with VAE: images = vae.decode(latents)

# Note: This is pseudocode — requires diffusers library for full execution
print("Stable Diffusion: denoising in 64x64 latent space")
print("  -> 8x compression from 512x512 pixel space")
print("  -> ~60x less compute than pixel-space diffusion")`}),e.jsxs(p,{title:"Compute Savings from Latent Space",children:[e.jsx("p",{children:"For a 512x512 image with the Stable Diffusion VAE (downsampling factor 8):"}),e.jsx(t.BlockMath,{math:"\\text{Pixel space: } 512 \\times 512 \\times 3 = 786{,}432 \\text{ values}"}),e.jsx(t.BlockMath,{math:"\\text{Latent space: } 64 \\times 64 \\times 4 = 16{,}384 \\text{ values}"}),e.jsxs("p",{children:["This is a ",e.jsx("strong",{children:"48x reduction"})," in dimensionality, making training and inference dramatically cheaper."]})]}),e.jsx(h,{type:"note",title:"Stable Diffusion Components",children:e.jsx("p",{children:"The full pipeline has three pretrained components: (1) a VAE encoder/decoder for pixel-latent conversion, (2) a CLIP text encoder for conditioning, and (3) a U-Net with cross-attention that performs the actual denoising. Only the U-Net is trained for the diffusion task."})})]})}const ue=Object.freeze(Object.defineProperty({__proto__:null,default:O},Symbol.toStringTag,{value:"Module"}));function G(){const[a,l]=d.useState("canny"),i={canny:{name:"Canny Edges",desc:"Edge maps extracted from images to preserve structure and boundaries.",useCase:"Architectural drawings, product design"},depth:{name:"Depth Maps",desc:"Estimated depth to control 3D layout and perspective of generated scene.",useCase:"Scene composition, room layouts"},pose:{name:"OpenPose",desc:"Human body keypoints to control pose and position of people.",useCase:"Character art, fashion design"},segmentation:{name:"Semantic Segmentation",desc:"Pixel-level class labels define what goes where in the image.",useCase:"Landscape design, scene manipulation"}},n=i[a];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"ControlNet Conditioning Types"}),e.jsx("div",{className:"flex gap-2 mb-4 flex-wrap",children:Object.entries(i).map(([s,o])=>e.jsx("button",{onClick:()=>l(s),className:`px-3 py-1 rounded-lg text-sm transition ${a===s?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:o.name},s))}),e.jsxs("div",{className:"p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20 text-sm space-y-1",children:[e.jsx("p",{children:e.jsx("strong",{children:n.name})}),e.jsx("p",{className:"text-gray-600 dark:text-gray-400",children:n.desc}),e.jsxs("p",{className:"text-xs text-gray-500",children:["Use case: ",n.useCase]})]})]})}function V(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"ControlNet adds spatial conditioning to pretrained diffusion models by creating a trainable copy of the encoder blocks. This enables precise control over generated images using edge maps, depth maps, poses, and other structural guides."}),e.jsxs(u,{title:"ControlNet Architecture",children:[e.jsx("p",{children:"ControlNet creates a trainable copy of the locked U-Net encoder and connects it via zero convolutions:"}),e.jsx(t.BlockMath,{math:"y_c = \\mathcal{F}(x; \\Theta) + \\mathcal{Z}(\\mathcal{F}(x + \\mathcal{Z}(c; \\Theta_{z1}); \\Theta_c); \\Theta_{z2})"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"\\mathcal{F}"})," is the frozen U-Net block, ",e.jsx(t.InlineMath,{math:"\\Theta_c"})," is its trainable copy, ",e.jsx(t.InlineMath,{math:"c"})," is the control signal, and ",e.jsx(t.InlineMath,{math:"\\mathcal{Z}"})," is a zero-initialized convolution (output starts at zero, preserving the original model)."]})]}),e.jsx(G,{}),e.jsxs(p,{title:"Zero Convolution: Why It Works",children:[e.jsxs("p",{children:["The zero convolution ",e.jsx(t.InlineMath,{math:"\\mathcal{Z}(x) = W \\cdot x + b"})," is initialized with ",e.jsx(t.InlineMath,{math:"W=0, b=0"}),":"]}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsxs("li",{children:["At initialization: ",e.jsx(t.InlineMath,{math:"\\mathcal{Z}(x) = 0"}),", so the pretrained model is unchanged"]}),e.jsxs("li",{children:["Gradients are non-zero: ",e.jsx(t.InlineMath,{math:"\\frac{\\partial \\mathcal{Z}}{\\partial W} = x \\neq 0"})]}),e.jsx("li",{children:"The network gradually learns to inject control signals without disrupting pretrained features"})]})]}),e.jsx(f,{title:"ControlNet Zero Convolution Block",code:`import torch
import torch.nn as nn

class ZeroConv(nn.Module):
    """Zero-initialized convolution for ControlNet."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)

class ControlNetBlock(nn.Module):
    """Simplified ControlNet block with zero convolutions."""
    def __init__(self, channels=320):
        super().__init__()
        self.zero_in = ZeroConv(channels, channels)
        self.trainable_copy = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.zero_out = ZeroConv(channels, channels)

    def forward(self, frozen_features, control_signal):
        # Inject control via zero conv -> process -> zero conv
        h = frozen_features + self.zero_in(control_signal)
        h = self.trainable_copy(h)
        return self.zero_out(h)  # Added to frozen U-Net output

block = ControlNetBlock(channels=320)
feat = torch.randn(1, 320, 64, 64)
ctrl = torch.randn(1, 320, 64, 64)
out = block(feat, ctrl)
print(f"Output shape: {out.shape}")
print(f"Initial output magnitude: {out.abs().mean():.6f}")  # ~0 at init`}),e.jsx(h,{type:"note",title:"IP-Adapter and Other Control Methods",children:e.jsx("p",{children:"Beyond ControlNet, other methods add control to diffusion: IP-Adapter uses image embeddings as conditioning (image-prompted generation), T2I-Adapter adds lightweight control modules, and LoRA fine-tunes specific weight matrices for style transfer. These can be composed together for multi-condition generation — for example, combining a pose ControlNet with a depth ControlNet and a style LoRA to generate a specifically posed character in a particular style and spatial layout. This composability makes ControlNet a foundational tool in production image generation pipelines."})})]})}const fe=Object.freeze(Object.defineProperty({__proto__:null,default:V},Symbol.toStringTag,{value:"Module"}));function D(){const[a,l]=d.useState(10),[i,n]=d.useState(12),s=Math.pow(10,a),o=Math.pow(10,i),r=6*s*o,m=20*s,j=o>m*2,g=o<m*.5;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"LLM Training Compute Calculator"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3 flex-wrap",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Parameters: 10^",a," (",(s/1e9).toFixed(1),"B)",e.jsx("input",{type:"range",min:8,max:12,step:.5,value:a,onChange:b=>l(parseFloat(b.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Tokens: 10^",i," (",(o/1e12).toFixed(1),"T)",e.jsx("input",{type:"range",min:10,max:14,step:.5,value:i,onChange:b=>n(parseFloat(b.target.value)),className:"w-28 accent-violet-500"})]})]}),e.jsxs("div",{className:"grid grid-cols-2 gap-3 text-sm",children:[e.jsxs("div",{className:"p-2 rounded bg-violet-50 dark:bg-violet-900/20 text-center",children:[e.jsx("p",{className:"text-violet-700 dark:text-violet-300 font-medium",children:"Training FLOPs"}),e.jsx("p",{className:"font-bold",children:r.toExponential(1)})]}),e.jsxs("div",{className:`p-2 rounded text-center ${j?"bg-yellow-50 dark:bg-yellow-900/20":g?"bg-red-50 dark:bg-red-900/20":"bg-green-50 dark:bg-green-900/20"}`,children:[e.jsx("p",{className:"font-medium text-gray-700 dark:text-gray-300",children:"Chinchilla Optimal Tokens"}),e.jsx("p",{className:"font-bold",children:m.toExponential(1)}),e.jsx("p",{className:"text-xs",children:j?"Over-trained (inference-optimal)":g?"Under-trained":"Near optimal"})]})]})]})}function R(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Training large language models requires carefully balancing model size, data volume, and compute budget. Scaling laws provide principled guidance for these tradeoffs, while practical recipes address the engineering challenges of distributed training."}),e.jsxs(k,{title:"Chinchilla Scaling Law",id:"chinchilla-scaling",children:[e.jsxs("p",{children:["For a given compute budget ",e.jsx(t.InlineMath,{math:"C"}),", the optimal model size ",e.jsx(t.InlineMath,{math:"N^*"})," and training tokens ",e.jsx(t.InlineMath,{math:"D^*"})," scale equally:"]}),e.jsx(t.BlockMath,{math:"N^* \\propto C^{0.5}, \\quad D^* \\propto C^{0.5}"}),e.jsxs("p",{className:"mt-2",children:["Rule of thumb: train on ",e.jsx(t.InlineMath,{math:"\\sim 20"})," tokens per parameter. A 70B model should see ~1.4T tokens. The training FLOPs are approximately:"]}),e.jsx(t.BlockMath,{math:"C \\approx 6ND"})]}),e.jsx(D,{}),e.jsx(p,{title:"Notable LLM Training Configurations",children:e.jsxs("ul",{className:"list-disc list-inside space-y-1",children:[e.jsx("li",{children:"GPT-3 (175B): 300B tokens, ~3.6e23 FLOPs (under-trained by Chinchilla standards)"}),e.jsx("li",{children:"Chinchilla (70B): 1.4T tokens, ~5.8e23 FLOPs (compute-optimal)"}),e.jsx("li",{children:"LLaMA-2 70B: 2T tokens, ~8.4e23 FLOPs (over-trained for better inference)"}),e.jsx("li",{children:"LLaMA-3 70B: 15T tokens — heavily over-trained for inference efficiency"})]})}),e.jsx(f,{title:"Estimating LLM Training Cost",code:`import math

def estimate_training(params_b, tokens_t, gpu_tflops=312, gpu_util=0.4):
    """Estimate LLM training requirements.

    Args:
        params_b: parameters in billions
        tokens_t: training tokens in trillions
        gpu_tflops: peak GPU TFLOPS (H100 = 989 BF16, A100 = 312)
        gpu_util: model FLOPs utilization (typically 0.3-0.5)
    """
    params = params_b * 1e9
    tokens = tokens_t * 1e12
    flops = 6 * params * tokens

    # GPU-hours
    effective_tflops = gpu_tflops * gpu_util * 1e12
    gpu_seconds = flops / effective_tflops
    gpu_hours = gpu_seconds / 3600

    # Chinchilla optimal tokens
    optimal_tokens = 20 * params

    print(f"Model: {params_b}B params, {tokens_t}T tokens")
    print(f"Training FLOPs: {flops:.2e}")
    print(f"GPU-hours (single GPU): {gpu_hours:,.0f}")
    print(f"With 1024 GPUs: {gpu_hours/1024:,.0f} hours = {gpu_hours/1024/24:,.0f} days")
    print(f"Chinchilla optimal tokens: {optimal_tokens/1e12:.1f}T")
    print(f"Token ratio: {tokens/optimal_tokens:.1f}x optimal")

estimate_training(70, 2.0, gpu_tflops=312)  # LLaMA-2 70B on A100
print()
estimate_training(70, 15.0, gpu_tflops=989)  # LLaMA-3 70B on H100`}),e.jsx(h,{type:"note",title:"Beyond Compute-Optimal Training",children:e.jsx("p",{children:"Modern LLMs are deliberately over-trained relative to Chinchilla optimality. Since inference cost scales with model size (not training tokens), it is cheaper to train a smaller model on more data than to train a larger compute-optimal model. LLaMA-3 trained a 70B model on 15T tokens — 10x more than Chinchilla would recommend."})})]})}const ye=Object.freeze(Object.defineProperty({__proto__:null,default:R},Symbol.toStringTag,{value:"Module"}));function K(){const[a,l]=d.useState("exact_match"),i=400,n=200,s=40,o=[.1,1,10,100,1e3],r=o.map(x=>Math.log10(x)),m={exact_match:{name:"Exact Match (sharp emergence)",values:[0,.01,.02,.15,.85]},log_likelihood:{name:"Log-Likelihood (smooth scaling)",values:[.1,.25,.42,.6,.78]}},j=m[a],g=x=>s+(x-r[0])/(r[r.length-1]-r[0])*(i-2*s),b=x=>n-s-x*(n-2*s),_=j.values.map((x,c)=>`${c===0?"M":"L"}${g(r[c])},${b(x)}`).join(" ");return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Emergence vs Smooth Scaling"}),e.jsx("div",{className:"flex gap-2 mb-3",children:Object.entries(m).map(([x,c])=>e.jsx("button",{onClick:()=>l(x),className:`px-3 py-1 rounded-lg text-xs transition ${a===x?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:c.name},x))}),e.jsxs("svg",{width:i,height:n,className:"mx-auto block",children:[e.jsx("line",{x1:s,y1:n-s,x2:i-s,y2:n-s,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("line",{x1:s,y1:s,x2:s,y2:n-s,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("path",{d:_,fill:"none",stroke:"#8b5cf6",strokeWidth:2.5}),j.values.map((x,c)=>e.jsx("circle",{cx:g(r[c]),cy:b(x),r:4,fill:"#8b5cf6"},c)),o.map((x,c)=>e.jsx("text",{x:g(r[c]),y:n-10,textAnchor:"middle",className:"text-[9px] fill-gray-500",children:x>=1?x+"B":x*1e3+"M"},c)),e.jsx("text",{x:i/2,y:n-1,textAnchor:"middle",className:"text-[10px] fill-gray-500",children:"Model Size"}),e.jsx("text",{x:12,y:n/2,textAnchor:"middle",transform:`rotate(-90,12,${n/2})`,className:"text-[10px] fill-gray-500",children:"Score"})]})]})}function W(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:'As language models scale, they exhibit qualitative changes in capability: in-context learning, chain-of-thought reasoning, and instruction following appear to emerge at specific scale thresholds. Whether these are truly "emergent" or artifacts of metric choice is debated.'}),e.jsxs(u,{title:"In-Context Learning (ICL)",children:[e.jsx("p",{children:"The ability to learn new tasks from examples provided in the prompt, without gradient updates:"}),e.jsx(t.BlockMath,{math:"p(y|x, \\{(x_1, y_1), \\ldots, (x_k, y_k)\\})"}),e.jsxs("p",{className:"mt-2",children:["The model conditions on ",e.jsx(t.InlineMath,{math:"k"})," demonstration examples to predict ",e.jsx(t.InlineMath,{math:"y"})," for a new input ",e.jsx(t.InlineMath,{math:"x"}),". This behavior is not explicitly trained — it emerges from next-token prediction on diverse text."]})]}),e.jsx(K,{}),e.jsx(v,{title:"Are Emergent Abilities a Mirage?",children:e.jsx("p",{children:'Schaeffer et al. (2023) argue that "emergence" is an artifact of nonlinear metrics like exact match. When evaluated with continuous metrics (log-likelihood, Brier score), performance scales smoothly. The apparent phase transition comes from thresholding continuous improvement, not from a qualitative change in the model.'})}),e.jsx(f,{title:"In-Context Learning vs Fine-Tuning",code:`import torch
import torch.nn.functional as F

def simulate_icl_performance(model_size_b, num_shots, task_difficulty=0.5):
    """Simulate how ICL performance scales with model size and shots.

    Key finding: ICL performance improves log-linearly with model size
    and log-linearly with number of examples (up to context window).
    """
    # Approximate performance model (from empirical observations)
    import math
    base = 0.5 * math.log10(model_size_b + 1) / 3  # size contribution
    shot_bonus = 0.1 * math.log2(num_shots + 1)      # shot contribution
    perf = min(base + shot_bonus - task_difficulty * 0.3, 1.0)
    return max(perf, 0.0)

# Compare ICL at different scales
for size in [1, 7, 70, 405]:
    scores = [simulate_icl_performance(size, k) for k in [0, 1, 4, 16]]
    print(f"{size:>4}B params | 0-shot: {scores[0]:.2f} | 1-shot: {scores[1]:.2f} "
          f"| 4-shot: {scores[2]:.2f} | 16-shot: {scores[3]:.2f}")`}),e.jsxs(p,{title:"Chain-of-Thought Reasoning",children:[e.jsx("p",{children:"Chain-of-thought (CoT) prompting asks models to show their reasoning steps. It reliably improves performance on multi-step tasks but only works above ~100B parameters:"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsx("li",{children:"GSM8K (math): 17.9% (standard) vs 57.1% (CoT) with PaLM 540B"}),e.jsx("li",{children:"No improvement below ~10B parameters — models produce plausible but wrong reasoning"}),e.jsx("li",{children:`"Let's think step by step" (zero-shot CoT) works surprisingly well`})]})]}),e.jsx(h,{type:"note",title:"Instruction Tuning and RLHF",children:e.jsx("p",{children:"Raw language models are poor at following instructions. Instruction tuning (fine-tuning on instruction-response pairs) and RLHF (learning from human preferences) dramatically improve usability. A 7B instruction-tuned model can outperform a 175B base model on user-facing tasks."})})]})}const be=Object.freeze(Object.defineProperty({__proto__:null,default:W},Symbol.toStringTag,{value:"Module"}));function U(){const[a,l]=d.useState(16),i=70,n=i*a/8,s=i*2,o=((1-n/s)*100).toFixed(1),r=a>=8?"Negligible":a>=4?"Minor (~1% accuracy drop)":"Significant";return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"LLM Quantization Memory Calculator"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Quantization bits: ",a,e.jsx("input",{type:"range",min:2,max:16,step:1,value:a,onChange:m=>l(parseInt(m.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("div",{className:"grid grid-cols-3 gap-3 text-sm text-center",children:[e.jsxs("div",{className:"p-2 rounded bg-violet-50 dark:bg-violet-900/20",children:[e.jsx("p",{className:"text-violet-700 dark:text-violet-300 font-medium",children:"70B Model Memory"}),e.jsxs("p",{className:"text-lg font-bold",children:[n.toFixed(1)," GB"]})]}),e.jsxs("div",{className:"p-2 rounded bg-violet-50 dark:bg-violet-900/20",children:[e.jsx("p",{className:"text-violet-700 dark:text-violet-300 font-medium",children:"Savings vs FP16"}),e.jsxs("p",{className:"text-lg font-bold",children:[o,"%"]})]}),e.jsxs("div",{className:"p-2 rounded bg-violet-50 dark:bg-violet-900/20",children:[e.jsx("p",{className:"text-violet-700 dark:text-violet-300 font-medium",children:"Quality Impact"}),e.jsx("p",{className:"text-sm font-bold",children:r})]})]}),e.jsx("div",{className:"mt-2 h-4 bg-gray-100 dark:bg-gray-800 rounded overflow-hidden",children:e.jsx("div",{className:"h-full bg-violet-500 transition-all duration-200",style:{width:`${n/s*100}%`}})}),e.jsxs("p",{className:"text-xs text-gray-500 mt-1 text-center",children:[n.toFixed(1)," GB / ",s," GB (FP16 baseline)"]})]})}function Q(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"LLM inference is memory-bandwidth bound during autoregressive decoding. Techniques like quantization, KV-cache optimization, and speculative decoding dramatically reduce cost while preserving quality."}),e.jsxs(u,{title:"KV-Cache and Memory Bandwidth Bottleneck",children:[e.jsx("p",{children:"During autoregressive generation, each token requires reading all model weights and the KV-cache:"}),e.jsx(t.BlockMath,{math:"\\text{KV cache size} = 2 \\times n_{\\text{layers}} \\times n_{\\text{heads}} \\times d_{\\text{head}} \\times \\text{seq\\_len} \\times \\text{bytes}"}),e.jsxs("p",{className:"mt-2",children:["For LLaMA-2 70B with 4K context: KV cache = ",e.jsx(t.InlineMath,{math:"2 \\times 80 \\times 64 \\times 128 \\times 4096 \\times 2 \\approx 10.7"})," GB in FP16. The arithmetic intensity is just ~1 FLOP/byte, making inference memory-bound."]})]}),e.jsx(U,{}),e.jsxs(p,{title:"Speculative Decoding",children:[e.jsxs("p",{children:["Use a small draft model to generate ",e.jsx(t.InlineMath,{math:"K"})," candidate tokens, then verify all at once with the large model:"]}),e.jsx(t.BlockMath,{math:"\\text{Speedup} \\approx \\frac{K}{1 + (1-\\alpha)K} \\quad \\text{where } \\alpha = P(\\text{draft accepted})"}),e.jsxs("p",{className:"mt-2",children:["With acceptance rate ",e.jsx(t.InlineMath,{math:"\\alpha = 0.8"})," and ",e.jsx(t.InlineMath,{math:"K = 5"})," draft tokens: ~2.5x speedup with ",e.jsx("strong",{children:"zero quality loss"})," (mathematically equivalent to sampling from the large model)."]})]}),e.jsx(f,{title:"Post-Training Quantization (Simplified)",code:`import torch

def absmax_quantize(tensor, bits=8):
    """Simple absmax quantization to int8/int4."""
    qmax = 2**(bits - 1) - 1
    scale = tensor.abs().max() / qmax
    quantized = (tensor / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return quantized, scale

def dequantize(quantized, scale):
    return quantized.float() * scale

# Simulate quantizing a weight matrix
W = torch.randn(4096, 4096)  # typical LLM layer
W_q, scale = absmax_quantize(W, bits=8)
W_deq = dequantize(W_q, scale)

# Measure quantization error
error = (W - W_deq).abs().mean()
print(f"Original dtype: {W.dtype}, size: {W.numel() * 4 / 1e6:.1f} MB")
print(f"Quantized dtype: {W_q.dtype}, size: {W_q.numel() * 1 / 1e6:.1f} MB")
print(f"Mean absolute error: {error:.6f}")
print(f"Relative error: {(error / W.abs().mean() * 100):.2f}%")
print(f"Memory savings: {(1 - 1/4) * 100:.0f}%")`}),e.jsx(h,{type:"note",title:"Grouped Query Attention (GQA)",children:e.jsx("p",{children:"GQA reduces KV-cache size by sharing key-value heads across multiple query heads. LLaMA-2 70B uses 8 KV-heads shared across 64 query heads, reducing KV-cache by 8x. Multi-Query Attention (MQA) takes this further with a single KV-head, though GQA offers a better quality-efficiency tradeoff."})}),e.jsxs(p,{title:"GPTQ and AWQ: Advanced Quantization",children:[e.jsx("p",{children:"Modern quantization methods go beyond simple absmax:"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"GPTQ:"})," Layer-wise quantization minimizing reconstruction error using Hessian information"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"AWQ:"})," Activation-aware quantization that protects salient weight channels"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"GGUF:"})," File format for CPU-friendly quantized inference (llama.cpp)"]}),e.jsx("li",{children:"4-bit GPTQ on LLaMA-2 70B: only ~1% accuracy drop, fits on a single 48GB GPU"}),e.jsxs("li",{children:[e.jsx("strong",{children:"FP8:"})," Native 8-bit floating point supported on H100 GPUs for near-lossless inference"]})]}),e.jsx("p",{className:"mt-2",children:"The trend toward lower precision continues: 2-bit and 1.58-bit (ternary) quantization are active research areas with promising early results."})]})]})}const je=Object.freeze(Object.defineProperty({__proto__:null,default:Q},Symbol.toStringTag,{value:"Module"}));function $(){const[a,l]=d.useState(0),i=[{label:"User Query",content:'"What is the weather in Paris today?"',actor:"User"},{label:"LLM Decides to Call Tool",content:'function_call: get_weather(location="Paris")',actor:"LLM"},{label:"Tool Execution",content:'{"temp": 18, "condition": "partly cloudy"}',actor:"System"},{label:"LLM Final Response",content:`"It's 18C and partly cloudy in Paris today."`,actor:"LLM"}],n=i[a],s={User:"bg-gray-100 dark:bg-gray-800",LLM:"bg-violet-50 dark:bg-violet-900/20",System:"bg-green-50 dark:bg-green-900/20"};return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Tool Use Flow"}),e.jsx("div",{className:"flex gap-1 mb-3",children:i.map((o,r)=>e.jsxs("button",{onClick:()=>l(r),className:`flex-1 px-2 py-1 rounded text-xs transition ${a===r?"bg-violet-500 text-white":"bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"}`,children:["Step ",r+1]},r))}),e.jsxs("div",{className:`p-3 rounded-lg ${s[n.actor]} text-sm`,children:[e.jsxs("p",{className:"font-medium text-gray-700 dark:text-gray-300",children:[n.label," ",e.jsxs("span",{className:"text-xs text-gray-500",children:["(",n.actor,")"]})]}),e.jsx("code",{className:"text-xs block mt-1 text-gray-600 dark:text-gray-400",children:n.content})]}),e.jsx("div",{className:"flex mt-2 gap-1",children:i.map((o,r)=>e.jsx("div",{className:`h-1 flex-1 rounded ${r<=a?"bg-violet-500":"bg-gray-200 dark:bg-gray-700"}`},r))})]})}function H(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Tool use extends LLMs beyond text generation by allowing them to invoke external functions, APIs, and databases. The model learns to generate structured function calls and incorporate results into its responses."}),e.jsxs(u,{title:"Function Calling Formulation",children:[e.jsxs("p",{children:["Given a user query ",e.jsx(t.InlineMath,{math:"q"})," and available tools ",e.jsx(t.InlineMath,{math:"\\mathcal{T} = \\{t_1, \\ldots, t_K\\}"}),", the model generates:"]}),e.jsx(t.BlockMath,{math:"a = \\text{LLM}(q, \\mathcal{T}) = \\begin{cases} \\text{text response} & \\text{if no tool needed} \\\\ (t_k, \\text{args}) & \\text{if tool } t_k \\text{ should be called} \\end{cases}"}),e.jsxs("p",{className:"mt-2",children:["The tool schema (name, description, parameters with types) is provided in the system prompt or as structured metadata. The model must decide ",e.jsx("em",{children:"whether"})," to call a tool, ",e.jsx("em",{children:"which"})," tool, and ",e.jsx("em",{children:"what arguments"})," to pass."]})]}),e.jsx($,{}),e.jsxs(p,{title:"Training for Tool Use",children:[e.jsx("p",{children:"Models learn tool use through:"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Fine-tuning:"})," Supervised training on (query, tool_call, result, response) tuples"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Self-play:"})," Model generates tool calls, executes them, and is trained on successful trajectories"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"RLHF:"})," Human preference on responses that use tools correctly vs incorrectly"]})]}),e.jsx("p",{className:"mt-2",children:"GPT-4 and Claude support parallel tool calls — invoking multiple tools simultaneously when appropriate."})]}),e.jsx(f,{title:"Simple Tool-Use Loop",code:`from dataclasses import dataclass
from typing import Any

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    function: Any  # callable

def tool_use_loop(query, tools, llm_generate, max_steps=5):
    """Execute a tool-use conversation loop.

    Args:
        query: user question
        tools: list of Tool objects
        llm_generate: function(messages, tools) -> response
    """
    messages = [{"role": "user", "content": query}]
    tool_schemas = [{"name": t.name, "description": t.description,
                     "parameters": t.parameters} for t in tools]

    for step in range(max_steps):
        response = llm_generate(messages, tool_schemas)

        if response.get("tool_call"):
            # Execute the tool
            tool_name = response["tool_call"]["name"]
            args = response["tool_call"]["arguments"]
            tool = next(t for t in tools if t.name == tool_name)
            result = tool.function(**args)

            messages.append({"role": "assistant", "tool_call": response["tool_call"]})
            messages.append({"role": "tool", "content": str(result)})
        else:
            return response["content"]  # Final text response

    return "Max tool-use steps reached"

# Example usage (pseudocode)
print("Tool-use loop: query -> [tool_call -> result]* -> final_response")
print("Key: LLM decides WHEN and WHICH tools to call")`}),e.jsx(h,{type:"note",title:"Structured Output and JSON Mode",children:e.jsx("p",{children:"Tool use relies on the model producing valid structured output (JSON). Constrained decoding techniques force the model to only generate tokens that form valid JSON matching the tool schema, eliminating parsing errors. This is critical for reliable agentic systems."})})]})}const _e=Object.freeze(Object.defineProperty({__proto__:null,default:H},Symbol.toStringTag,{value:"Module"}));function Z(){const[a,l]=d.useState("cot"),i={cot:{name:"Chain-of-Thought",desc:"Linear step-by-step reasoning in a single pass.",branches:1,verifications:0,strength:"Simple, no extra infrastructure"},sc:{name:"Self-Consistency",desc:"Sample multiple CoT paths, take majority vote.",branches:"K (e.g., 40)",verifications:0,strength:"Robust to individual errors"},tot:{name:"Tree-of-Thought",desc:"Explore multiple reasoning branches with backtracking.",branches:"b^d (branching)",verifications:"At each node",strength:"Handles complex search problems"}},n=i[a];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Reasoning Strategies Comparison"}),e.jsx("div",{className:"flex gap-2 mb-3 flex-wrap",children:Object.entries(i).map(([s,o])=>e.jsx("button",{onClick:()=>l(s),className:`px-3 py-1 rounded-lg text-sm transition ${a===s?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:o.name},s))}),e.jsxs("div",{className:"p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20 text-sm space-y-1",children:[e.jsx("p",{className:"font-medium text-violet-700 dark:text-violet-300",children:n.name}),e.jsx("p",{className:"text-gray-600 dark:text-gray-400",children:n.desc}),e.jsxs("p",{className:"text-xs text-gray-500",children:["Reasoning paths: ",n.branches," | Verifications: ",n.verifications," | Strength: ",n.strength]})]})]})}function J(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Chain-of-thought prompting unlocked LLM reasoning, but more sophisticated strategies — self-consistency, tree-of-thought, and process reward models — push reasoning accuracy further by exploring multiple solution paths and verifying intermediate steps."}),e.jsxs(u,{title:"Self-Consistency Decoding",children:[e.jsxs("p",{children:["Sample ",e.jsx(t.InlineMath,{math:"K"})," independent chain-of-thought solutions and take the majority vote on the final answer:"]}),e.jsx(t.BlockMath,{math:"\\hat{a} = \\arg\\max_{a} \\sum_{k=1}^{K} \\mathbb{1}[\\text{answer}(r_k) = a], \\quad r_k \\sim p(r | q, T)"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"r_k"})," are sampled reasoning chains at temperature ",e.jsx(t.InlineMath,{math:"T > 0"}),". This exploits the fact that correct reasoning paths are more common than any specific incorrect path."]})]}),e.jsx(Z,{}),e.jsxs(p,{title:"Process Reward Models (PRM)",children:[e.jsx("p",{children:"Instead of only scoring the final answer, PRMs score each reasoning step:"}),e.jsx(t.BlockMath,{math:"\\text{PRM}(r) = \\prod_{i=1}^{n} p(\\text{step}_i \\text{ is correct} | \\text{step}_{1:i})"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsx("li",{children:"Outcome Reward Model (ORM): only scores the final answer"}),e.jsx("li",{children:"Process Reward Model (PRM): scores every intermediate step"}),e.jsx("li",{children:"PRM + best-of-N: sample N solutions, pick the one with highest PRM score"}),e.jsx("li",{children:"On MATH benchmark: PRM + best-of-1860 achieves 78.2% (vs 72.4% majority voting)"})]})]}),e.jsx(f,{title:"Self-Consistency with Majority Voting",code:`import random
from collections import Counter

def self_consistency(generate_cot, question, K=40, temperature=0.7):
    """Self-consistency decoding: sample K chains, majority vote.

    Args:
        generate_cot: function(question, temp) -> (reasoning, answer)
        question: the input question
        K: number of samples
        temperature: sampling temperature
    Returns:
        best_answer: majority vote answer
        confidence: fraction of samples agreeing
    """
    answers = []
    for _ in range(K):
        reasoning, answer = generate_cot(question, temperature)
        answers.append(answer)

    # Majority voting
    counts = Counter(answers)
    best_answer = counts.most_common(1)[0][0]
    confidence = counts[best_answer] / K
    return best_answer, confidence

# Simulate: correct answer appears more often in diverse samples
def mock_cot(q, temp):
    # 70% chance of correct reasoning at high temperature
    answer = "42" if random.random() < 0.7 else random.choice(["41", "43", "44"])
    return "...", answer

ans, conf = self_consistency(mock_cot, "What is 6 * 7?", K=40)
print(f"Answer: {ans}, Confidence: {conf:.2f}")
# With 70% per-sample accuracy, majority vote >> individual accuracy`}),e.jsx(h,{type:"note",title:"Test-Time Compute Scaling",children:e.jsx("p",{children:"Self-consistency, tree-of-thought, and PRM-guided search all trade inference compute for accuracy. This creates a new scaling axis: instead of making models larger, spend more compute at inference time. For many reasoning tasks, doubling inference compute is more effective than doubling model parameters. OpenAI's o1 model and DeepSeek R1 demonstrate that training models specifically for extended reasoning produces large gains on math, coding, and science benchmarks. The key insight is to generate long reasoning traces during RL training, rewarding correct final answers regardless of the specific reasoning path taken. This teaches models to allocate variable compute based on problem difficulty."})})]})}const ve=Object.freeze(Object.defineProperty({__proto__:null,default:J},Symbol.toStringTag,{value:"Module"}));function X(){const[a,l]=d.useState(0),i=[{type:"Thought",content:"I need to find the population of France and Germany to compare them."},{type:"Action",content:'search("population of France 2024")'},{type:"Observation",content:"France population: approximately 68.2 million (2024)"},{type:"Thought",content:"Now I need Germany's population."},{type:"Action",content:'search("population of Germany 2024")'},{type:"Observation",content:"Germany population: approximately 84.5 million (2024)"},{type:"Thought",content:"Germany (84.5M) has a larger population than France (68.2M) by about 16.3 million."},{type:"Answer",content:"Germany has a larger population than France by approximately 16.3 million people."}],n={Thought:"bg-violet-50 dark:bg-violet-900/20 border-violet-200",Action:"bg-blue-50 dark:bg-blue-900/20 border-blue-200",Observation:"bg-green-50 dark:bg-green-900/20 border-green-200",Answer:"bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200"};return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"ReAct Trace Walkthrough"}),e.jsxs("div",{className:"flex items-center gap-2 mb-3",children:[e.jsx("button",{onClick:()=>l(Math.max(0,a-1)),disabled:a===0,className:"px-2 py-1 rounded text-sm bg-gray-100 dark:bg-gray-800 disabled:opacity-40",children:"Prev"}),e.jsxs("span",{className:"text-sm text-gray-500",children:["Step ",a+1,"/",i.length]}),e.jsx("button",{onClick:()=>l(Math.min(i.length-1,a+1)),disabled:a===i.length-1,className:"px-2 py-1 rounded text-sm bg-gray-100 dark:bg-gray-800 disabled:opacity-40",children:"Next"})]}),e.jsx("div",{className:"space-y-2",children:i.slice(0,a+1).map((s,o)=>e.jsxs("div",{className:`p-2 rounded border text-sm ${n[s.type]} ${o===a?"ring-2 ring-violet-400":"opacity-70"}`,children:[e.jsxs("span",{className:"font-medium text-xs",children:[s.type,":"]})," ",e.jsx("span",{className:"text-gray-700 dark:text-gray-300",children:s.content})]},o))})]})}function Y(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"LLM agents combine reasoning, tool use, and memory to autonomously complete complex multi-step tasks. Frameworks like ReAct, Reflexion, and multi-agent systems provide structured approaches to agentic behavior."}),e.jsxs(u,{title:"ReAct: Reasoning + Acting",children:[e.jsx("p",{children:"ReAct interleaves reasoning traces with tool actions in a loop:"}),e.jsx(t.BlockMath,{math:"\\text{Thought}_t \\to \\text{Action}_t \\to \\text{Observation}_t \\to \\text{Thought}_{t+1} \\to \\cdots"}),e.jsx("p",{className:"mt-2",children:"The thought step enables the model to plan and reflect before acting, while observations from the environment ground reasoning in real-world feedback. This outperforms both reasoning-only and acting-only approaches."})]}),e.jsx(X,{}),e.jsx(p,{title:"Agent Design Patterns",children:e.jsxs("ul",{className:"list-disc list-inside space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"ReAct:"})," Single-turn thought-action-observation loop"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Reflexion:"})," Self-reflection on failures to improve on retry"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Plan-and-Execute:"})," Create full plan first, then execute steps"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Multi-agent:"})," Specialized agents (coder, reviewer, planner) collaborate"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Hierarchical:"})," Manager agent delegates subtasks to worker agents"]})]})}),e.jsx(f,{title:"Minimal ReAct Agent Loop",code:`class ReActAgent:
    """Minimal ReAct agent with thought-action-observation loop."""
    def __init__(self, llm, tools, max_steps=10):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps

    def run(self, question):
        trajectory = [f"Question: {question}"]

        for step in range(self.max_steps):
            # Generate thought + action
            prompt = "\\n".join(trajectory)
            response = self.llm(prompt)  # Returns thought + action

            if "Answer:" in response:
                return response.split("Answer:")[-1].strip()

            # Parse action
            thought, action = self.parse_response(response)
            trajectory.append(f"Thought: {thought}")
            trajectory.append(f"Action: {action}")

            # Execute tool and get observation
            tool_name, args = self.parse_action(action)
            observation = self.tools[tool_name].execute(**args)
            trajectory.append(f"Observation: {observation}")

        return "Max steps reached without answer"

    def parse_response(self, response):
        # Extract thought and action from LLM output
        lines = response.strip().split("\\n")
        thought = lines[0].replace("Thought:", "").strip()
        action = lines[1].replace("Action:", "").strip() if len(lines) > 1 else ""
        return thought, action

    def parse_action(self, action_str):
        # Parse "tool_name(arg1, arg2)" format
        name = action_str.split("(")[0]
        args_str = action_str.split("(")[1].rstrip(")")
        return name, {"query": args_str}

print("ReAct loop: Think -> Act -> Observe -> Think -> ... -> Answer")`}),e.jsx(v,{title:"Agent Reliability Challenges",children:e.jsxs("p",{children:["Current LLM agents face compounding errors: if each step has 90% accuracy, a 10-step task succeeds only ",e.jsx(t.InlineMath,{math:"0.9^{10} \\approx 35\\%"})," of the time. Error recovery, verification, and human-in-the-loop checkpoints are essential for real-world deployment. The field is actively working on improving single-step reliability and adding self-correction."]})})]})}const ke=Object.freeze(Object.defineProperty({__proto__:null,default:Y},Symbol.toStringTag,{value:"Module"}));function ee(){const[a,l]=d.useState(8),[i,n]=d.useState(2),[s,o]=d.useState(42),r=c=>{let y=Math.sin(c)*1e4;return y-Math.floor(y)},m=Array.from({length:a},(c,y)=>r(s+y*7)),j=m.reduce((c,y)=>c+Math.exp(y*3),0),g=m.map(c=>Math.exp(c*3)/j),b=g.map((c,y)=>({p:c,i:y})).sort((c,y)=>y.p-c.p),_=new Set(b.slice(0,i).map(c=>c.i)),x=b.slice(0,i).reduce((c,y)=>c+y.p,0);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Expert Gating Network"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3 flex-wrap",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Experts: ",a,e.jsx("input",{type:"range",min:4,max:64,step:4,value:a,onChange:c=>l(parseInt(c.target.value)),className:"w-24 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Top-K: ",i,e.jsx("input",{type:"range",min:1,max:Math.min(8,a),step:1,value:i,onChange:c=>n(parseInt(c.target.value)),className:"w-24 accent-violet-500"})]}),e.jsx("button",{onClick:()=>o(s+1),className:"px-2 py-1 rounded text-xs bg-violet-100 text-violet-700 dark:bg-violet-900/30 dark:text-violet-300",children:"New Token"})]}),e.jsx("div",{className:"flex gap-1 items-end h-24",children:g.map((c,y)=>e.jsxs("div",{className:"flex-1 flex flex-col items-center",children:[e.jsx("div",{className:`w-full rounded-t transition-all ${_.has(y)?"bg-violet-500":"bg-gray-300 dark:bg-gray-600"}`,style:{height:`${c*300}px`}}),e.jsxs("span",{className:"text-[8px] text-gray-500 mt-1",children:["E",y]})]},y))}),e.jsxs("p",{className:"mt-2 text-xs text-gray-500 text-center",children:["Top-",i," experts activated (",(x*100).toFixed(1),"% of gating weight). Active FLOPs: ",i,"/",a," = ",(i/a*100).toFixed(0),"%"]})]})}function te(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:'Mixture of Experts (MoE) scales model parameters without proportionally scaling compute by routing each token through only a subset of "expert" sub-networks. This enables trillion-parameter models with practical training and inference costs.'}),e.jsxs(u,{title:"Sparse MoE Layer",children:[e.jsxs("p",{children:["An MoE layer replaces the standard FFN with a set of ",e.jsx(t.InlineMath,{math:"N"})," expert networks and a gating function:"]}),e.jsx(t.BlockMath,{math:"y = \\sum_{i=1}^{N} G(x)_i \\cdot E_i(x), \\quad G(x) = \\text{TopK}(\\text{softmax}(W_g x))"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"E_i"})," are expert FFNs and ",e.jsx(t.InlineMath,{math:"G(x)"})," is the gating network that selects the top-K experts. Only the selected experts compute their output, making the layer sparse."]})]}),e.jsx(ee,{}),e.jsxs(p,{title:"MoE Efficiency Gains",children:[e.jsxs("p",{children:["With ",e.jsx(t.InlineMath,{math:"N = 64"})," experts and ",e.jsx(t.InlineMath,{math:"K = 2"}),":"]}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsx("li",{children:"Total parameters: 64x the FFN size"}),e.jsx("li",{children:"Active parameters per token: 2x the FFN size (only 3.1% of experts)"}),e.jsx("li",{children:"FLOPs per token: ~2x a single expert (comparable to a dense model 64x smaller)"}),e.jsx("li",{children:"Example: Mixtral 8x7B has 47B total params but uses ~13B per token"})]})]}),e.jsx(f,{title:"Simple Top-K MoE Layer",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    """Mixture of Experts layer with top-k routing."""
    def __init__(self, dim=512, num_experts=8, top_k=2, expert_dim=2048):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, expert_dim), nn.ReLU(), nn.Linear(expert_dim, dim))
            for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: [batch, seq_len, dim]
        B, S, D = x.shape
        gate_logits = self.gate(x)                    # [B, S, num_experts]
        weights, indices = gate_logits.topk(self.top_k, dim=-1)  # [B, S, top_k]
        weights = F.softmax(weights, dim=-1)

        # Compute weighted sum of top-k expert outputs
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            for e_idx in range(len(self.experts)):
                mask = (indices[:, :, k] == e_idx)     # [B, S]
                if mask.any():
                    expert_input = x[mask]              # [num_tokens, D]
                    expert_out = self.experts[e_idx](expert_input)
                    output[mask] += weights[:, :, k][mask].unsqueeze(-1) * expert_out

        return output

moe = MoELayer(dim=512, num_experts=8, top_k=2)
x = torch.randn(2, 64, 512)
out = moe(x)
print(f"Input: {x.shape} -> Output: {out.shape}")
print(f"Total params: {sum(p.numel() for p in moe.parameters()):,}")`}),e.jsx(h,{type:"note",title:"Load Balancing Loss",children:e.jsxs("p",{children:["Without regularization, the gating network can collapse — routing all tokens to a few experts while others are unused. An auxiliary load balancing loss encourages uniform expert utilization: ",e.jsx(t.InlineMath,{math:"\\mathcal{L}_{\\text{aux}} = N \\cdot \\sum_{i=1}^{N} f_i \\cdot P_i"})," where",e.jsx(t.InlineMath,{math:"f_i"})," is the fraction of tokens routed to expert ",e.jsx(t.InlineMath,{math:"i"})," and",e.jsx(t.InlineMath,{math:"P_i"})," is the average gating probability for expert ",e.jsx(t.InlineMath,{math:"i"}),"."]})})]})}const Ne=Object.freeze(Object.defineProperty({__proto__:null,default:te},Symbol.toStringTag,{value:"Module"}));function ae(){const[a,l]=d.useState(128),[i,n]=d.useState(1.25),[s,o]=d.useState(4096),r=Math.ceil(s/a*i),j=((r*a/s-1)*100).toFixed(1);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Expert Capacity Calculator"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3 flex-wrap",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Experts: ",a,e.jsx("input",{type:"range",min:8,max:256,step:8,value:a,onChange:g=>l(parseInt(g.target.value)),className:"w-24 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Capacity factor: ",i.toFixed(2),e.jsx("input",{type:"range",min:1,max:2,step:.05,value:i,onChange:g=>n(parseFloat(g.target.value)),className:"w-24 accent-violet-500"})]})]}),e.jsxs("div",{className:"grid grid-cols-3 gap-3 text-sm text-center",children:[e.jsxs("div",{className:"p-2 rounded bg-violet-50 dark:bg-violet-900/20",children:[e.jsx("p",{className:"text-violet-700 dark:text-violet-300 font-medium",children:"Tokens/Expert"}),e.jsx("p",{className:"font-bold",children:r})]}),e.jsxs("div",{className:"p-2 rounded bg-violet-50 dark:bg-violet-900/20",children:[e.jsx("p",{className:"text-violet-700 dark:text-violet-300 font-medium",children:"Batch Tokens"}),e.jsx("p",{className:"font-bold",children:s})]}),e.jsxs("div",{className:"p-2 rounded bg-violet-50 dark:bg-violet-900/20",children:[e.jsx("p",{className:"text-violet-700 dark:text-violet-300 font-medium",children:"Buffer Overhead"}),e.jsxs("p",{className:"font-bold",children:[j,"%"]})]})]}),e.jsx("p",{className:"mt-2 text-xs text-gray-500 text-center",children:"Capacity factor > 1 provides buffer for imbalanced routing. Tokens exceeding capacity are dropped."})]})}function se(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Switch Transformer simplifies MoE routing to top-1 expert selection, achieving better scaling with reduced communication cost. GShard demonstrated trillion-parameter MoE models across thousands of devices with expert parallelism."}),e.jsxs(u,{title:"Switch Routing (Top-1)",children:[e.jsx("p",{children:"Switch Transformer routes each token to exactly one expert, simplifying the MoE formulation:"}),e.jsx(t.BlockMath,{math:"y = G(x)_{i^*} \\cdot E_{i^*}(x), \\quad i^* = \\arg\\max_i (W_g x)_i"}),e.jsxs("p",{className:"mt-2",children:["The gating weight ",e.jsx(t.InlineMath,{math:"G(x)_{i^*}"})," acts as a confidence score. The expert capacity ",e.jsx(t.InlineMath,{math:"C"})," limits tokens per expert per batch:"]}),e.jsx(t.BlockMath,{math:"C = \\text{CF} \\times \\frac{\\text{tokens\\_in\\_batch}}{N_{\\text{experts}}}"}),e.jsx("p",{className:"mt-1",children:"where CF is the capacity factor (typically 1.0-1.5). Tokens routed to a full expert are dropped."})]}),e.jsx(ae,{}),e.jsxs(p,{title:"Switch Transformer Scaling Results",children:[e.jsx("p",{children:"Switch Transformer demonstrates superior scaling over dense models:"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsx("li",{children:"1.6T parameters with 128 experts, matching T5-XXL quality with 7x fewer training steps"}),e.jsx("li",{children:"Same FLOPs as T5-Base but 7x more parameters → significant quality improvement"}),e.jsx("li",{children:"Expert parallelism: each expert on a separate device, all-to-all communication between them"}),e.jsx("li",{children:"Key finding: top-1 routing works as well as top-2 with simpler implementation"})]})]}),e.jsx(f,{title:"Switch Router with Capacity and Load Balancing",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class SwitchRouter(nn.Module):
    """Top-1 expert routing with capacity and load balancing."""
    def __init__(self, dim, num_experts, capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x):
        # x: [B*S, D]
        logits = self.gate(x)                           # [B*S, E]
        probs = F.softmax(logits, dim=-1)

        # Top-1 selection
        gate_values, expert_idx = probs.max(dim=-1)     # [B*S]
        num_tokens = x.shape[0]
        capacity = int(self.capacity_factor * num_tokens / self.num_experts)

        # Build dispatch mask with capacity constraint
        dispatch = torch.zeros(num_tokens, self.num_experts, device=x.device)
        expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)

        for i in range(num_tokens):
            e = expert_idx[i].item()
            if expert_counts[e] < capacity:
                dispatch[i, e] = gate_values[i]
                expert_counts[e] += 1
            # else: token is dropped (overflow)

        # Load balancing loss
        f = expert_counts.float() / num_tokens  # fraction routed
        P = probs.mean(dim=0)                     # average probability
        aux_loss = self.num_experts * (f * P).sum()

        return dispatch, aux_loss, expert_counts

router = SwitchRouter(dim=512, num_experts=8)
tokens = torch.randn(64, 512)
dispatch, loss, counts = router(tokens)
print(f"Expert load: {counts.tolist()}")
print(f"Auxiliary loss: {loss.item():.4f}")
print(f"Dropped tokens: {64 - dispatch.sum().item():.0f}")`}),e.jsx(h,{type:"note",title:"Expert Parallelism in Distributed Training",children:e.jsx("p",{children:"GShard places each expert on a separate accelerator. An all-to-all communication step dispatches tokens to their assigned experts across devices, then gathers results back. This is communication-efficient because each token goes to exactly one expert (top-1), minimizing cross-device traffic. With 2048 experts across 2048 TPUs, GShard trained a 600B parameter model."})})]})}const we=Object.freeze(Object.defineProperty({__proto__:null,default:se},Symbol.toStringTag,{value:"Module"}));function ie(){const[a,l]=d.useState("mixtral"),i={mixtral:{name:"Mixtral 8x7B",totalParams:"46.7B",activeParams:"12.9B",experts:8,topK:2,performance:"Matches LLaMA-2 70B"},llama70:{name:"LLaMA-2 70B",totalParams:"70B",activeParams:"70B",experts:1,topK:1,performance:"Dense baseline"},gpt35:{name:"GPT-3.5 (est.)",totalParams:"~175B (MoE?)",activeParams:"~20-30B",experts:"~16",topK:"~2",performance:"Reference commercial model"}},n=i[a];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"MoE Model Comparison"}),e.jsx("div",{className:"flex gap-2 mb-3 flex-wrap",children:Object.entries(i).map(([s,o])=>e.jsx("button",{onClick:()=>l(s),className:`px-3 py-1 rounded-lg text-sm transition ${a===s?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:o.name},s))}),e.jsx("div",{className:"grid grid-cols-2 gap-2 text-sm",children:Object.entries({"Total Params":n.totalParams,"Active Params/Token":n.activeParams,Experts:n.experts,"Top-K":n.topK,Performance:n.performance}).map(([s,o])=>e.jsxs("div",{className:"p-2 rounded bg-violet-50 dark:bg-violet-900/20",children:[e.jsx("span",{className:"text-xs text-gray-500",children:s}),e.jsx("p",{className:"font-medium text-gray-700 dark:text-gray-300",children:o})]},s))})]})}function ne(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Mixtral 8x7B by Mistral AI demonstrated that open-source MoE models can match or exceed much larger dense models. It uses 8 expert FFN blocks with top-2 routing, achieving LLaMA-2 70B quality at a fraction of the inference cost."}),e.jsxs(u,{title:"Mixtral Architecture",children:[e.jsx("p",{children:"Mixtral replaces each Transformer FFN with 8 expert FFNs, using top-2 routing:"}),e.jsx(t.BlockMath,{math:"\\text{FFN}_{\\text{MoE}}(x) = \\sum_{i \\in \\text{Top2}(G(x))} g_i(x) \\cdot \\text{FFN}_i(x)"}),e.jsx("p",{className:"mt-2",children:"All other components (attention, normalization) are shared across experts. Total parameters: 46.7B. Active parameters per token: 12.9B (two 6.45B experts). This gives 70B-class quality with 13B-class inference cost."})]}),e.jsx(ie,{}),e.jsxs(p,{title:"Expert Specialization in Mixtral",children:[e.jsx("p",{children:"Analysis of Mixtral's routing reveals soft specialization patterns:"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsx("li",{children:"Experts show domain preferences but are not strictly specialized"}),e.jsx("li",{children:"Routing is primarily syntax-driven (e.g., by token position, not semantics)"}),e.jsx("li",{children:"Different layers show different specialization patterns"}),e.jsx("li",{children:"No single expert can be removed without degrading all domains"})]})]}),e.jsx(f,{title:"Mixtral-Style MoE Forward Pass",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class MixtralMoEBlock(nn.Module):
    """Mixtral-style MoE with 8 experts and top-2 routing."""
    def __init__(self, dim=4096, ffn_dim=14336, num_experts=8):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(dim, num_experts, bias=False)

        # Each expert is a standard SwiGLU FFN
        self.w1 = nn.ModuleList([nn.Linear(dim, ffn_dim, bias=False) for _ in range(num_experts)])
        self.w2 = nn.ModuleList([nn.Linear(ffn_dim, dim, bias=False) for _ in range(num_experts)])
        self.w3 = nn.ModuleList([nn.Linear(dim, ffn_dim, bias=False) for _ in range(num_experts)])

    def expert_fn(self, x, idx):
        """SwiGLU expert: w2(SiLU(w1(x)) * w3(x))"""
        return self.w2[idx](F.silu(self.w1[idx](x)) * self.w3[idx](x))

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(-1, D)

        # Gate and select top-2
        logits = self.gate(x_flat)
        weights, indices = logits.topk(2, dim=-1)
        weights = F.softmax(weights, dim=-1)

        # Weighted combination of top-2 experts
        output = torch.zeros_like(x_flat)
        for k in range(2):
            for e in range(self.num_experts):
                mask = (indices[:, k] == e)
                if mask.any():
                    expert_out = self.expert_fn(x_flat[mask], e)
                    output[mask] += weights[mask, k:k+1] * expert_out

        return output.view(B, S, D)

# Mixtral dimensions (scaled down for demo)
moe = MixtralMoEBlock(dim=256, ffn_dim=512, num_experts=8)
x = torch.randn(1, 32, 256)
out = moe(x)
total_p = sum(p.numel() for p in moe.parameters())
print(f"Output: {out.shape}, Total params: {total_p:,}")`}),e.jsx(h,{type:"note",title:"MoE Deployment Challenges",children:e.jsx("p",{children:"MoE models require all expert weights in memory even though only a few are active per token. Mixtral 8x7B needs ~90GB in FP16 (all 47B params loaded), limiting it to multi-GPU setups. Expert offloading (keeping idle experts on CPU/disk) and expert merging are active research areas for making MoE models more practical for deployment."})})]})}const Me=Object.freeze(Object.defineProperty({__proto__:null,default:ne},Symbol.toStringTag,{value:"Module"}));export{de as a,me as b,he as c,pe as d,xe as e,ge as f,ue as g,fe as h,ye as i,be as j,je as k,_e as l,ve as m,ke as n,Ne as o,we as p,Me as q,ce as s};

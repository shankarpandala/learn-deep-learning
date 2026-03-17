import{j as e,r as h}from"./vendor-DpISuAX6.js";import{r as t}from"./vendor-katex-CbWCYdth.js";import{D as c,E as g,P as f,N as y,W as k,T as b}from"./subject-01-foundations-D0A1VJsr.js";function N(){const[a,d]=h.useState(1),n=["I","love","deep","learning"],o=[.8,2.5,1.2,3.1];function i(r,l){const m=r.map(u=>u/l),x=Math.max(...m),p=m.map(u=>Math.exp(u-x)),j=p.reduce((u,_)=>u+_,0);return p.map(u=>u/j)}const s=i(o,a);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Attention Weight Visualization"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-4",children:["Temperature: ",a.toFixed(2),e.jsx("input",{type:"range",min:.1,max:3,step:.05,value:a,onChange:r=>d(parseFloat(r.target.value)),className:"w-40 accent-violet-500"})]}),e.jsx("div",{className:"flex gap-3 justify-center",children:n.map((r,l)=>e.jsxs("div",{className:"flex flex-col items-center gap-1",children:[e.jsx("div",{className:"w-16 rounded",style:{height:`${Math.max(4,s[l]*120)}px`,backgroundColor:`rgba(139, 92, 246, ${.3+s[l]*.7})`}}),e.jsx("span",{className:"text-xs font-mono text-gray-700 dark:text-gray-300",children:r}),e.jsx("span",{className:"text-xs text-violet-600 dark:text-violet-400",children:s[l].toFixed(3)})]},r))})]})}function M(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"The attention mechanism allows a model to dynamically focus on different parts of the input when producing each output element. The Query-Key-Value (QKV) framework provides an elegant formulation for computing these attention weights."}),e.jsxs(c,{title:"Queries, Keys, and Values",children:[e.jsxs("p",{children:["Given input embeddings ",e.jsx(t.InlineMath,{math:"X \\in \\mathbb{R}^{n \\times d}"}),", we project into three spaces:"]}),e.jsx(t.BlockMath,{math:"Q = XW^Q, \\quad K = XW^K, \\quad V = XW^V"}),e.jsxs("p",{className:"mt-2",children:[e.jsx("strong",{children:"Query"})," — what am I looking for? ",e.jsx("strong",{children:"Key"})," — what do I contain?",e.jsx("strong",{children:" Value"})," — what information do I provide?"]})]}),e.jsxs(c,{title:"Scaled Dot-Product Attention",children:[e.jsx(t.BlockMath,{math:"\\text{Attention}(Q, K, V) = \\text{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right) V"}),e.jsxs("p",{className:"mt-2",children:["The scaling factor ",e.jsx(t.InlineMath,{math:"\\sqrt{d_k}"})," prevents the dot products from growing large in magnitude, which would push the softmax into regions with extremely small gradients."]})]}),e.jsx(N,{}),e.jsxs(g,{title:"Why Scale by sqrt(d_k)?",children:[e.jsxs("p",{children:["If ",e.jsx(t.InlineMath,{math:"q, k \\in \\mathbb{R}^{d_k}"})," have components drawn i.i.d. from ",e.jsx(t.InlineMath,{math:"\\mathcal{N}(0,1)"}),", then:"]}),e.jsx(t.BlockMath,{math:"\\text{Var}(q \\cdot k) = \\sum_{i=1}^{d_k} \\text{Var}(q_i k_i) = d_k"}),e.jsxs("p",{children:["Dividing by ",e.jsx(t.InlineMath,{math:"\\sqrt{d_k}"})," normalizes the variance back to 1."]})]}),e.jsx(f,{title:"Scaled Dot-Product Attention in PyTorch",code:`import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Compute scaled dot-product attention.

    Args:
        Q: (batch, seq_q, d_k)
        K: (batch, seq_k, d_k)
        V: (batch, seq_k, d_v)
        mask: optional (batch, seq_q, seq_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights

# Example: batch=1, seq_len=4, d_k=d_v=8
Q = torch.randn(1, 4, 8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)
output, attn_weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")       # (1, 4, 8)
print(f"Attention weights:\\n{attn_weights}")`}),e.jsx(y,{type:"note",title:"Attention as Soft Dictionary Lookup",children:e.jsx("p",{children:"Think of attention as a differentiable dictionary. The query looks up the most relevant keys, and the returned value is a weighted combination of all values. Unlike a hard lookup, every entry contributes — just with different weights determined by the query-key similarity."})})]})}const se=Object.freeze(Object.defineProperty({__proto__:null,default:M},Symbol.toStringTag,{value:"Module"}));function q(){const[a,d]=h.useState(1),n=["The","cat","sat","on"],o=[[1,.3,.1,.2],[.2,1,.5,.1],[.1,.6,1,.8],[.3,.1,.7,1]];function i(l,m){const x=l.map(_=>_/m),p=Math.max(...x),j=x.map(_=>Math.exp(_-p)),u=j.reduce((_,v)=>_+v,0);return j.map(_=>_/u)}const s=o.map(l=>i(l,a)),r=52;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Attention Heatmap"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Temperature: ",a.toFixed(2),e.jsx("input",{type:"range",min:.1,max:3,step:.05,value:a,onChange:l=>d(parseFloat(l.target.value)),className:"w-40 accent-violet-500"})]}),e.jsx("div",{className:"overflow-x-auto flex justify-center",children:e.jsxs("table",{className:"border-collapse",children:[e.jsx("thead",{children:e.jsxs("tr",{children:[e.jsx("td",{}),n.map(l=>e.jsx("th",{className:"text-xs text-gray-500 dark:text-gray-400 px-1 pb-1 font-mono",children:l},l))]})}),e.jsx("tbody",{children:n.map((l,m)=>e.jsxs("tr",{children:[e.jsx("td",{className:"text-xs text-gray-500 dark:text-gray-400 pr-2 font-mono",children:l}),s[m].map((x,p)=>e.jsx("td",{style:{width:r,height:r,backgroundColor:`rgba(139, 92, 246, ${x})`},className:"text-center text-xs font-mono text-gray-800 dark:text-gray-100 border border-gray-200 dark:border-gray-700",children:x.toFixed(2)},p))]},l))})]})})]})}function A(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Attention score distributions determine how the model allocates focus across input positions. Understanding how temperature and score magnitude shape these distributions is critical for diagnosing and tuning transformer behavior."}),e.jsxs(c,{title:"Softmax Temperature",children:[e.jsx(t.BlockMath,{math:"\\text{softmax}(z_i / \\tau) = \\frac{e^{z_i / \\tau}}{\\sum_j e^{z_j / \\tau}}"}),e.jsxs("p",{className:"mt-2",children:["As ",e.jsx(t.InlineMath,{math:"\\tau \\to 0"}),", the distribution approaches a one-hot vector (hard attention). As ",e.jsx(t.InlineMath,{math:"\\tau \\to \\infty"}),", it approaches a uniform distribution."]})]}),e.jsx(q,{}),e.jsxs(g,{title:"Entropy of Attention Weights",children:[e.jsx("p",{children:"The entropy of the attention distribution measures how spread the focus is:"}),e.jsx(t.BlockMath,{math:"H(\\alpha) = -\\sum_i \\alpha_i \\log \\alpha_i"}),e.jsxs("p",{children:["Uniform over ",e.jsx(t.InlineMath,{math:"n"})," tokens gives ",e.jsx(t.InlineMath,{math:"H = \\log n"})," (maximum entropy). A peaked distribution gives ",e.jsx(t.InlineMath,{math:"H \\approx 0"}),"."]})]}),e.jsx(k,{title:"Attention Collapse",children:e.jsxs("p",{children:["When attention weights become too peaked (low entropy), the model may ignore useful context — a phenomenon called ",e.jsx("em",{children:"attention collapse"}),". This can happen when the model overfits or when temperature scaling is improperly tuned."]})}),e.jsx(f,{title:"Visualizing Attention Score Distributions",code:`import torch
import torch.nn.functional as F

scores = torch.tensor([1.0, 2.5, 0.8, 3.1])

# Effect of temperature on attention distribution
for temp in [0.5, 1.0, 2.0, 5.0]:
    weights = F.softmax(scores / temp, dim=-1)
    entropy = -(weights * weights.log()).sum().item()
    print(f"T={temp:.1f}: weights={weights.numpy().round(3)}, H={entropy:.3f}")

# Output shows sharper distributions at low temp,
# more uniform at high temp`}),e.jsx(y,{type:"note",title:"Common Attention Patterns",children:e.jsxs("p",{children:["Trained transformers exhibit recurring patterns: ",e.jsx("strong",{children:"diagonal"})," (attending to same position),",e.jsx("strong",{children:"vertical stripes"})," (attending to specific tokens like [CLS] or punctuation), and ",e.jsx("strong",{children:"broad"})," (roughly uniform). Different heads learn different patterns, enabling the model to capture diverse relationships."]})})]})}const re=Object.freeze(Object.defineProperty({__proto__:null,default:A},Symbol.toStringTag,{value:"Module"}));function T(){const[a,d]=h.useState("multiplicative"),n={additive:{name:"Additive (Bahdanau)",formula:"score(s_i, h_j) = v^\\top \\tanh(W_1 s_i + W_2 h_j)",pros:["More expressive with learnable parameters","Works well with different Q/K dimensions"],cons:["Slower — requires feed-forward pass per pair","More parameters to train"]},multiplicative:{name:"Multiplicative (Luong)",formula:"score(s_i, h_j) = s_i^\\top W h_j",pros:["Efficient — single matrix multiply","Easily batched on GPUs"],cons:["Assumes Q and K have compatible dimensions","Can suffer from large dot products without scaling"]}},o=n[a];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Attention Variant Comparison"}),e.jsx("div",{className:"flex gap-3 mb-4",children:Object.keys(n).map(i=>e.jsx("button",{onClick:()=>d(i),className:`px-3 py-1.5 rounded-lg text-sm font-medium transition ${a===i?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:n[i].name},i))}),e.jsx(t.BlockMath,{math:o.formula}),e.jsxs("div",{className:"grid grid-cols-2 gap-4 mt-3 text-sm",children:[e.jsxs("div",{children:[e.jsx("p",{className:"font-semibold text-green-600 dark:text-green-400 mb-1",children:"Advantages"}),e.jsx("ul",{className:"list-disc ml-4 text-gray-600 dark:text-gray-400",children:o.pros.map((i,s)=>e.jsx("li",{children:i},s))})]}),e.jsxs("div",{children:[e.jsx("p",{className:"font-semibold text-red-500 dark:text-red-400 mb-1",children:"Disadvantages"}),e.jsx("ul",{className:"list-disc ml-4 text-gray-600 dark:text-gray-400",children:o.cons.map((i,s)=>e.jsx("li",{children:i},s))})]})]})]})}function B(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Before the modern transformer, two major attention variants emerged from the sequence-to-sequence literature: additive attention (Bahdanau et al., 2015) and multiplicative attention (Luong et al., 2015). Understanding their differences illuminates key design choices in attention mechanisms."}),e.jsxs(c,{title:"Additive Attention (Bahdanau)",children:[e.jsx(t.BlockMath,{math:"e_{ij} = v^\\top \\tanh(W_1 s_i + W_2 h_j)"}),e.jsxs("p",{className:"mt-2",children:["Uses a learned feed-forward network with weight matrices ",e.jsx(t.InlineMath,{math:"W_1, W_2"})," and vector ",e.jsx(t.InlineMath,{math:"v"})," to compute alignment scores between decoder state",e.jsx(t.InlineMath,{math:"s_i"})," and encoder hidden state ",e.jsx(t.InlineMath,{math:"h_j"}),"."]})]}),e.jsxs(c,{title:"Multiplicative Attention (Luong)",children:[e.jsx(t.BlockMath,{math:"e_{ij} = s_i^\\top W h_j \\quad \\text{(general)} \\quad \\text{or} \\quad e_{ij} = s_i^\\top h_j \\quad \\text{(dot)}"}),e.jsxs("p",{className:"mt-2",children:["The dot-product variant (without ",e.jsx(t.InlineMath,{math:"W"}),") becomes scaled dot-product attention when divided by ",e.jsx(t.InlineMath,{math:"\\sqrt{d_k}"})," — the basis of the Transformer."]})]}),e.jsx(T,{}),e.jsxs(b,{title:"Computational Complexity",id:"attention-complexity",children:[e.jsxs("p",{children:["For sequence length ",e.jsx(t.InlineMath,{math:"n"})," and dimension ",e.jsx(t.InlineMath,{math:"d"}),":"]}),e.jsx(t.BlockMath,{math:"\\text{Additive: } O(n^2 \\cdot d) \\quad \\text{Multiplicative: } O(n^2 \\cdot d)"}),e.jsxs("p",{className:"mt-2",children:["Both are ",e.jsx(t.InlineMath,{math:"O(n^2)"})," in sequence length, but multiplicative attention has a much smaller constant factor due to optimized matrix multiplication on modern hardware."]})]}),e.jsx(f,{title:"Additive vs Multiplicative Attention",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W1 = nn.Linear(dim, dim, bias=False)
        self.W2 = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, query, keys, values):
        # query: (B, 1, D), keys: (B, N, D)
        scores = self.v(torch.tanh(self.W1(query) + self.W2(keys)))
        weights = F.softmax(scores.squeeze(-1), dim=-1)
        return torch.bmm(weights.unsqueeze(1), values)

class MultiplicativeAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5

    def forward(self, query, keys, values):
        scores = torch.bmm(query, keys.transpose(1, 2)) / self.scale
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, values)

d = 64
additive = AdditiveAttention(d)
multiplicative = MultiplicativeAttention(d)
q = torch.randn(2, 1, d)
k = v = torch.randn(2, 10, d)
print("Additive output:", additive(q, k, v).shape)
print("Multiplicative output:", multiplicative(q, k, v).shape)`}),e.jsx(y,{type:"note",title:"Why Transformers Use Multiplicative Attention",children:e.jsx("p",{children:"The Transformer adopts scaled dot-product (multiplicative) attention because it can be computed entirely with matrix multiplications, which are highly optimized on GPUs. The scaling factor addresses the gradient issues of large dot products, making it both fast and numerically stable."})}),e.jsx(g,{title:"Historical Timeline",children:e.jsxs("p",{children:[e.jsx("strong",{children:"2015:"})," Bahdanau introduces additive attention for machine translation."," ",e.jsx("strong",{children:"2015:"})," Luong proposes multiplicative variants."," ",e.jsx("strong",{children:"2017:"})," Vaswani et al. use scaled dot-product attention in the Transformer, dispensing with recurrence entirely."]})})]})}const ie=Object.freeze(Object.defineProperty({__proto__:null,default:B},Symbol.toStringTag,{value:"Module"}));function S(){const[a,d]=h.useState(1),n=["The","bank","of","the","river"],i=[[.35,.15,.1,.1,.3],[.08,.3,.07,.05,.5],[.2,.25,.15,.2,.2],[.3,.1,.15,.25,.2],[.1,.45,.1,.05,.3]][a];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Self-Attention: Which tokens does each word attend to?"}),e.jsx("div",{className:"flex gap-2 mb-4 mt-3",children:n.map((s,r)=>e.jsx("button",{onClick:()=>d(r),className:`px-3 py-1 rounded-lg text-sm font-medium transition ${a===r?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:s},r))}),e.jsxs("p",{className:"text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Query: ",e.jsx("strong",{className:"text-violet-600 dark:text-violet-400",children:n[a]})," attends to:"]}),e.jsx("div",{className:"flex gap-3 justify-center",children:n.map((s,r)=>e.jsxs("div",{className:"flex flex-col items-center gap-1",children:[e.jsx("div",{className:"w-14 rounded",style:{height:`${Math.max(4,i[r]*100)}px`,backgroundColor:`rgba(139, 92, 246, ${.2+i[r]*.8})`}}),e.jsx("span",{className:"text-xs font-mono text-gray-700 dark:text-gray-300",children:s}),e.jsx("span",{className:"text-xs text-violet-600 dark:text-violet-400",children:i[r].toFixed(2)})]},r))})]})}function I(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Self-attention lets every position in a sequence attend to every other position, enabling the model to capture long-range dependencies in a single operation. Unlike RNNs, the path length between any two positions is O(1), not O(n)."}),e.jsxs(c,{title:"Self-Attention",children:[e.jsx("p",{children:"In self-attention, the queries, keys, and values all come from the same sequence:"}),e.jsx(t.BlockMath,{math:"Q = XW^Q, \\quad K = XW^K, \\quad V = XW^V"}),e.jsx(t.BlockMath,{math:"\\text{SelfAttn}(X) = \\text{softmax}\\!\\left(\\frac{(XW^Q)(XW^K)^\\top}{\\sqrt{d_k}}\\right)(XW^V)"})]}),e.jsx(S,{}),e.jsx(g,{title:"Disambiguation via Self-Attention",children:e.jsxs("p",{children:['Consider "The ',e.jsx("strong",{children:"bank"}),' of the river" vs "The ',e.jsx("strong",{children:"bank"}),' approved the loan." Self-attention allows "bank" to attend to surrounding context (river vs. loan), enabling different representations of the same word depending on context.']})}),e.jsxs(b,{title:"Self-Attention Complexity",id:"self-attn-complexity",children:[e.jsx("p",{children:"Self-attention has:"}),e.jsx(t.BlockMath,{math:"\\text{Time: } O(n^2 \\cdot d), \\quad \\text{Memory: } O(n^2 + n \\cdot d)"}),e.jsxs("p",{className:"mt-2",children:["The ",e.jsx(t.InlineMath,{math:"O(n^2)"})," term comes from computing all pairwise attention scores. This quadratic cost is the primary bottleneck for long sequences."]})]}),e.jsx(f,{title:"Self-Attention from Scratch",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)
        self.scale = d_k ** 0.5

    def forward(self, x):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V), attn

# Example: batch=2, seq_len=5, d_model=16, d_k=8
x = torch.randn(2, 5, 16)
sa = SelfAttention(d_model=16, d_k=8)
output, attn_weights = sa(x)
print(f"Output: {output.shape}")          # (2, 5, 8)
print(f"Attn weights: {attn_weights.shape}")  # (2, 5, 5)`}),e.jsx(y,{type:"note",title:"Self-Attention vs Convolution vs Recurrence",children:e.jsx("p",{children:"Self-attention connects all positions with O(1) path length and O(n) sequential operations (parallelizable). Convolutions have O(n/k) path length and are also parallel, but with limited receptive field. Recurrence has O(n) path length and O(n) sequential steps (not parallelizable)."})})]})}const oe=Object.freeze(Object.defineProperty({__proto__:null,default:I},Symbol.toStringTag,{value:"Module"}));function L(){const[a,d]=h.useState(4),n=256,o=n/a,i=["#8b5cf6","#a78bfa","#c4b5fd","#7c3aed","#6d28d9","#5b21b6","#4c1d95","#ddd6fe"];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Multi-Head Attention Structure"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Number of heads: ",a,e.jsx("input",{type:"range",min:1,max:8,step:1,value:a,onChange:s=>d(parseInt(s.target.value)),className:"w-32 accent-violet-500"})]}),e.jsxs("p",{className:"text-sm text-gray-600 dark:text-gray-400 mb-3",children:[e.jsx(t.InlineMath,{math:`d_{\\text{model}} = ${n}`}),", each head: ",e.jsx(t.InlineMath,{math:`d_k = d_v = ${o}`})]}),e.jsx("div",{className:"flex gap-1 justify-center flex-wrap",children:Array.from({length:a},(s,r)=>e.jsxs("div",{className:"flex flex-col items-center rounded-lg p-2 border border-gray-200 dark:border-gray-700",style:{backgroundColor:i[r]+"20"},children:[e.jsxs("div",{className:"text-xs font-bold mb-1",style:{color:i[r]},children:["Head ",r+1]}),e.jsx("div",{className:"text-xs text-gray-500 dark:text-gray-400",children:"Q K V"}),e.jsxs("div",{className:"text-xs text-gray-500 dark:text-gray-400",children:[o,"d"]})]},r))}),e.jsxs("div",{className:"text-center mt-3 text-sm text-gray-600 dark:text-gray-400",children:["Concat all heads → Linear projection → ",e.jsx(t.InlineMath,{math:`\\mathbb{R}^{${n}}`})]})]})}function V(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Multi-head attention runs several attention functions in parallel, allowing the model to jointly attend to information from different representation subspaces at different positions. A single attention head tends to average over multiple patterns — multiple heads allow specialization."}),e.jsxs(c,{title:"Multi-Head Attention",children:[e.jsx(t.BlockMath,{math:"\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\ldots, \\text{head}_h) W^O"}),e.jsx(t.BlockMath,{math:"\\text{where head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)"}),e.jsxs("p",{className:"mt-2",children:["With ",e.jsx(t.InlineMath,{math:"h"})," heads and model dimension ",e.jsx(t.InlineMath,{math:"d_{\\text{model}}"}),", each head operates on ",e.jsx(t.InlineMath,{math:"d_k = d_v = d_{\\text{model}} / h"}),"."]})]}),e.jsx(L,{}),e.jsxs(b,{title:"Parameter Count",id:"mha-params",children:[e.jsxs("p",{children:["Multi-head attention with ",e.jsx(t.InlineMath,{math:"h"})," heads has the same parameter count as single-head:"]}),e.jsx(t.BlockMath,{math:"3 \\cdot d_{\\text{model}} \\cdot d_k \\cdot h + d_{\\text{model}}^2 = 4 \\cdot d_{\\text{model}}^2"}),e.jsxs("p",{className:"mt-2",children:["Since ",e.jsx(t.InlineMath,{math:"d_k = d_{\\text{model}} / h"}),", the total computation is the same as single-head attention with full dimensionality, but with richer representations."]})]}),e.jsx(g,{title:"What Different Heads Learn",children:e.jsx("p",{children:"Research shows heads specialize: some track syntactic relations (subject-verb), others coreference (pronoun-antecedent), and others positional patterns (attending to adjacent tokens). This diversity is key to the transformer's representational power."})}),e.jsx(f,{title:"Multi-Head Attention Implementation",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        qkv = self.W_qkv(x).reshape(B, N, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, h, N, d_k)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, V)  # (B, h, N, d_k)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.W_o(out)

mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)
print(f"Output: {mha(x).shape}")  # (2, 10, 512)
print(f"Params: {sum(p.numel() for p in mha.parameters()):,}")`}),e.jsx(y,{type:"note",title:"PyTorch Built-in",children:e.jsxs("p",{children:["In practice, use ",e.jsx("code",{children:"torch.nn.MultiheadAttention"})," which implements fused multi-head attention with optimized kernels. The manual implementation above is for pedagogical clarity."]})})]})}const le=Object.freeze(Object.defineProperty({__proto__:null,default:V},Symbol.toStringTag,{value:"Module"}));function P(){const[a,d]=h.useState(0),n=["Le","chat","est","noir"],o=["The","cat","is"],s=[[.55,.15,.2,.1],[.1,.6,.15,.15],[.15,.1,.55,.2]][a];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Cross-Attention: Decoder attends to Encoder"}),e.jsx("div",{className:"flex gap-2 mb-4 mt-3",children:o.map((r,l)=>e.jsx("button",{onClick:()=>d(l),className:`px-3 py-1 rounded-lg text-sm font-medium transition ${a===l?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:r},l))}),e.jsxs("div",{className:"grid grid-cols-2 gap-6",children:[e.jsxs("div",{children:[e.jsx("p",{className:"text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2",children:"Decoder (Query)"}),e.jsx("div",{className:"flex gap-2",children:o.map((r,l)=>e.jsx("span",{className:`px-2 py-1 rounded text-sm font-mono ${l===a?"bg-violet-100 text-violet-700 dark:bg-violet-900 dark:text-violet-300 font-bold":"text-gray-500 dark:text-gray-400"}`,children:r},l))})]}),e.jsxs("div",{children:[e.jsx("p",{className:"text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2",children:"Encoder (Key/Value)"}),e.jsx("div",{className:"flex gap-2",children:n.map((r,l)=>e.jsxs("div",{className:"flex flex-col items-center gap-1",children:[e.jsx("div",{className:"w-12 rounded",style:{height:`${Math.max(4,s[l]*80)}px`,backgroundColor:`rgba(139, 92, 246, ${.2+s[l]*.8})`}}),e.jsx("span",{className:"text-xs font-mono text-gray-700 dark:text-gray-300",children:r}),e.jsx("span",{className:"text-xs text-violet-600 dark:text-violet-400",children:s[l].toFixed(2)})]},l))})]})]})]})}function F(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Cross-attention bridges two different sequences — typically the encoder output and the decoder input. The decoder generates queries from its own representation, while the keys and values come from the encoder, allowing each decoder position to attend to the full source sequence."}),e.jsxs(c,{title:"Cross-Attention",children:[e.jsxs("p",{children:["Given encoder output ",e.jsx(t.InlineMath,{math:"H^{enc}"})," and decoder hidden states ",e.jsx(t.InlineMath,{math:"H^{dec}"}),":"]}),e.jsx(t.BlockMath,{math:"Q = H^{dec} W^Q, \\quad K = H^{enc} W^K, \\quad V = H^{enc} W^V"}),e.jsx(t.BlockMath,{math:"\\text{CrossAttn} = \\text{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right) V"})]}),e.jsx(P,{}),e.jsx(g,{title:"Translation with Cross-Attention",children:e.jsx("p",{children:'When translating "Le chat est noir" to "The cat is black", cross-attention helps each decoder token align with its source counterpart. "The" primarily attends to "Le", "cat" attends to "chat", and so on — learning soft alignments without explicit alignment supervision.'})}),e.jsx(f,{title:"Cross-Attention Module",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, decoder_hidden, encoder_output):
        B, Nq, D = decoder_hidden.shape
        Nk = encoder_output.shape[1]
        h, dk = self.num_heads, self.d_k

        Q = self.W_q(decoder_hidden).view(B, Nq, h, dk).transpose(1, 2)
        K = self.W_k(encoder_output).view(B, Nk, h, dk).transpose(1, 2)
        V = self.W_v(encoder_output).view(B, Nk, h, dk).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (dk ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, Nq, D)
        return self.W_o(out)

# Encoder output: 10 source tokens; decoder: 6 target tokens
enc_out = torch.randn(2, 10, 256)
dec_hidden = torch.randn(2, 6, 256)
cross_attn = CrossAttention(d_model=256, num_heads=8)
output = cross_attn(dec_hidden, enc_out)
print(f"Output: {output.shape}")  # (2, 6, 256)`}),e.jsx(k,{title:"Encoder KV Caching",children:e.jsx("p",{children:"Since encoder outputs do not change during decoding, the K and V projections from the encoder can be computed once and cached. Recomputing them at each decoder step is a common source of unnecessary overhead in naive implementations."})}),e.jsx(y,{type:"note",title:"Beyond Seq2Seq",children:e.jsx("p",{children:"Cross-attention appears in many architectures beyond translation: vision transformers use it to combine image patches with text queries (CLIP), diffusion models use it to condition on text prompts, and retrieval-augmented models use it to attend over retrieved documents."})})]})}const de=Object.freeze(Object.defineProperty({__proto__:null,default:F},Symbol.toStringTag,{value:"Module"}));function K(){const[a,d]=h.useState(!0),n=40,o=[{label:"Input Embeddings + Positional Encoding",color:"#ddd6fe"},{label:"Multi-Head Self-Attention",color:"#c4b5fd"},{label:"Add & Layer Norm",color:"#a78bfa",isResidual:!0},{label:"Feed-Forward Network (FFN)",color:"#c4b5fd"},{label:"Add & Layer Norm",color:"#a78bfa",isResidual:!0},{label:"Encoder Output",color:"#ddd6fe"}];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Transformer Encoder Block"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:[e.jsx("input",{type:"checkbox",checked:a,onChange:i=>d(i.target.checked),className:"accent-violet-500"}),"Show residual connections"]}),e.jsxs("svg",{width:320,height:o.length*(n+16)+8,className:"mx-auto block",children:[o.map((i,s)=>{const r=s*(n+16)+4;return e.jsxs("g",{children:[e.jsx("rect",{x:40,y:r,width:240,height:n,rx:8,fill:i.color,stroke:"#7c3aed",strokeWidth:1}),e.jsx("text",{x:160,y:r+n/2+5,textAnchor:"middle",fontSize:11,fill:"#3b0764",children:i.label}),a&&i.isResidual&&e.jsx("path",{d:`M 35 ${r-n-12} C 15 ${r-n-12}, 15 ${r+n/2}, 35 ${r+n/2}`,fill:"none",stroke:"#8b5cf6",strokeWidth:1.5,strokeDasharray:"4,3",markerEnd:"url(#arrowhead)"})]},s)}),e.jsx("defs",{children:e.jsx("marker",{id:"arrowhead",markerWidth:"6",markerHeight:"4",refX:"6",refY:"2",orient:"auto",children:e.jsx("polygon",{points:"0 0, 6 2, 0 4",fill:"#8b5cf6"})})})]})]})}function W(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"The transformer encoder processes the entire input sequence in parallel through a stack of identical blocks. Each block contains multi-head self-attention and a position-wise feed-forward network, with residual connections and layer normalization stabilizing training."}),e.jsxs(c,{title:"Encoder Block",children:[e.jsx("p",{children:"Each encoder block applies two sub-layers with residual connections:"}),e.jsx(t.BlockMath,{math:"h = \\text{LayerNorm}(x + \\text{MultiHeadAttn}(x, x, x))"}),e.jsx(t.BlockMath,{math:"\\text{out} = \\text{LayerNorm}(h + \\text{FFN}(h))"}),e.jsxs("p",{className:"mt-2",children:["The FFN is a two-layer MLP: ",e.jsx(t.InlineMath,{math:"\\text{FFN}(x) = \\text{ReLU}(xW_1 + b_1)W_2 + b_2"})," with inner dimension typically ",e.jsx(t.InlineMath,{math:"4 \\cdot d_{\\text{model}}"}),"."]})]}),e.jsx(K,{}),e.jsxs(b,{title:"Why Residual Connections Matter",id:"residual-encoder",children:[e.jsx("p",{children:"Residual connections ensure the gradient flows directly through the network:"}),e.jsx(t.BlockMath,{math:"\\frac{\\partial \\mathcal{L}}{\\partial x_l} = \\frac{\\partial \\mathcal{L}}{\\partial x_L} \\prod_{i=l}^{L-1}\\left(1 + \\frac{\\partial F_i}{\\partial x_i}\\right)"}),e.jsx("p",{className:"mt-2",children:'The additive "1" term prevents gradient vanishing even in very deep stacks (BERT-large uses 24 encoder layers).'})]}),e.jsxs(g,{title:"Layer Normalization vs Batch Normalization",children:[e.jsxs("p",{children:["Transformers use ",e.jsx("strong",{children:"Layer Norm"})," (normalize across features for each token) rather than Batch Norm (normalize across the batch). Layer Norm is invariant to batch size and works naturally with variable-length sequences:"]}),e.jsx(t.BlockMath,{math:"\\text{LN}(x) = \\frac{x - \\mu}{\\sigma} \\cdot \\gamma + \\beta, \\quad \\mu = \\frac{1}{d}\\sum_i x_i"})]}),e.jsx(f,{title:"Transformer Encoder Block in PyTorch",code:`import torch
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.drop1(attn_out))
        ff_out = self.ffn(x)
        x = self.norm2(x + self.drop2(ff_out))
        return x

block = TransformerEncoderBlock(d_model=512, num_heads=8, d_ff=2048)
x = torch.randn(4, 20, 512)  # batch=4, seq=20
out = block(x)
print(f"Output: {out.shape}")  # (4, 20, 512)`}),e.jsx(y,{type:"note",title:"Pre-Norm vs Post-Norm",children:e.jsxs("p",{children:["The original Transformer uses ",e.jsx("strong",{children:"Post-Norm"})," (normalize after residual addition). Many modern models use ",e.jsx("strong",{children:"Pre-Norm"})," (normalize before the sub-layer), which is more stable for training deep models but may have slightly lower performance at convergence."]})})]})}const ce=Object.freeze(Object.defineProperty({__proto__:null,default:W},Symbol.toStringTag,{value:"Module"}));function z(){const[a,d]=h.useState(5);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Causal (Autoregressive) Mask"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Sequence length: ",a,e.jsx("input",{type:"range",min:3,max:7,step:1,value:a,onChange:n=>d(parseInt(n.target.value)),className:"w-28 accent-violet-500"})]}),e.jsx("div",{className:"flex justify-center",children:e.jsx("table",{className:"border-collapse",children:e.jsx("tbody",{children:Array.from({length:a},(n,o)=>e.jsx("tr",{children:Array.from({length:a},(i,s)=>e.jsx("td",{className:"w-10 h-10 text-center text-xs font-mono border border-gray-200 dark:border-gray-700",style:{backgroundColor:s<=o?"rgba(139, 92, 246, 0.5)":"rgba(220, 38, 38, 0.15)"},children:s<=o?"1":"0"},s))},o))})})}),e.jsxs("p",{className:"text-xs text-center mt-2 text-gray-500 dark:text-gray-400",children:[e.jsx("span",{className:"text-violet-600",children:"Violet = visible"}),", ",e.jsx("span",{className:"text-red-400",children:"Red = masked (future tokens)"})]})]})}function Q(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"The transformer decoder generates output tokens autoregressively — one at a time, left to right. It contains masked self-attention (preventing future token visibility), cross-attention to the encoder, and a feed-forward network, all with residual connections."}),e.jsxs(c,{title:"Decoder Block",children:[e.jsx("p",{children:"Each decoder block has three sub-layers:"}),e.jsx(t.BlockMath,{math:"h_1 = \\text{LN}(x + \\text{MaskedSelfAttn}(x))"}),e.jsx(t.BlockMath,{math:"h_2 = \\text{LN}(h_1 + \\text{CrossAttn}(h_1, H^{enc}))"}),e.jsx(t.BlockMath,{math:"\\text{out} = \\text{LN}(h_2 + \\text{FFN}(h_2))"})]}),e.jsxs(c,{title:"Masked Self-Attention",children:[e.jsxs("p",{children:["The causal mask sets future positions to ",e.jsx(t.InlineMath,{math:"-\\infty"})," before softmax:"]}),e.jsx(t.BlockMath,{math:"\\text{mask}_{ij} = \\begin{cases} 0 & \\text{if } j \\leq i \\\\ -\\infty & \\text{if } j > i \\end{cases}"}),e.jsxs("p",{className:"mt-2",children:["This ensures token ",e.jsx(t.InlineMath,{math:"i"})," can only attend to positions ",e.jsx(t.InlineMath,{math:"\\leq i"}),", preserving the autoregressive property."]})]}),e.jsx(z,{}),e.jsxs(g,{title:"Autoregressive Generation",children:[e.jsxs("p",{children:["At inference, the decoder generates one token at a time. To produce token ",e.jsx(t.InlineMath,{math:"t"}),", it conditions on all previous tokens ",e.jsx(t.InlineMath,{math:"y_{<t}"})," and the encoder output. The next-token probability is:"]}),e.jsx(t.BlockMath,{math:"P(y_t \\mid y_{<t}, X) = \\text{softmax}(h_t W_{\\text{vocab}})"})]}),e.jsx(f,{title:"Transformer Decoder Block",code:`import torch
import torch.nn as nn

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def causal_mask(seq_len, device):
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1
        )

    def forward(self, x, enc_output):
        mask = self.causal_mask(x.size(1), x.device)
        h, _ = self.masked_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + h)
        h, _ = self.cross_attn(x, enc_output, enc_output)
        x = self.norm2(x + h)
        x = self.norm3(x + self.ffn(x))
        return x

dec = TransformerDecoderBlock(d_model=512, num_heads=8, d_ff=2048)
tgt = torch.randn(2, 15, 512)
memory = torch.randn(2, 20, 512)
out = dec(tgt, memory)
print(f"Decoder output: {out.shape}")  # (2, 15, 512)`}),e.jsx(k,{title:"Training vs Inference Mismatch",children:e.jsxs("p",{children:["During training, all target positions are processed in parallel using teacher forcing (feeding ground truth tokens). At inference, tokens are generated one by one. This discrepancy can cause ",e.jsx("em",{children:"exposure bias"})," — the model never sees its own mistakes during training."]})}),e.jsx(y,{type:"note",title:"Decoder-Only Models",children:e.jsx("p",{children:"Models like GPT use only the decoder stack (no cross-attention or encoder). The entire input and output are treated as a single sequence with causal masking. This simplification has proven remarkably effective for language modeling and generation tasks."})})]})}const he=Object.freeze(Object.defineProperty({__proto__:null,default:Q},Symbol.toStringTag,{value:"Module"}));function C(){const[a,d]=h.useState(1),n=6,o=["<s>","The","cat","sat","on","the"];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"KV Cache During Autoregressive Generation"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Decoding step: ",a,e.jsx("input",{type:"range",min:1,max:n,step:1,value:a,onChange:i=>d(parseInt(i.target.value)),className:"w-32 accent-violet-500"})]}),e.jsx("div",{className:"flex gap-2 mb-2",children:o.slice(0,a).map((i,s)=>e.jsx("div",{className:`px-2 py-1 rounded text-sm font-mono ${s<a-1?"bg-violet-100 text-violet-700 dark:bg-violet-900/50 dark:text-violet-300":"bg-violet-500 text-white font-bold"}`,children:i},s))}),e.jsxs("div",{className:"text-sm text-gray-600 dark:text-gray-400",children:[e.jsxs("p",{children:["Cached K/V: ",e.jsx("strong",{className:"text-violet-600 dark:text-violet-400",children:a-1})," positions"]}),e.jsxs("p",{children:["New computation: only for token ",e.jsxs("strong",{className:"text-violet-600 dark:text-violet-400",children:['"',o[a-1],'"']})]}),e.jsxs("p",{className:"mt-1",children:["Without cache: ",a," Q/K/V computations. With cache: ",e.jsx("strong",{children:"1"})," new + ",a-1," cached."]})]})]})}function E(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Training and inference in transformers involve fundamentally different computational patterns. Training leverages parallelism with teacher forcing, while inference requires sequential generation with optimization techniques like KV caching and beam search."}),e.jsxs(c,{title:"Teacher Forcing",children:[e.jsxs("p",{children:["During training, the decoder receives the ground-truth target tokens as input rather than its own predictions. For a target sequence ",e.jsx(t.InlineMath,{math:"y = (y_1, \\ldots, y_T)"}),":"]}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = -\\sum_{t=1}^{T} \\log P(y_t \\mid y_1, \\ldots, y_{t-1}, X)"}),e.jsx("p",{className:"mt-2",children:"All positions are computed in parallel since we have the full target during training."})]}),e.jsxs(c,{title:"KV Caching",children:[e.jsxs("p",{children:["During autoregressive inference, previously computed key and value vectors are cached and reused. At step ",e.jsx(t.InlineMath,{math:"t"}),", only the new token's Q, K, V are computed:"]}),e.jsx(t.BlockMath,{math:"K_t = [K_{\\text{cache}}; k_t], \\quad V_t = [V_{\\text{cache}}; v_t]"}),e.jsxs("p",{className:"mt-2",children:["This reduces per-step complexity from ",e.jsx(t.InlineMath,{math:"O(t \\cdot d)"})," to ",e.jsx(t.InlineMath,{math:"O(d)"}),"for the projection, though the attention still requires ",e.jsx(t.InlineMath,{math:"O(t)"}),"."]})]}),e.jsx(C,{}),e.jsxs(g,{title:"Beam Search",children:[e.jsxs("p",{children:["Beam search maintains ",e.jsx(t.InlineMath,{math:"B"})," candidate sequences at each step, expanding each by the top-",e.jsx(t.InlineMath,{math:"k"})," tokens and keeping the ",e.jsx(t.InlineMath,{math:"B"})," best overall:"]}),e.jsx(t.BlockMath,{math:"\\text{score}(y) = \\frac{1}{|y|^\\alpha} \\sum_{t=1}^{|y|} \\log P(y_t \\mid y_{<t})"}),e.jsxs("p",{children:["The length penalty ",e.jsx(t.InlineMath,{math:"\\alpha"})," (typically 0.6-0.7) prevents the model from favoring short sequences."]})]}),e.jsx(f,{title:"KV Cache Implementation",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class CachedSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None):
        B, N, D = x.shape
        qkv = self.W_qkv(x).reshape(B, N, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)
        return self.W_o(out), (k, v)

# Simulate autoregressive generation with KV cache
attn = CachedSelfAttention(d_model=256, num_heads=8)
cache = None
for step in range(5):
    token = torch.randn(1, 1, 256)  # one new token
    out, cache = attn(token, kv_cache=cache)
    print(f"Step {step}: cache K shape = {cache[0].shape}")`}),e.jsx(k,{title:"KV Cache Memory",children:e.jsxs("p",{children:["KV cache grows linearly with sequence length and batch size. For a model with",e.jsx(t.InlineMath,{math:"L"})," layers, ",e.jsx(t.InlineMath,{math:"h"})," heads, and dimension ",e.jsx(t.InlineMath,{math:"d"}),": memory = ",e.jsx(t.InlineMath,{math:"2 \\times L \\times n \\times d \\times \\text{bytes}"})," per sequence. For long contexts, this can exceed GPU memory."]})}),e.jsx(y,{type:"note",title:"Speculative Decoding",children:e.jsx("p",{children:"Speculative decoding uses a small draft model to propose several tokens, then verifies them in parallel with the large model. Accepted tokens skip individual decoding steps, providing 2-3x speedup without changing the output distribution."})})]})}const me=Object.freeze(Object.defineProperty({__proto__:null,default:E},Symbol.toStringTag,{value:"Module"}));function O(){const[a,d]=h.useState(8),[n,o]=h.useState(20),i=360,s=180;function r(x,p,j){const u=x/Math.pow(1e4,2*Math.floor(p/2)/j);return p%2===0?Math.sin(u):Math.cos(u)}const l=["#8b5cf6","#a78bfa","#c4b5fd","#7c3aed","#6d28d9","#5b21b6","#ddd6fe","#4c1d95"],m=Array.from({length:n},(x,p)=>p);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Sinusoidal Positional Encoding"}),e.jsxs("div",{className:"flex gap-4 mb-3",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Dimensions: ",a,e.jsx("input",{type:"range",min:4,max:8,step:2,value:a,onChange:x=>d(parseInt(x.target.value)),className:"w-24 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Positions: ",n,e.jsx("input",{type:"range",min:10,max:40,step:5,value:n,onChange:x=>o(parseInt(x.target.value)),className:"w-24 accent-violet-500"})]})]}),e.jsxs("svg",{width:i,height:s,className:"mx-auto block",children:[e.jsx("line",{x1:30,y1:s/2,x2:i,y2:s/2,stroke:"#d1d5db",strokeWidth:.5}),Array.from({length:a},(x,p)=>{const j=m.map((u,_)=>{const v=30+u/(n-1)*(i-40),w=s/2-r(u,p,a)*(s/2-10);return`${_===0?"M":"L"}${v},${w}`}).join(" ");return e.jsx("path",{d:j,fill:"none",stroke:l[p%l.length],strokeWidth:1.5,opacity:.8},p)})]}),e.jsx("p",{className:"text-xs text-center mt-1 text-gray-500 dark:text-gray-400",children:"Each curve is one dimension of the encoding; different frequencies capture position at different scales."})]})}function H(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Since transformers process all positions in parallel without recurrence, they need an explicit mechanism to encode position information. Vaswani et al. (2017) introduced sinusoidal positional encodings, which are added to the input embeddings."}),e.jsxs(c,{title:"Sinusoidal Positional Encoding",children:[e.jsx(t.BlockMath,{math:"PE_{(pos, 2i)} = \\sin\\!\\left(\\frac{pos}{10000^{2i / d_{\\text{model}}}}\\right)"}),e.jsx(t.BlockMath,{math:"PE_{(pos, 2i+1)} = \\cos\\!\\left(\\frac{pos}{10000^{2i / d_{\\text{model}}}}\\right)"}),e.jsxs("p",{className:"mt-2",children:["Each dimension corresponds to a sinusoid with wavelength forming a geometric progression from ",e.jsx(t.InlineMath,{math:"2\\pi"})," to ",e.jsx(t.InlineMath,{math:"10000 \\cdot 2\\pi"}),"."]})]}),e.jsx(O,{}),e.jsxs(b,{title:"Relative Position as Linear Transformation",id:"pe-relative",children:[e.jsxs("p",{children:["For any fixed offset ",e.jsx(t.InlineMath,{math:"k"}),", there exists a linear transformation ",e.jsx(t.InlineMath,{math:"M_k"})," such that:"]}),e.jsx(t.BlockMath,{math:"PE_{pos+k} = M_k \\cdot PE_{pos}"}),e.jsx("p",{className:"mt-2",children:"This means the model can learn to attend to relative positions through linear operations on the sinusoidal encodings, a key property that enables length generalization."})]}),e.jsx(g,{title:"Encoding Properties",children:e.jsxs("p",{children:["The dot product ",e.jsx(t.InlineMath,{math:"PE_{pos} \\cdot PE_{pos+k}"})," depends only on the offset ",e.jsx(t.InlineMath,{math:"k"}),", not the absolute position. Nearby positions have higher similarity, and the similarity decreases smoothly with distance."]})}),e.jsx(f,{title:"Sinusoidal Positional Encoding",code:`import torch
import math

def sinusoidal_pe(max_len, d_model):
    """Generate sinusoidal positional encoding."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Visualize: nearby positions have similar encodings
pe = sinusoidal_pe(50, 128)
print(f"Shape: {pe.shape}")  # (50, 128)

# Similarity between positions
sim = torch.cosine_similarity(pe[10].unsqueeze(0), pe, dim=-1)
print(f"Sim(pos=10, pos=10): {sim[10]:.4f}")  # 1.0
print(f"Sim(pos=10, pos=11): {sim[11]:.4f}")  # ~0.98
print(f"Sim(pos=10, pos=40): {sim[40]:.4f}")  # ~0.5`}),e.jsx(y,{type:"note",title:"Fixed vs Learned",children:e.jsx("p",{children:"Sinusoidal encodings are fixed (no learnable parameters) and can theoretically generalize to longer sequences than seen during training. In practice, the original Transformer paper found no significant difference between sinusoidal and learned positional embeddings, but sinusoidal encodings use zero additional parameters."})})]})}const xe=Object.freeze(Object.defineProperty({__proto__:null,default:H},Symbol.toStringTag,{value:"Module"}));function D(){const[a,d]=h.useState(null),n=[{property:"Parameters",sinusoidal:"None (fixed)",learned:"max_len x d_model"},{property:"Length generalization",sinusoidal:"Theoretically yes",learned:"No — limited to training length"},{property:"Expressiveness",sinusoidal:"Fixed frequency patterns",learned:"Data-adaptive patterns"},{property:"Used in",sinusoidal:"Original Transformer",learned:"BERT, GPT-2, ViT"},{property:"Training cost",sinusoidal:"Zero",learned:"Marginal (small table)"}];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Sinusoidal vs Learned Positional Embeddings"}),e.jsxs("table",{className:"w-full text-sm",children:[e.jsx("thead",{children:e.jsxs("tr",{children:[e.jsx("th",{className:"text-left py-1 px-2 text-gray-500 dark:text-gray-400 font-medium",children:"Property"}),e.jsx("th",{className:"text-left py-1 px-2 text-violet-600 dark:text-violet-400 font-medium cursor-pointer",onClick:()=>d(o=>o==="sin"?null:"sin"),children:"Sinusoidal"}),e.jsx("th",{className:"text-left py-1 px-2 text-violet-600 dark:text-violet-400 font-medium cursor-pointer",onClick:()=>d(o=>o==="learn"?null:"learn"),children:"Learned"})]})}),e.jsx("tbody",{children:n.map((o,i)=>e.jsxs("tr",{className:"border-t border-gray-100 dark:border-gray-800",children:[e.jsx("td",{className:"py-1.5 px-2 text-gray-700 dark:text-gray-300 font-medium",children:o.property}),e.jsx("td",{className:`py-1.5 px-2 ${a==="sin"?"bg-violet-50 dark:bg-violet-900/20":""} text-gray-600 dark:text-gray-400`,children:o.sinusoidal}),e.jsx("td",{className:`py-1.5 px-2 ${a==="learn"?"bg-violet-50 dark:bg-violet-900/20":""} text-gray-600 dark:text-gray-400`,children:o.learned})]},i))})]}),e.jsx("p",{className:"text-xs mt-2 text-gray-400",children:"Click column headers to highlight."})]})}function R(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Learned positional embeddings replace the fixed sinusoidal functions with a trainable embedding table. Models like BERT, GPT-2, and ViT use this approach, allowing the model to discover optimal positional representations from data."}),e.jsxs(c,{title:"Learned Positional Embeddings",children:[e.jsxs("p",{children:["A learnable embedding table ",e.jsx(t.InlineMath,{math:"E_{pos} \\in \\mathbb{R}^{L_{\\max} \\times d}"})," is added to the token embeddings:"]}),e.jsx(t.BlockMath,{math:"h_i^{(0)} = E_{\\text{token}}(x_i) + E_{\\text{pos}}(i)"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"L_{\\max}"})," is the maximum sequence length (e.g., 512 for BERT, 1024 for GPT-2). Both embedding tables are learned end-to-end via backpropagation."]})]}),e.jsx(D,{}),e.jsxs(g,{title:"BERT Positional Embeddings",children:[e.jsxs("p",{children:["BERT supports sequences up to 512 tokens and learns a ",e.jsx(t.InlineMath,{math:"512 \\times 768"})," position embedding table — only 393K parameters out of 110M total (0.36%). BERT also adds ",e.jsx("strong",{children:"segment embeddings"})," to distinguish sentence A from sentence B:"]}),e.jsx(t.BlockMath,{math:"h_i = E_{\\text{token}}(x_i) + E_{\\text{pos}}(i) + E_{\\text{seg}}(s_i)"})]}),e.jsx(f,{title:"Learned Positional Embedding in PyTorch",code:`import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.d_model = d_model

    def forward(self, x):
        B, N = x.shape
        positions = torch.arange(N, device=x.device).unsqueeze(0)
        tok = self.token_emb(x) * (self.d_model ** 0.5)  # scale
        pos = self.pos_emb(positions)
        return tok + pos

emb = TokenEmbedding(vocab_size=30000, d_model=768, max_len=512)
tokens = torch.randint(0, 30000, (2, 128))
output = emb(tokens)
print(f"Embedding output: {output.shape}")  # (2, 128, 768)

# Visualize learned position similarity
with torch.no_grad():
    pe = emb.pos_emb.weight  # (512, 768)
    sim = torch.cosine_similarity(pe[0:1], pe, dim=-1)
    print(f"Position 0 vs 1: {sim[1]:.4f}")
    print(f"Position 0 vs 100: {sim[100]:.4f}")`}),e.jsx(k,{title:"Length Extrapolation Failure",children:e.jsxs("p",{children:["Learned positional embeddings cannot extrapolate beyond the maximum training length. If a model trained with ",e.jsx(t.InlineMath,{math:"L_{\\max} = 512"})," receives 600 tokens, positions 513-600 have no valid embedding. This limitation motivated the development of relative positional encoding methods like RoPE and ALiBi."]})}),e.jsx(y,{type:"note",title:"Vision Transformer (ViT) Positions",children:e.jsx("p",{children:"ViT treats image patches as tokens and uses learned 2D positional embeddings. Interestingly, the learned embeddings recover a grid structure resembling the spatial layout of patches, demonstrating that the model naturally discovers meaningful positional information."})})]})}const pe=Object.freeze(Object.defineProperty({__proto__:null,default:R},Symbol.toStringTag,{value:"Module"}));function G(){const[a,d]=h.useState(.25),n=6,o=44;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"ALiBi Attention Bias"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Slope (m): ",a.toFixed(2),e.jsx("input",{type:"range",min:.05,max:1,step:.05,value:a,onChange:i=>d(parseFloat(i.target.value)),className:"w-32 accent-violet-500"})]}),e.jsx("div",{className:"flex justify-center overflow-x-auto",children:e.jsx("table",{className:"border-collapse",children:e.jsx("tbody",{children:Array.from({length:n},(i,s)=>e.jsx("tr",{children:Array.from({length:n},(r,l)=>{const m=l<=s?-a*(s-l):null;return e.jsx("td",{className:"text-center text-xs font-mono border border-gray-200 dark:border-gray-700",style:{width:o,height:o,backgroundColor:m!==null?`rgba(139, 92, 246, ${Math.max(.1,1+m/3)})`:"rgba(220, 38, 38, 0.1)"},children:m!==null?m.toFixed(1):"-inf"},l)})},s))})})}),e.jsx("p",{className:"text-xs text-center mt-2 text-gray-500 dark:text-gray-400",children:"Bias = -m * |i - j|. More distant positions receive larger negative bias."})]})}function $(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Rotary Position Embeddings (RoPE) and Attention with Linear Biases (ALiBi) are modern positional encoding methods designed to enable better length generalization while encoding relative positional information directly into the attention computation."}),e.jsxs(c,{title:"Rotary Position Embeddings (RoPE)",children:[e.jsx("p",{children:"RoPE applies a rotation matrix to query and key vectors based on position:"}),e.jsx(t.BlockMath,{math:"f(x_m, m) = R_{\\Theta,m} x_m = \\begin{pmatrix} x_1 \\cos m\\theta_1 - x_2 \\sin m\\theta_1 \\\\ x_1 \\sin m\\theta_1 + x_2 \\cos m\\theta_1 \\\\ \\vdots \\end{pmatrix}"}),e.jsxs("p",{className:"mt-2",children:["The dot product ",e.jsx(t.InlineMath,{math:"f(q, m)^\\top f(k, n)"})," depends only on the relative position ",e.jsx(t.InlineMath,{math:"m - n"})," and the token content, elegantly combining both."]})]}),e.jsxs(b,{title:"RoPE Relative Position Property",id:"rope-relative",children:[e.jsx(t.BlockMath,{math:"\\langle f(q, m), f(k, n) \\rangle = \\langle R_{\\Theta, n-m} q, k \\rangle = g(q, k, m - n)"}),e.jsxs("p",{className:"mt-2",children:["The attention score between positions ",e.jsx(t.InlineMath,{math:"m"})," and ",e.jsx(t.InlineMath,{math:"n"})," depends only on relative distance ",e.jsx(t.InlineMath,{math:"m - n"}),", not absolute positions. This is achieved without any additive encoding — the position is baked into the rotation."]})]}),e.jsxs(c,{title:"ALiBi (Attention with Linear Biases)",children:[e.jsx("p",{children:"ALiBi adds a static, non-learned bias to attention scores proportional to distance:"}),e.jsx(t.BlockMath,{math:"\\text{softmax}\\!\\left(\\frac{q_i^\\top k_j}{\\sqrt{d_k}} - m \\cdot |i - j|\\right)"}),e.jsxs("p",{className:"mt-2",children:["Each head uses a different slope ",e.jsx(t.InlineMath,{math:"m"}),", set as a geometric sequence. No positional embeddings are added to the input — position is encoded purely via attention biases."]})]}),e.jsx(G,{}),e.jsx(g,{title:"Length Generalization",children:e.jsx("p",{children:"ALiBi trained on 1024 tokens can generalize to 2048+ tokens at inference time. RoPE achieves similar extrapolation, especially with techniques like NTK-aware scaling that adjust the base frequency. Both significantly outperform learned positional embeddings at unseen lengths."})}),e.jsx(f,{title:"RoPE Implementation",code:`import torch

def precompute_freqs(dim, max_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len)
    freqs = torch.outer(t, freqs)  # (max_len, dim/2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex

def apply_rope(x, freqs):
    # x: (B, H, N, D) -> view as complex pairs
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    x_rotated = x_complex * freqs
    return torch.view_as_real(x_rotated).reshape_as(x)

# Example
freqs = precompute_freqs(dim=64, max_len=2048)
q = torch.randn(2, 8, 128, 64)  # (B, heads, seq, d_k)
q_rope = apply_rope(q, freqs)
print(f"RoPE output: {q_rope.shape}")  # same as input`}),e.jsx(y,{type:"note",title:"Which to Choose?",children:e.jsxs("p",{children:[e.jsx("strong",{children:"RoPE"})," is used in LLaMA, PaLM, and most modern LLMs — it preserves the full dot-product structure and generalizes well. ",e.jsx("strong",{children:"ALiBi"})," is simpler and has zero learnable parameters but provides a softer form of relative positioning. Both vastly outperform absolute positional embeddings for long-context tasks."]})})]})}const ue=Object.freeze(Object.defineProperty({__proto__:null,default:$},Symbol.toStringTag,{value:"Module"}));function U(){const[a,d]=h.useState(4),n=200,o=n/a;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Tiled Attention Computation"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Tiles per dimension: ",a,e.jsx("input",{type:"range",min:2,max:8,step:1,value:a,onChange:i=>d(parseInt(i.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("div",{className:"flex justify-center gap-8 items-start",children:[e.jsxs("div",{children:[e.jsx("p",{className:"text-xs text-gray-500 dark:text-gray-400 mb-1 text-center",children:"Standard: full N x N in HBM"}),e.jsxs("svg",{width:n,height:n,children:[e.jsx("rect",{x:0,y:0,width:n,height:n,fill:"rgba(220, 38, 38, 0.2)",stroke:"#dc2626",strokeWidth:1,rx:4}),e.jsx("text",{x:n/2,y:n/2+4,textAnchor:"middle",fontSize:11,fill:"#dc2626",children:"O(N^2) memory"})]})]}),e.jsxs("div",{children:[e.jsx("p",{className:"text-xs text-gray-500 dark:text-gray-400 mb-1 text-center",children:"Flash: tiles in SRAM"}),e.jsxs("svg",{width:n,height:n,children:[Array.from({length:a*a},(i,s)=>{const r=Math.floor(s/a),l=s%a;return e.jsx("rect",{x:l*o+1,y:r*o+1,width:o-2,height:o-2,rx:2,fill:`rgba(139, 92, 246, ${.15+s%3*.15})`,stroke:"#8b5cf6",strokeWidth:.5},s)}),e.jsx("text",{x:n/2,y:n/2+4,textAnchor:"middle",fontSize:11,fill:"#7c3aed",children:"O(N) memory"})]})]})]})]})}function X(){return e.jsxs("div",{className:"space-y-6",children:[e.jsxs("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:["Flash Attention is an IO-aware algorithm that computes exact attention without materializing the full ",e.jsx(t.InlineMath,{math:"N \\times N"})," attention matrix in GPU high-bandwidth memory (HBM). By tiling the computation to fit in fast SRAM, it achieves significant speedups and memory savings."]}),e.jsxs(c,{title:"The Memory Bottleneck",children:[e.jsx("p",{children:"Standard attention requires storing the full attention matrix:"}),e.jsx(t.BlockMath,{math:"\\underbrace{S = QK^\\top}_{N \\times N} \\rightarrow \\underbrace{P = \\text{softmax}(S)}_{N \\times N} \\rightarrow \\underbrace{O = PV}_{N \\times d}"}),e.jsxs("p",{className:"mt-2",children:["The ",e.jsx(t.InlineMath,{math:"N \\times N"})," matrices ",e.jsx(t.InlineMath,{math:"S"})," and ",e.jsx(t.InlineMath,{math:"P"})," dominate memory for long sequences. Flash Attention never materializes them in full."]})]}),e.jsx(U,{}),e.jsxs(b,{title:"Flash Attention IO Complexity",id:"flash-io",children:[e.jsx("p",{children:"Flash Attention requires:"}),e.jsx(t.BlockMath,{math:"O\\!\\left(\\frac{N^2 d^2}{M}\\right) \\text{ HBM accesses}"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"M"})," is SRAM size. Standard attention requires ",e.jsx(t.InlineMath,{math:"O(N^2 + Nd)"})," HBM reads/writes. For typical SRAM sizes, Flash Attention is 2-4x faster despite doing the same FLOPs, because it is memory-access efficient."]})]}),e.jsxs(g,{title:"Tiling and Online Softmax",children:[e.jsx("p",{children:'The key insight: softmax can be computed incrementally using the "online softmax trick". For each tile, compute local softmax with a running maximum and accumulator, then rescale when processing the next tile. This avoids needing all scores simultaneously.'}),e.jsx(t.BlockMath,{math:"m_{\\text{new}} = \\max(m_{\\text{old}}, \\max(S_{\\text{tile}})), \\quad \\ell_{\\text{new}} = e^{m_{\\text{old}} - m_{\\text{new}}} \\ell_{\\text{old}} + \\sum e^{S_{\\text{tile}} - m_{\\text{new}}}"})]}),e.jsx(f,{title:"Using Flash Attention in PyTorch",code:`import torch
import torch.nn.functional as F

# PyTorch 2.0+ includes Flash Attention via SDPA
# (Scaled Dot Product Attention)
Q = torch.randn(2, 8, 4096, 64, device='cuda', dtype=torch.float16)
K = torch.randn(2, 8, 4096, 64, device='cuda', dtype=torch.float16)
V = torch.randn(2, 8, 4096, 64, device='cuda', dtype=torch.float16)

# This automatically uses Flash Attention when possible
with torch.backends.cuda.sdp_kernel(
    enable_flash=True, enable_math=False, enable_mem_efficient=False
):
    output = F.scaled_dot_product_attention(Q, K, V)
    print(f"Output: {output.shape}")  # (2, 8, 4096, 64)

# Memory comparison (conceptual):
# Standard: 4096^2 * 2 bytes * 8 heads = 512 MB for attn matrix
# Flash: O(N) memory = only a few MB for tiles`}),e.jsx(k,{title:"Hardware Requirements",children:e.jsxs("p",{children:["Flash Attention requires GPU hardware with sufficient SRAM (modern NVIDIA GPUs like A100/H100). It is an ",e.jsx("em",{children:"exact"})," computation — not an approximation — but the implementation is hardware-specific. Always benchmark on your target hardware."]})}),e.jsx(y,{type:"note",title:"Flash Attention v2 and Beyond",children:e.jsx("p",{children:"Flash Attention v2 further optimizes work partitioning across GPU thread blocks and warps, achieving near-optimal occupancy. It supports causal masking natively and is now the default attention backend in most deep learning frameworks."})})]})}const ge=Object.freeze(Object.defineProperty({__proto__:null,default:X},Symbol.toStringTag,{value:"Module"}));function Z(){const[a,d]=h.useState("full"),n=8;function o(s,r){return a==="full"?!0:a==="local"?Math.abs(s-r)<=1:a==="strided"?Math.abs(s-r)<=1||r%3===0:a==="bigbird"?Math.abs(s-r)<=1||r===0||s===0||Math.random()<.15:!0}const i=Array.from({length:n},(s,r)=>Array.from({length:n},(l,m)=>o(r,m)));return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Sparse Attention Patterns"}),e.jsx("div",{className:"flex gap-2 mb-4 mt-2 flex-wrap",children:["full","local","strided","bigbird"].map(s=>e.jsx("button",{onClick:()=>d(s),className:`px-3 py-1 rounded-lg text-sm font-medium transition capitalize ${a===s?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:s==="bigbird"?"BigBird":s},s))}),e.jsx("div",{className:"flex justify-center",children:e.jsx("div",{className:"grid",style:{gridTemplateColumns:`repeat(${n}, 28px)`,gap:"2px"},children:i.flat().map((s,r)=>e.jsx("div",{className:"w-7 h-7 rounded-sm",style:{backgroundColor:s?"rgba(139, 92, 246, 0.6)":"rgba(156, 163, 175, 0.15)"}},r))})}),e.jsxs("p",{className:"text-xs text-center mt-2 text-gray-500 dark:text-gray-400",children:["Violet cells = computed attention, gray = skipped. Pattern: ",a,"."]})]})}function Y(){return e.jsxs("div",{className:"space-y-6",children:[e.jsxs("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:["Standard attention has ",e.jsx(t.InlineMath,{math:"O(n^2)"})," complexity, making it prohibitive for long sequences. Linear attention and sparse attention methods reduce this cost while preserving most of the model's expressiveness."]}),e.jsxs(c,{title:"Linear Attention",children:[e.jsxs("p",{children:["By applying kernel feature maps ",e.jsx(t.InlineMath,{math:"\\phi"})," to Q and K, attention can be rewritten to avoid the ",e.jsx(t.InlineMath,{math:"N \\times N"})," matrix:"]}),e.jsx(t.BlockMath,{math:"\\text{Attn}(Q, K, V)_i = \\frac{\\phi(q_i)^\\top \\sum_j \\phi(k_j) v_j^\\top}{\\phi(q_i)^\\top \\sum_j \\phi(k_j)}"}),e.jsxs("p",{className:"mt-2",children:["The sum ",e.jsx(t.InlineMath,{math:"\\sum_j \\phi(k_j) v_j^\\top"})," is computed once in ",e.jsx(t.InlineMath,{math:"O(nd^2)"}),", then each query uses it in ",e.jsx(t.InlineMath,{math:"O(d^2)"})," — total ",e.jsx(t.InlineMath,{math:"O(nd^2)"})," instead of ",e.jsx(t.InlineMath,{math:"O(n^2 d)"}),"."]})]}),e.jsx(Z,{}),e.jsxs(c,{title:"Longformer: Local + Global Attention",children:[e.jsx("p",{children:"Longformer combines a local sliding window with global tokens:"}),e.jsx(t.BlockMath,{math:"\\text{Attn}_i = \\text{LocalWindow}(i, w) \\cup \\text{GlobalTokens}"}),e.jsxs("p",{className:"mt-2",children:["Window size ",e.jsx(t.InlineMath,{math:"w"})," gives ",e.jsx(t.InlineMath,{math:"O(nw)"})," complexity. Selected global tokens (e.g., [CLS]) attend to all positions, providing a bridge across the full sequence."]})]}),e.jsxs(b,{title:"Complexity Comparison",id:"sparse-complexity",children:[e.jsx(t.BlockMath,{math:"\\begin{aligned} &\\text{Full attention: } O(n^2 d) \\\\ &\\text{Linear attention: } O(n d^2) \\\\ &\\text{Longformer: } O(nw d) \\quad (w \\ll n) \\\\ &\\text{BigBird: } O(n(w + r + g) d) \\end{aligned}"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"w"})," = window size, ",e.jsx(t.InlineMath,{math:"r"})," = random connections,",e.jsx(t.InlineMath,{math:"g"})," = global tokens. All are linear in ",e.jsx(t.InlineMath,{math:"n"}),"."]})]}),e.jsx(g,{title:"BigBird: Sparse is Enough",children:e.jsxs("p",{children:["BigBird combines three patterns: ",e.jsx("strong",{children:"local window"})," (nearby tokens),",e.jsx("strong",{children:"random"})," (random connections), and ",e.jsx("strong",{children:"global"})," (special tokens attending everywhere). This achieves Turing-completeness — meaning the sparse pattern is theoretically as expressive as full attention."]})}),e.jsx(f,{title:"Linear Attention with Feature Maps",code:`import torch
import torch.nn as nn

def elu_feature_map(x):
    """ELU-based feature map for linear attention (positive)."""
    return torch.nn.functional.elu(x) + 1

def linear_attention(Q, K, V):
    """O(N*d^2) attention using kernel trick."""
    Q = elu_feature_map(Q)  # (B, N, d)
    K = elu_feature_map(K)  # (B, N, d)

    # Compute KV aggregate: O(N*d^2)
    KV = torch.bmm(K.transpose(1, 2), V)  # (B, d, d)

    # Compute normalizer
    Z = 1.0 / (torch.bmm(Q, K.sum(dim=1, keepdim=True).transpose(1, 2)) + 1e-6)

    # Final output: O(N*d^2)
    out = torch.bmm(Q, KV) * Z
    return out

B, N, d = 2, 8192, 64
Q = torch.randn(B, N, d)
K = torch.randn(B, N, d)
V = torch.randn(B, N, d)
output = linear_attention(Q, K, V)
print(f"Output: {output.shape}")  # (2, 8192, 64)
# Note: full attention on N=8192 would need 8192^2 = 67M entries!`}),e.jsx(y,{type:"note",title:"Trade-offs",children:e.jsx("p",{children:"Linear and sparse attention methods trade expressiveness for efficiency. For most NLP tasks with moderate sequence lengths (under 4K tokens), Flash Attention with full attention often outperforms these approximations. Sparse methods shine for very long documents (8K-128K tokens) where quadratic attention is truly infeasible."})})]})}const fe=Object.freeze(Object.defineProperty({__proto__:null,default:Y},Symbol.toStringTag,{value:"Module"}));function J(){const[a,d]=h.useState("gqa"),n=8,o={mha:{kvHeads:8,label:"Multi-Head (MHA)",ratio:"1:1"},gqa:{kvHeads:2,label:"Grouped-Query (GQA)",ratio:"4:1"},mqa:{kvHeads:1,label:"Multi-Query (MQA)",ratio:"8:1"}},i=o[a],s=n/i.kvHeads;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"KV Head Sharing"}),e.jsx("div",{className:"flex gap-2 mb-4 mt-2",children:Object.keys(o).map(r=>e.jsx("button",{onClick:()=>d(r),className:`px-3 py-1 rounded-lg text-sm font-medium transition ${a===r?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:o[r].label},r))}),e.jsxs("div",{className:"space-y-3",children:[e.jsxs("div",{children:[e.jsxs("p",{className:"text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1",children:["Query Heads (",n,")"]}),e.jsx("div",{className:"flex gap-1",children:Array.from({length:n},(r,l)=>e.jsxs("div",{className:"w-9 h-8 rounded text-xs flex items-center justify-center font-mono",style:{backgroundColor:`rgba(139, 92, 246, ${.3+l%s*.15})`,border:"1px solid #8b5cf6"},children:["Q",l]},l))})]}),e.jsxs("div",{className:"text-center text-gray-400 text-xs",children:["shares KV with ratio ",i.ratio]}),e.jsxs("div",{children:[e.jsxs("p",{className:"text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1",children:["KV Heads (",i.kvHeads,")"]}),e.jsx("div",{className:"flex gap-1",children:Array.from({length:i.kvHeads},(r,l)=>e.jsxs("div",{className:"h-8 rounded text-xs flex items-center justify-center font-mono bg-violet-200 dark:bg-violet-800 text-violet-800 dark:text-violet-200 border border-violet-400",style:{width:`${n/i.kvHeads*40-4}px`},children:["KV",l]},l))})]})]}),e.jsxs("p",{className:"text-sm mt-3 text-gray-600 dark:text-gray-400",children:["KV cache memory: ",e.jsxs("strong",{className:"text-violet-600 dark:text-violet-400",children:[(i.kvHeads/n*100).toFixed(0),"%"]})," of full MHA"]})]})}function ee(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Grouped-Query Attention (GQA) and Multi-Query Attention (MQA) reduce the KV cache memory footprint by sharing key-value heads across multiple query heads. This is critical for serving large language models efficiently during inference."}),e.jsxs(c,{title:"Multi-Query Attention (MQA)",children:[e.jsx("p",{children:"All query heads share a single set of keys and values:"}),e.jsx(t.BlockMath,{math:"Q_i = XW_i^Q \\quad (h \\text{ different}), \\quad K = XW^K, \\quad V = XW^V \\quad (\\text{shared})"}),e.jsxs("p",{className:"mt-2",children:["KV cache is reduced by factor ",e.jsx(t.InlineMath,{math:"h"})," (number of heads), but quality can degrade due to reduced capacity."]})]}),e.jsxs(c,{title:"Grouped-Query Attention (GQA)",children:[e.jsxs("p",{children:["Query heads are divided into ",e.jsx(t.InlineMath,{math:"g"})," groups, each sharing one KV head:"]}),e.jsx(t.BlockMath,{math:"\\text{head}_i = \\text{Attn}(Q_i, K_{\\lfloor i/G \\rfloor}, V_{\\lfloor i/G \\rfloor})"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"G = h / g"})," is the group size. GQA interpolates between MHA (",e.jsx(t.InlineMath,{math:"g = h"}),") and MQA (",e.jsx(t.InlineMath,{math:"g = 1"}),")."]})]}),e.jsx(J,{}),e.jsxs(g,{title:"KV Cache Memory Savings",children:[e.jsx("p",{children:"For LLaMA-2 70B with 64 query heads, 8 KV heads, 128 dim/head, 80 layers:"}),e.jsx(t.BlockMath,{math:"\\text{KV cache per token} = 2 \\times 80 \\times 8 \\times 128 \\times 2 \\text{ bytes} = 327 \\text{ KB}"}),e.jsxs("p",{children:["Compared to full MHA: ",e.jsx(t.InlineMath,{math:"2 \\times 80 \\times 64 \\times 128 \\times 2 = 2.6"})," MB per token. That is an ",e.jsx("strong",{children:"8x reduction"}),", enabling much larger batch sizes during serving."]})]}),e.jsx(f,{title:"Grouped-Query Attention Implementation",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_heads):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_q_heads // num_kv_heads
        self.d_k = d_model // num_q_heads

        self.W_q = nn.Linear(d_model, num_q_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, N, _ = x.shape
        Q = self.W_q(x).view(B, N, self.num_q_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_kv_heads, self.d_k).transpose(1, 2)

        # Repeat KV heads to match query heads
        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, N, -1)
        return self.W_o(out)

# LLaMA-2 style: 32 query heads, 8 KV heads
gqa = GroupedQueryAttention(d_model=4096, num_q_heads=32, num_kv_heads=8)
x = torch.randn(1, 128, 4096)
print(f"Output: {gqa(x).shape}")  # (1, 128, 4096)
print(f"KV params: {sum(p.numel() for p in [gqa.W_k, gqa.W_v]):,}")
print(f"Q params:  {gqa.W_q.weight.numel():,}")`}),e.jsx(k,{title:"Converting MHA to GQA",children:e.jsx("p",{children:'Existing MHA models can be converted to GQA by mean-pooling the KV heads within each group, then fine-tuning briefly. This "uptrained" GQA model recovers most of the original quality while getting the inference memory benefits. Simply dropping heads without fine-tuning degrades performance significantly.'})}),e.jsx(y,{type:"note",title:"Industry Adoption",children:e.jsx("p",{children:"GQA is now standard in production LLMs: LLaMA-2/3 (8 KV heads), Mistral (8 KV heads), and Gemma all use it. The quality-efficiency trade-off has proven favorable at scale, with GQA matching MHA quality while enabling 4-8x larger batch sizes during inference."})})]})}const ye=Object.freeze(Object.defineProperty({__proto__:null,default:ee},Symbol.toStringTag,{value:"Module"}));export{re as a,ie as b,oe as c,le as d,de as e,ce as f,he as g,me as h,xe as i,pe as j,ue as k,ge as l,fe as m,ye as n,se as s};

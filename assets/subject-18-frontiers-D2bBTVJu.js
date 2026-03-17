import{j as e,r as c}from"./vendor-DpISuAX6.js";import{r as t}from"./vendor-katex-CbWCYdth.js";import{D as h,T as _,P as p,E as x,N as g,W as v}from"./subject-01-foundations-D0A1VJsr.js";function N(){const[i,o]=c.useState("recurrent"),n={recurrent:{name:"Recurrent Mode (Inference)",complexity:"O(L)",desc:"Process one token at a time with hidden state. Sequential but constant memory.",formula:"h_t = \\bar{A} h_{t-1} + \\bar{B} x_t, \\quad y_t = C h_t"},convolutional:{name:"Convolutional Mode (Training)",complexity:"O(L log L)",desc:"Compute all outputs in parallel using a global convolution kernel via FFT.",formula:"y = \\bar{K} * x, \\quad \\bar{K} = (C\\bar{B}, C\\bar{A}\\bar{B}, C\\bar{A}^2\\bar{B}, \\ldots)"}},s=n[i];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"S4 Dual Computation Modes"}),e.jsx("div",{className:"flex gap-2 mb-3",children:Object.entries(n).map(([a,r])=>e.jsx("button",{onClick:()=>o(a),className:`px-3 py-1 rounded-lg text-sm transition ${i===a?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:r.name},a))}),e.jsxs("div",{className:"p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20 text-sm space-y-2",children:[e.jsx("p",{className:"text-gray-600 dark:text-gray-400",children:s.desc}),e.jsx(t.BlockMath,{math:s.formula}),e.jsxs("p",{className:"text-xs text-gray-500",children:["Complexity: ",s.complexity]})]})]})}function M(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"S4 (Structured State Spaces for Sequence Modeling) provides an alternative to attention for long-range sequence modeling. By parameterizing a continuous-time linear system and discretizing it, S4 achieves efficient training via convolution and efficient inference via recurrence."}),e.jsxs(h,{title:"Continuous-Time State Space Model",children:[e.jsxs("p",{children:["A linear state space model maps input ",e.jsx(t.InlineMath,{math:"x(t)"})," to output ",e.jsx(t.InlineMath,{math:"y(t)"})," through a hidden state ",e.jsx(t.InlineMath,{math:"h(t)"}),":"]}),e.jsx(t.BlockMath,{math:"h'(t) = Ah(t) + Bx(t), \\quad y(t) = Ch(t) + Dx(t)"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"A \\in \\mathbb{R}^{N \\times N}"}),", ",e.jsx(t.InlineMath,{math:"B \\in \\mathbb{R}^{N \\times 1}"}),", ",e.jsx(t.InlineMath,{math:"C \\in \\mathbb{R}^{1 \\times N}"}),". After discretization with step size ",e.jsx(t.InlineMath,{math:"\\Delta"}),":"]}),e.jsx(t.BlockMath,{math:"\\bar{A} = \\exp(\\Delta A), \\quad \\bar{B} = (\\Delta A)^{-1}(\\exp(\\Delta A) - I) \\cdot \\Delta B"})]}),e.jsx(N,{}),e.jsxs(_,{title:"HiPPO Initialization for Long-Range Memory",id:"hippo",children:[e.jsxs("p",{children:["The key to S4's long-range capability is the HiPPO (High-order Polynomial Projection Operators) initialization of matrix ",e.jsx(t.InlineMath,{math:"A"}),":"]}),e.jsx(t.BlockMath,{math:"A_{nk} = -\\begin{cases} (2n+1)^{1/2}(2k+1)^{1/2} & \\text{if } n > k \\\\ n+1 & \\text{if } n = k \\\\ 0 & \\text{if } n < k \\end{cases}"}),e.jsxs("p",{className:"mt-2",children:["This initialization ensures the state ",e.jsx(t.InlineMath,{math:"h(t)"})," optimally approximates the history of the input signal using Legendre polynomials, enabling memory over thousands of timesteps."]})]}),e.jsx(p,{title:"Simplified S4 Convolution Kernel",code:`import torch
import torch.nn.functional as F

def s4_kernel(A, B, C, L, dt=1.0):
    """Compute the S4 convolution kernel of length L.

    Args:
        A: [N, N] state matrix
        B: [N, 1] input matrix
        C: [1, N] output matrix
        L: sequence length
        dt: discretization step size
    """
    N = A.shape[0]
    # Discretize (simplified zero-order hold)
    A_bar = torch.matrix_exp(A * dt)
    B_bar = torch.linalg.solve(A, (A_bar - torch.eye(N)) @ B)

    # Build kernel: K[i] = C @ A_bar^i @ B_bar
    kernel = torch.zeros(L)
    A_power = torch.eye(N)
    for i in range(L):
        kernel[i] = (C @ A_power @ B_bar).squeeze()
        A_power = A_power @ A_bar

    return kernel

def s4_convolve(x, kernel):
    """Apply S4 kernel as a global convolution using FFT."""
    L = x.shape[-1]
    # Pad for causal convolution
    K = F.pad(kernel, (0, L))
    X = F.pad(x, (0, kernel.shape[-1]))
    return torch.fft.irfft(torch.fft.rfft(X) * torch.fft.rfft(K))[:L]

# Example: 64-dim state, 1024-length sequence
N, L = 64, 1024
A = -torch.eye(N) + 0.1 * torch.randn(N, N)  # Stable A
B, C = torch.randn(N, 1), torch.randn(1, N)
kernel = s4_kernel(A, B, C, L)
print(f"Kernel shape: {kernel.shape}, decays: {kernel[:3]} ... {kernel[-3:]}")`}),e.jsxs(x,{title:"S4 on Long Range Arena",children:[e.jsx("p",{children:"S4 achieved state-of-the-art on the Long Range Arena benchmark (sequences of 1K-16K tokens):"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsx("li",{children:"Path-X (16K tokens): 96.4% — first model above random chance on this task"}),e.jsx("li",{children:"Overall LRA average: 86.1% vs Transformer's 59.3%"}),e.jsxs("li",{children:["Key advantage: ",e.jsx(t.InlineMath,{math:"O(L \\log L)"})," training vs ",e.jsx(t.InlineMath,{math:"O(L^2)"})," for attention"]})]})]}),e.jsx(g,{type:"note",title:"From S4 to Modern SSMs",children:e.jsx("p",{children:"S4 spawned a family of models: S4D (diagonal approximation), S5 (parallel scan), H3 (combining SSM with attention), and ultimately Mamba. Each simplification improved speed while maintaining the core benefit of efficient long-range modeling."})})]})}const ie=Object.freeze(Object.defineProperty({__proto__:null,default:M},Symbol.toStringTag,{value:"Module"}));function S(){const[i,o]=c.useState("relevant"),n={relevant:{label:"Relevant Token",delta:.8,bScale:1,desc:"Large delta -> retain in state; large B -> strong input projection"},irrelevant:{label:"Irrelevant Token",delta:.05,bScale:.1,desc:"Small delta -> skip/forget; small B -> weak input projection"},reset:{label:"Reset Token",delta:2,bScale:0,desc:"Very large delta -> clear history; zero B -> no new information stored"}},s=n[i],a=Math.exp(-s.delta);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Mamba Selective State Space"}),e.jsx("div",{className:"flex gap-2 mb-3 flex-wrap",children:Object.entries(n).map(([r,l])=>e.jsx("button",{onClick:()=>o(r),className:`px-3 py-1 rounded-lg text-sm transition ${i===r?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:l.label},r))}),e.jsxs("div",{className:"grid grid-cols-3 gap-3 text-sm text-center mb-2",children:[e.jsxs("div",{className:"p-2 rounded bg-violet-50 dark:bg-violet-900/20",children:[e.jsx("p",{className:"text-violet-700 dark:text-violet-300 font-medium",children:"Delta"}),e.jsx("p",{className:"font-bold",children:s.delta.toFixed(2)})]}),e.jsxs("div",{className:"p-2 rounded bg-violet-50 dark:bg-violet-900/20",children:[e.jsx("p",{className:"text-violet-700 dark:text-violet-300 font-medium",children:"State Decay"}),e.jsx("p",{className:"font-bold",children:a.toFixed(3)})]}),e.jsxs("div",{className:"p-2 rounded bg-violet-50 dark:bg-violet-900/20",children:[e.jsx("p",{className:"text-violet-700 dark:text-violet-300 font-medium",children:"Input Scale"}),e.jsx("p",{className:"font-bold",children:s.bScale.toFixed(2)})]})]}),e.jsx("p",{className:"text-xs text-gray-500 text-center",children:s.desc})]})}function L(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Mamba introduces input-dependent selection into SSMs, making the state transition parameters vary with the input. This gives SSMs the ability to selectively remember or forget information — a capability previously unique to attention mechanisms."}),e.jsxs(h,{title:"Selective State Space (Mamba)",children:[e.jsxs("p",{children:["Unlike S4 where ",e.jsx(t.InlineMath,{math:"A, B, C, \\Delta"})," are fixed, Mamba makes ",e.jsx(t.InlineMath,{math:"B, C, \\Delta"})," functions of the input:"]}),e.jsx(t.BlockMath,{math:"B_t = \\text{Linear}(x_t), \\quad C_t = \\text{Linear}(x_t), \\quad \\Delta_t = \\text{softplus}(\\text{Linear}(x_t))"}),e.jsx("p",{className:"mt-2",children:"The discretized recurrence becomes input-dependent:"}),e.jsx(t.BlockMath,{math:"h_t = \\bar{A}_t h_{t-1} + \\bar{B}_t x_t, \\quad y_t = C_t h_t"}),e.jsxs("p",{className:"mt-1",children:["where ",e.jsx(t.InlineMath,{math:"\\bar{A}_t = \\exp(\\Delta_t A)"}),". This ",e.jsx("strong",{children:"breaks the convolution"})," structure but enables content-based reasoning."]})]}),e.jsx(S,{}),e.jsxs(x,{title:"Mamba vs Transformer Efficiency",children:[e.jsxs("p",{children:["For sequence length ",e.jsx(t.InlineMath,{math:"L"})," with model dimension ",e.jsx(t.InlineMath,{math:"D"})," and state dimension ",e.jsx(t.InlineMath,{math:"N"}),":"]}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsxs("li",{children:["Transformer attention: ",e.jsx(t.InlineMath,{math:"O(L^2 D)"})," FLOPs, ",e.jsx(t.InlineMath,{math:"O(L^2)"})," memory"]}),e.jsxs("li",{children:["Mamba (parallel scan): ",e.jsx(t.InlineMath,{math:"O(L D N)"})," FLOPs, ",e.jsx(t.InlineMath,{math:"O(L D N)"})," memory"]}),e.jsxs("li",{children:["Mamba generation: ",e.jsx(t.InlineMath,{math:"O(DN)"})," per token (constant, no KV-cache growth)"]}),e.jsx("li",{children:"At L=8192: Mamba is 5x faster inference, 3x faster training than Transformer++"})]})]}),e.jsx(p,{title:"Mamba Selective Scan (Simplified)",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMambaBlock(nn.Module):
    """Simplified Mamba block with selective state space."""
    def __init__(self, d_model=256, d_state=16, d_conv=4):
        super().__init__()
        self.d_state = d_state
        # Input projections
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv-1, groups=d_model)
        # Selection parameters (input-dependent)
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1, bias=False)  # B, C, dt
        self.dt_proj = nn.Linear(1, d_model, bias=True)
        # Fixed A (diagonal, log-parameterized)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()))

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)             # [B, L, 2D]
        x_main, z = xz.chunk(2, dim=-1)  # each [B, L, D]

        # Causal conv1d
        x_main = self.conv1d(x_main.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_main = F.silu(x_main)

        # Input-dependent parameters
        x_proj = self.x_proj(x_main)
        B_t = x_proj[:, :, :self.d_state]         # [B, L, N]
        C_t = x_proj[:, :, self.d_state:2*self.d_state]  # [B, L, N]
        dt = F.softplus(x_proj[:, :, -1:])         # [B, L, 1]

        # Selective scan (sequential for clarity)
        A = -torch.exp(self.A_log)                  # [N]
        h = torch.zeros(B, D, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            A_bar = torch.exp(dt[:, t] * A)         # [B, 1] * [N] -> [B, N]
            h = h * A_bar.unsqueeze(1) + x_main[:, t, :, None] * B_t[:, t, None, :]
            y_t = (h * C_t[:, t, None, :]).sum(-1)  # [B, D]
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                  # [B, L, D]
        return y * F.silu(z)                        # Gated output

mamba = SimpleMambaBlock(d_model=256, d_state=16)
x = torch.randn(2, 128, 256)
out = mamba(x)
print(f"Input: {x.shape} -> Output: {out.shape}")`}),e.jsx(g,{type:"note",title:"Hardware-Aware Algorithm",children:e.jsxs("p",{children:["Mamba uses a hardware-aware parallel scan algorithm that avoids materializing the full",e.jsx(t.InlineMath,{math:"(B, L, D, N)"})," state tensor in GPU HBM. Instead, it fuses the discretization, scan, and output computation in a single kernel, achieving near-optimal memory bandwidth utilization. This engineering is as important as the algorithmic innovation."]})})]})}const se=Object.freeze(Object.defineProperty({__proto__:null,default:L},Symbol.toStringTag,{value:"Module"}));function A(){const[i,o]=c.useState(32),[n,s]=c.useState(.25),a=Math.round(i*n),r=i-a,l=Array.from({length:i},(d,u)=>{const m=a>0?Math.round(i/a):i+1;return(u+1)%m===0?"attn":"ssm"});return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Hybrid Architecture Layer Configuration"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3 flex-wrap",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Total layers: ",i,e.jsx("input",{type:"range",min:8,max:64,step:4,value:i,onChange:d=>o(parseInt(d.target.value)),className:"w-24 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Attention ratio: ",(n*100).toFixed(0),"%",e.jsx("input",{type:"range",min:0,max:.5,step:.05,value:n,onChange:d=>s(parseFloat(d.target.value)),className:"w-24 accent-violet-500"})]})]}),e.jsx("div",{className:"flex gap-0.5 flex-wrap mb-2",children:l.map((d,u)=>e.jsx("div",{className:`w-5 h-5 rounded-sm text-[7px] flex items-center justify-center ${d==="attn"?"bg-violet-500 text-white":"bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300"}`,children:d==="attn"?"A":"S"},u))}),e.jsxs("p",{className:"text-xs text-gray-500",children:[r," SSM layers + ",a," attention layers. Attention provides in-context recall; SSMs handle long-range dependencies efficiently."]})]})}function C(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Hybrid architectures combine SSM layers with sparse attention layers, getting the efficiency of SSMs for most computation while retaining attention's ability for precise in-context recall. This approach powers models like Jamba, Zamba, and Mamba-2."}),e.jsxs(h,{title:"Hybrid SSM-Attention Block",children:[e.jsx("p",{children:"A hybrid model interleaves SSM and attention layers, often with a ratio of ~6:1 SSM to attention:"}),e.jsx(t.BlockMath,{math:"h_l = \\begin{cases} \\text{SSM}(h_{l-1}) + h_{l-1} & \\text{if } l \\notin \\mathcal{A} \\\\ \\text{Attn}(h_{l-1}) + h_{l-1} & \\text{if } l \\in \\mathcal{A} \\end{cases}"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"\\mathcal{A}"})," is the set of attention layers (e.g., every 6th layer). Both paths use pre-norm and residual connections."]})]}),e.jsx(A,{}),e.jsxs(x,{title:"Why Hybrids Outperform Pure Models",children:[e.jsx("p",{children:"SSMs and attention have complementary strengths:"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"SSMs excel at:"})," Long-range dependencies, efficient training, linear-time inference"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"SSMs struggle with:"})," Precise copying, in-context learning, associative recall"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Attention excels at:"})," Exact retrieval, in-context learning, copying patterns"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Attention struggles with:"})," Long sequences (quadratic cost), generation speed"]}),e.jsx("li",{children:"Jamba (AI21): 52B MoE + Mamba hybrid, outperforms Mixtral 8x7B"})]})]}),e.jsx(p,{title:"Hybrid SSM-Attention Model",code:`import torch
import torch.nn as nn

class SSMLayer(nn.Module):
    """Placeholder SSM layer (Mamba-style)."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)  # Simplified SSM
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.linear(self.norm(x)))

class AttentionLayer(nn.Module):
    """Standard multi-head attention layer."""
    def __init__(self, dim, heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):
        h = self.norm(x)
        out, _ = self.attn(h, h, h)
        return out

class HybridModel(nn.Module):
    """Hybrid SSM-Attention model with configurable ratio."""
    def __init__(self, dim=512, num_layers=24, attn_every=6):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if (i + 1) % attn_every == 0:
                self.layers.append(('attn', AttentionLayer(dim)))
            else:
                self.layers.append(('ssm', SSMLayer(dim)))
        self.layers = nn.ModuleList([l[1] for l in self.layers])
        self.layer_types = ['attn' if (i+1) % attn_every == 0 else 'ssm'
                           for i in range(num_layers)]

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
        return x

model = HybridModel(dim=256, num_layers=24, attn_every=6)
x = torch.randn(2, 128, 256)
out = model(x)
ssm_count = sum(1 for t in model.layer_types if t == 'ssm')
attn_count = sum(1 for t in model.layer_types if t == 'attn')
print(f"Layers: {ssm_count} SSM + {attn_count} Attention = {ssm_count + attn_count} total")
print(f"Output: {out.shape}")`}),e.jsx(g,{type:"note",title:"The Attention Tax",children:e.jsx("p",{children:"Even a small number of attention layers significantly improves in-context learning and retrieval performance. Research shows that 4 attention layers in a 24-layer model (17%) recovers nearly all of a pure Transformer's in-context learning ability, while keeping 83% of the inference speed advantage from SSM layers."})})]})}const re=Object.freeze(Object.defineProperty({__proto__:null,default:C},Symbol.toStringTag,{value:"Module"}));function T(){const[i,o]=c.useState("kaplan"),n=400,s=200,a=40,r={kaplan:{name:"Kaplan (OpenAI)",alpha:.076,label:"L(N) = (N_c/N)^0.076",color:"#8b5cf6"},chinchilla:{name:"Chinchilla (DeepMind)",alpha:.1,label:"L(N) = (N_c/N)^0.10",color:"#8b5cf6"}},l=r[i],d=Array.from({length:50},(f,b)=>{const j=6+b*.12,w=Math.pow(10,j),k=2*Math.pow(1e13/w,l.alpha)+1.5;return{logN:j,loss:k}}),u=f=>a+(f-6)/6*(n-2*a),m=f=>s-a-(f-1.5)/1.2*(s-2*a),y=d.map((f,b)=>`${b===0?"M":"L"}${u(f.logN)},${m(f.loss)}`).join(" ");return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Neural Scaling Law: Loss vs Parameters"}),e.jsx("div",{className:"flex gap-2 mb-3",children:Object.entries(r).map(([f,b])=>e.jsx("button",{onClick:()=>o(f),className:`px-3 py-1 rounded-lg text-sm transition ${i===f?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:b.name},f))}),e.jsxs("svg",{width:n,height:s,className:"mx-auto block",children:[e.jsx("line",{x1:a,y1:s-a,x2:n-a,y2:s-a,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("line",{x1:a,y1:a,x2:a,y2:s-a,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("path",{d:y,fill:"none",stroke:l.color,strokeWidth:2.5}),e.jsx("text",{x:n/2,y:s-5,textAnchor:"middle",className:"text-[10px] fill-gray-500",children:"log10(Parameters)"}),e.jsx("text",{x:12,y:s/2,textAnchor:"middle",transform:`rotate(-90,12,${s/2})`,className:"text-[10px] fill-gray-500",children:"Loss"})]}),e.jsxs("p",{className:"mt-1 text-xs text-gray-500 text-center",children:[l.label," — loss decreases as a power law with model size"]})]})}function B(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Neural scaling laws describe predictable power-law relationships between model performance and the key resources: parameters, training data, and compute. These laws enable principled decisions about resource allocation before training."}),e.jsxs(_,{title:"Kaplan Scaling Laws (2020)",id:"kaplan-scaling",children:[e.jsx("p",{children:"Cross-entropy loss follows power laws in each scaling variable independently:"}),e.jsx(t.BlockMath,{math:"L(N) = \\left(\\frac{N_c}{N}\\right)^{\\alpha_N}, \\quad L(D) = \\left(\\frac{D_c}{D}\\right)^{\\alpha_D}, \\quad L(C) = \\left(\\frac{C_c}{C}\\right)^{\\alpha_C}"}),e.jsxs("p",{className:"mt-2",children:["With ",e.jsx(t.InlineMath,{math:"\\alpha_N \\approx 0.076"}),", ",e.jsx(t.InlineMath,{math:"\\alpha_D \\approx 0.095"}),", ",e.jsx(t.InlineMath,{math:"\\alpha_C \\approx 0.050"}),". These exponents suggest model size matters more than data quantity — a conclusion later revised by Chinchilla."]})]}),e.jsx(T,{}),e.jsxs(h,{title:"Chinchilla Optimal Allocation",children:[e.jsxs("p",{children:["For a fixed compute budget ",e.jsx(t.InlineMath,{math:"C"}),", Chinchilla showed that parameters and tokens should scale equally:"]}),e.jsx(t.BlockMath,{math:"L(N, D) = E + \\frac{A}{N^{\\alpha}} + \\frac{B}{D^{\\beta}}"}),e.jsxs("p",{className:"mt-2",children:["with ",e.jsx(t.InlineMath,{math:"\\alpha \\approx 0.34"}),", ",e.jsx(t.InlineMath,{math:"\\beta \\approx 0.28"}),", and ",e.jsx(t.InlineMath,{math:"E"})," is the irreducible entropy. Minimizing ",e.jsx(t.InlineMath,{math:"L"})," subject to ",e.jsx(t.InlineMath,{math:"C = 6ND"})," gives ",e.jsx(t.InlineMath,{math:"N^* \\propto C^{0.5}"})," and ",e.jsx(t.InlineMath,{math:"D^* \\propto C^{0.5}"}),"."]})]}),e.jsx(p,{title:"Fitting and Predicting with Scaling Laws",code:`import numpy as np

def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    """Chinchilla parametric loss model.

    Args:
        N: number of parameters
        D: number of training tokens
        A, B, alpha, beta, E: fitted constants from Chinchilla paper
    """
    return E + A / N**alpha + B / D**beta

def optimal_allocation(C, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    """Find optimal N, D for compute budget C (FLOPs = 6*N*D)."""
    # Analytical solution from Lagrange multiplier
    a = alpha
    b = beta
    # N* proportional to C^(b/(a+b)), D* proportional to C^(a/(a+b))
    ratio = (a * B) / (b * A)
    N_star = (C / 6 * ratio**(b/(a+b)))**(1/(1 + b/a))
    D_star = C / (6 * N_star)
    return N_star, D_star

# Predict loss for different compute budgets
for log_c in [21, 22, 23, 24, 25]:
    C = 10**log_c
    N, D = optimal_allocation(C)
    loss = chinchilla_loss(N, D)
    print(f"C=10^{log_c}: N={N/1e9:.1f}B, D={D/1e9:.0f}B tokens, Loss={loss:.3f}")

# Compare: GPT-3 (undertrained) vs Chinchilla-optimal
gpt3_loss = chinchilla_loss(175e9, 300e9)
chin_loss = chinchilla_loss(70e9, 1.4e12)
print(f"\\nGPT-3 (175B, 300B tok): {gpt3_loss:.3f}")
print(f"Chinchilla (70B, 1.4T tok): {chin_loss:.3f}")`}),e.jsx(x,{title:"Key Takeaways from Scaling Laws",children:e.jsxs("ul",{className:"list-disc list-inside space-y-1",children:[e.jsx("li",{children:"Loss follows smooth, predictable power laws across many orders of magnitude"}),e.jsx("li",{children:"Small-scale experiments can predict large-scale performance"}),e.jsx("li",{children:"Kaplan overestimated the importance of model size (fixed training tokens)"}),e.jsx("li",{children:"Chinchilla showed data and parameters should scale equally with compute"}),e.jsx("li",{children:"Modern practice (LLaMA-3) deliberately over-trains for inference efficiency"})]})}),e.jsx(g,{type:"note",title:"Scaling Laws Beyond Language",children:e.jsx("p",{children:"Power-law scaling has been observed across modalities: vision (ViT), speech, code generation, mathematical reasoning, and multimodal models. However, the exponents and constants vary. Downstream task performance sometimes deviates from training loss scaling, particularly for tasks requiring specific capabilities that may emerge suddenly."})})]})}const oe=Object.freeze(Object.defineProperty({__proto__:null,default:B},Symbol.toStringTag,{value:"Module"}));function I(){const[i,o]=c.useState(23),n=Math.pow(10,i),s=Math.pow(n/6/20,.5)*Math.sqrt(20),a=n/(6*s),r=s*.5,l=n/(6*r);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Compute Budget Allocation Strategies"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Compute budget: 10^",i," FLOPs",e.jsx("input",{type:"range",min:20,max:26,step:.5,value:i,onChange:d=>o(parseFloat(d.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("div",{className:"grid grid-cols-2 gap-3 text-sm",children:[e.jsxs("div",{className:"p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20",children:[e.jsx("p",{className:"font-medium text-violet-700 dark:text-violet-300",children:"Chinchilla-Optimal"}),e.jsxs("p",{children:["N: ",(s/1e9).toFixed(1),"B params"]}),e.jsxs("p",{children:["D: ",(a/1e9).toFixed(0),"B tokens"]}),e.jsx("p",{className:"text-xs text-gray-500",children:"Tokens/param ratio: ~20"})]}),e.jsxs("div",{className:"p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20",children:[e.jsx("p",{className:"font-medium text-violet-700 dark:text-violet-300",children:"Inference-Optimal (LLaMA-style)"}),e.jsxs("p",{children:["N: ",(r/1e9).toFixed(1),"B params"]}),e.jsxs("p",{children:["D: ",(l/1e9).toFixed(0),"B tokens"]}),e.jsxs("p",{className:"text-xs text-gray-500",children:["Tokens/param ratio: ~",(l/r).toFixed(0)]})]})]}),e.jsx("p",{className:"mt-2 text-xs text-gray-500 text-center",children:"Same compute budget, different allocation — smaller model trained on more data is cheaper at inference."})]})}function D(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Compute-optimal training balances model size and training tokens for a fixed compute budget. The Chinchilla result showed most LLMs were significantly undertrained, but modern practice deliberately deviates for inference efficiency."}),e.jsxs(h,{title:"Compute-Optimal vs Inference-Optimal",children:[e.jsxs("p",{children:[e.jsx("strong",{children:"Compute-optimal"})," minimizes loss for a fixed training FLOP budget ",e.jsx(t.InlineMath,{math:"C"}),":"]}),e.jsx(t.BlockMath,{math:"\\min_{N,D: 6ND = C} L(N, D) \\implies D^* \\approx 20N^*"}),e.jsxs("p",{className:"mt-2",children:[e.jsx("strong",{children:"Inference-optimal"})," minimizes total cost (training + inference) over the model lifetime:"]}),e.jsx(t.BlockMath,{math:"\\min_{N,D} C_{\\text{train}}(N, D) + T \\cdot C_{\\text{inference}}(N) \\quad \\text{s.t. } L(N, D) \\leq L_{\\text{target}}"}),e.jsxs("p",{className:"mt-1",children:["where ",e.jsx(t.InlineMath,{math:"T"})," is the expected number of inference tokens. For high ",e.jsx(t.InlineMath,{math:"T"}),", smaller models trained on more data are preferred."]})]}),e.jsx(I,{}),e.jsxs(x,{title:"Why LLaMA-3 Over-Trains",children:[e.jsx("p",{children:"LLaMA-3 70B trains on 15T tokens (~215 tokens/parameter vs Chinchilla's 20):"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsx("li",{children:"Training cost: ~10x Chinchilla-optimal for this model size"}),e.jsx("li",{children:"But inference cost: 1x the 70B model (vs a 400B Chinchilla-optimal model)"}),e.jsx("li",{children:"After ~1M inference requests, the total cost is lower than the larger model"}),e.jsx("li",{children:"The log-loss reduction continues smoothly even at 200+ tokens/parameter"})]})]}),e.jsx(p,{title:"Compute-Optimal vs Inference-Optimal Analysis",code:`import math

def total_cost(N, D, inference_tokens, cost_per_flop=1e-18):
    """Total cost = training + inference over model lifetime."""
    train_flops = 6 * N * D
    # Inference: ~2N FLOPs per token (forward pass only)
    infer_flops = 2 * N * inference_tokens
    total_flops = train_flops + infer_flops
    return total_flops * cost_per_flop

def chinchilla_loss(N, D):
    """Approximate Chinchilla loss model."""
    return 1.69 + 406.4 / N**0.34 + 410.7 / D**0.28

# Compare strategies for target loss of 1.85
target_loss = 1.85

# Strategy 1: Chinchilla-optimal (N = D/20)
# Find N such that chinchilla_loss(N, 20*N) = target
N_chin = 70e9
D_chin = 20 * N_chin

# Strategy 2: Over-trained (same loss, smaller model)
N_small = 30e9
D_small = 200 * N_small  # 200 tokens/param

# Compare total costs for different inference workloads
print(f"{'Inference Tokens':>20} {'Chinchilla Cost':>18} {'Over-trained Cost':>18} {'Winner':>10}")
print("-" * 70)
for log_infer in [12, 13, 14, 15]:
    infer_tok = 10**log_infer
    cost_chin = total_cost(N_chin, D_chin, infer_tok)
    cost_small = total_cost(N_small, D_small, infer_tok)
    winner = "Chinchilla" if cost_chin < cost_small else "Over-train"
    print(f"  10^{log_infer} tokens    {cost_chin:.2f}          {cost_small:.2f}           {winner}")
print("\\nConclusion: Over-training wins when inference demand is high")`}),e.jsx(g,{type:"note",title:"Beyond Power Laws: Data Quality",children:e.jsx("p",{children:'Scaling laws assume infinite unique data. In practice, data quality, diversity, and deduplication matter as much as quantity. Training on curated, high-quality data can achieve the same loss as 10x more unfiltered data. The "data wall" — running out of high-quality internet text — is a growing concern for continued scaling.'})})]})}const le=Object.freeze(Object.defineProperty({__proto__:null,default:D},Symbol.toStringTag,{value:"Module"}));function P(){const[i,o]=c.useState(50),[n,s]=c.useState(20),[a,r]=c.useState(15),l=i+n+a,d=Math.max(0,100-l),u=[{name:"Web Text",pct:i,color:"bg-violet-400"},{name:"Code",pct:n,color:"bg-violet-600"},{name:"Books/Papers",pct:a,color:"bg-violet-300"},{name:"Other (wiki, etc.)",pct:d,color:"bg-gray-300"}];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Training Data Mixture"}),e.jsxs("div",{className:"space-y-2 mb-3",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Web: ",i,"% ",e.jsx("input",{type:"range",min:0,max:80,value:i,onChange:m=>o(parseInt(m.target.value)),className:"w-32 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Code: ",n,"% ",e.jsx("input",{type:"range",min:0,max:40,value:n,onChange:m=>s(parseInt(m.target.value)),className:"w-32 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Books: ",a,"% ",e.jsx("input",{type:"range",min:0,max:30,value:a,onChange:m=>r(parseInt(m.target.value)),className:"w-32 accent-violet-500"})]})]}),e.jsx("div",{className:"h-6 flex rounded overflow-hidden",children:u.map((m,y)=>m.pct>0&&e.jsx("div",{className:`${m.color} transition-all`,style:{width:`${m.pct}%`}},y))}),e.jsx("div",{className:"flex gap-3 mt-2 flex-wrap text-xs text-gray-500",children:u.map((m,y)=>e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:`w-2 h-2 rounded ${m.color}`}),m.name,": ",m.pct,"%"]},y))}),l>100&&e.jsx("p",{className:"text-xs text-red-500 mt-1",children:"Total exceeds 100% — reduce one category"})]})}function O(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Data quality, diversity, and mixture composition have emerged as critical factors in LLM training — often more impactful than raw data volume. Research into data curation, filtering, and synthetic data generation is reshaping how models are trained."}),e.jsxs(h,{title:"Data-Constrained Scaling",children:[e.jsxs("p",{children:["When unique data ",e.jsx(t.InlineMath,{math:"D_u"})," is limited, repeating data has diminishing returns. The effective data follows:"]}),e.jsx(t.BlockMath,{math:"D_{\\text{eff}}(R, D_u) = D_u \\cdot (1 - e^{-R})"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"R = D_{\\text{total}} / D_u"})," is the number of epochs. After ~4 epochs, additional repetitions contribute minimally. The loss modification becomes:"]}),e.jsx(t.BlockMath,{math:"L(N, D_u, R) = E + \\frac{A}{N^{\\alpha}} + \\frac{B}{D_{\\text{eff}}(R, D_u)^{\\beta}}"})]}),e.jsx(P,{}),e.jsxs(x,{title:"Data Quality Interventions",children:[e.jsx("p",{children:"Key data curation techniques and their impact:"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Deduplication:"})," Removing near-duplicates improves perplexity by 0.1-0.3 nats and reduces memorization"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Quality filtering:"})," Perplexity-based or classifier-based filtering yields 2-5x data efficiency gains"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Domain upsampling:"})," Over-representing high-quality domains (Wikipedia, textbooks) improves downstream tasks"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Synthetic data:"})," LLM-generated data can supplement scarce domains (math, reasoning)"]})]})]}),e.jsx(p,{title:"Data Quality Filtering Pipeline",code:`import math

def perplexity_filter(documents, reference_model_ppl, threshold_low=10, threshold_high=1000):
    """Filter documents by perplexity from a reference model.

    - Too low perplexity: likely repetitive/template text
    - Too high perplexity: likely noise/gibberish
    - Medium perplexity: natural, informative text
    """
    filtered = []
    stats = {"kept": 0, "too_low": 0, "too_high": 0}
    for doc, ppl in zip(documents, reference_model_ppl):
        if ppl < threshold_low:
            stats["too_low"] += 1
        elif ppl > threshold_high:
            stats["too_high"] += 1
        else:
            filtered.append(doc)
            stats["kept"] += 1
    return filtered, stats

def effective_data(unique_tokens, epochs):
    """Compute effective training data with repetition."""
    return unique_tokens * (1 - math.exp(-epochs))

# Demonstrate diminishing returns of data repetition
unique = 1e12  # 1T unique tokens
print("Epochs | Total Tokens | Effective Tokens | Efficiency")
print("-" * 55)
for e in [1, 2, 4, 8, 16]:
    total = unique * e
    eff = effective_data(unique, e)
    efficiency = eff / total * 100
    print(f"  {e:>4}  | {total/1e12:.1f}T          | {eff/1e12:.2f}T            | {efficiency:.1f}%")
print("\\nAfter ~4 epochs, >98% of effective data has been captured")`}),e.jsx(v,{title:"The Data Wall",children:e.jsx("p",{children:'Estimates suggest only 1-10T tokens of high-quality natural text exist on the internet. With leading models already consuming 15T+ tokens (with repetition), the field is approaching a "data wall." Solutions include synthetic data generation, multimodal data, and more data-efficient training methods. This is one of the most pressing challenges for continued scaling.'})})]})}const ce=Object.freeze(Object.defineProperty({__proto__:null,default:O},Symbol.toStringTag,{value:"Module"}));function z(){const[i,o]=c.useState("induction"),n={induction:{name:"Induction Heads",desc:'Copy patterns from earlier context: if "A B ... A" appears, predict "B". Found in layer 1-2 pairs.',mechanism:"Head 0 in L1 attends to previous tokens → Head 5 in L2 attends to token after previous occurrence of current token → copies that token to output"},ioi:{name:"IOI (Indirect Object)",desc:'In "Alice gave Bob the ball. Alice gave ___", predict "Bob" not "Alice".',mechanism:"Duplicate token heads detect repeated names → S-inhibition heads suppress the repeated name → Name mover heads promote the non-repeated name to output"},superposition:{name:"Superposition",desc:"Networks represent more features than dimensions by encoding sparse features in overlapping directions.",mechanism:"With D dimensions and F >> D features, each feature is a direction in R^D. Features are approximately orthogonal when sparse, allowing interference to be tolerable."}},s=n[i];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Known Neural Circuits"}),e.jsx("div",{className:"flex gap-2 mb-3 flex-wrap",children:Object.entries(n).map(([a,r])=>e.jsx("button",{onClick:()=>o(a),className:`px-3 py-1 rounded-lg text-sm transition ${i===a?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:r.name},a))}),e.jsxs("div",{className:"p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20 text-sm space-y-2",children:[e.jsx("p",{className:"font-medium text-violet-700 dark:text-violet-300",children:s.name}),e.jsx("p",{className:"text-gray-600 dark:text-gray-400",children:s.desc}),e.jsxs("p",{className:"text-xs text-gray-500",children:[e.jsx("strong",{children:"Mechanism:"})," ",s.mechanism]})]})]})}function q(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Mechanistic interpretability aims to reverse-engineer neural networks by identifying meaningful features (directions in activation space) and circuits (compositions of features across layers). This approach treats models as programs to be understood, not black boxes."}),e.jsxs(h,{title:"Features and Circuits",children:[e.jsxs("p",{children:["A ",e.jsx("strong",{children:"feature"})," is a direction in activation space that corresponds to a human-interpretable concept. A ",e.jsx("strong",{children:"circuit"})," is a subgraph of the network that implements a specific computation:"]}),e.jsx(t.BlockMath,{math:"\\text{Circuit} = \\{(\\text{feature}_i^{(l)}, W_{ij}^{(l \\to l+1)}) : \\text{implements behavior } f\\}"}),e.jsxs("p",{className:"mt-2",children:["The ",e.jsx("strong",{children:"linear representation hypothesis"})," claims that features are represented as directions in activation space: ",e.jsx(t.InlineMath,{math:"v_f \\in \\mathbb{R}^d"})," where ",e.jsx(t.InlineMath,{math:"\\langle h, v_f \\rangle"})," measures the presence of feature ",e.jsx(t.InlineMath,{math:"f"})," in hidden state ",e.jsx(t.InlineMath,{math:"h"}),"."]})]}),e.jsx(z,{}),e.jsxs(x,{title:"Induction Heads: A Universal Circuit",children:[e.jsx("p",{children:"Induction heads are found in virtually all transformer LLMs and implement in-context pattern matching:"}),e.jsxs("ol",{className:"list-decimal list-inside mt-2 space-y-1",children:[e.jsx("li",{children:'A "previous token head" in layer L attends to the token before the current token'}),e.jsx("li",{children:'An "induction head" in layer L+1 uses this to find where the current token appeared before'}),e.jsxs("li",{children:["It then copies the ",e.jsx("em",{children:"next"})," token from that earlier occurrence to the output"]})]}),e.jsx("p",{className:"mt-2",children:'This two-layer circuit is responsible for much of in-context learning ability and appears to form during a sharp phase transition in training ("induction bump").'})]}),e.jsx(p,{title:"Analyzing Attention Patterns for Circuits",code:`import torch
import torch.nn.functional as F

def compute_attention_pattern(Q, K, mask=None):
    """Compute attention weights for circuit analysis."""
    d_k = Q.shape[-1]
    attn = Q @ K.transpose(-2, -1) / d_k**0.5
    if mask is not None:
        attn = attn.masked_fill(mask == 0, -1e9)
    return F.softmax(attn, dim=-1)

def find_induction_heads(attn_weights, offset=1):
    """Score attention heads for induction behavior.

    Induction heads attend to positions where the current
    token appeared previously, shifted by 'offset'.
    High score on [seq-offset] diagonal = induction head.
    """
    # attn_weights: [heads, seq_len, seq_len]
    H, S, _ = attn_weights.shape
    scores = torch.zeros(H)
    for h in range(H):
        # Check attention on the offset-shifted diagonal
        diag_sum = 0
        count = 0
        for i in range(offset, S):
            diag_sum += attn_weights[h, i, i - offset].item()
            count += 1
        scores[h] = diag_sum / count if count > 0 else 0
    return scores

# Simulate: 12 heads, seq_len=32
attn = torch.rand(12, 32, 32)
attn = F.softmax(attn * 5, dim=-1)  # sharpen
# Make head 5 an induction head (attend to offset-1 diagonal)
for i in range(1, 32):
    attn[5, i, :] *= 0.01
    attn[5, i, i-1] = 0.9
scores = find_induction_heads(attn)
print("Induction head scores:", [f"H{i}:{s:.2f}" for i, s in enumerate(scores)])`}),e.jsx(g,{type:"note",title:"The Superposition Hypothesis",children:e.jsxs("p",{children:["Neural networks appear to represent more features than they have dimensions, using",e.jsx("strong",{children:" superposition"})," — encoding multiple features as nearly-orthogonal directions in the same space. This makes individual neurons ",e.jsx("em",{children:"polysemantic"})," (responding to multiple unrelated concepts), complicating interpretability. Sparse autoencoders aim to untangle superposition into monosemantic features."]})})]})}const de=Object.freeze(Object.defineProperty({__proto__:null,default:q},Symbol.toStringTag,{value:"Module"}));function R(){const[i,o]=c.useState("activation"),n={activation:{name:"Activation Patching",desc:"Replace activations at a specific position and layer from a clean run with those from a corrupted run. Measure how much the output changes.",formula:"\\Delta y = f(h_{\\text{clean}}) - f(h_{\\text{clean}} \\text{ with } h^{(l)}_i \\leftarrow h^{(l)}_{i,\\text{corrupt}})"},path:{name:"Path Patching",desc:"Patch only the connection between two specific components (e.g., head A to head B), isolating the causal effect of a specific pathway.",formula:"\\Delta y = f(h_{\\text{clean}}) - f(h_{\\text{clean}} \\text{ with edge } A{\\to}B \\text{ corrupted})"},resample:{name:"Causal Scrubbing",desc:"Systematically test a proposed computational graph by resampling activations at each node. If the hypothesis is correct, resampling should preserve behavior.",formula:"\\text{Match}(G) = 1 - \\frac{\\mathbb{E}[\\text{KL}(f(x) \\| f_{\\text{scrubbed}}(x))]}{\\mathbb{E}[\\text{KL}(f(x) \\| \\text{uniform})]}"}},s=n[i];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Interpretability Techniques"}),e.jsx("div",{className:"flex gap-2 mb-3 flex-wrap",children:Object.entries(n).map(([a,r])=>e.jsx("button",{onClick:()=>o(a),className:`px-3 py-1 rounded-lg text-sm transition ${i===a?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:r.name},a))}),e.jsxs("div",{className:"p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20 text-sm space-y-2",children:[e.jsx("p",{className:"text-gray-600 dark:text-gray-400",children:s.desc}),e.jsx(t.BlockMath,{math:s.formula})]})]})}function F(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Probing classifiers and activation patching are two key techniques for understanding what information neural networks encode and how that information flows through the network to produce outputs."}),e.jsxs(h,{title:"Linear Probing",children:[e.jsx("p",{children:"Train a linear classifier on frozen internal representations to test what information is encoded:"}),e.jsx(t.BlockMath,{math:"\\hat{y} = \\text{softmax}(W_{\\text{probe}} \\cdot h^{(l)}_i + b)"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"h^{(l)}_i"})," is the hidden state at layer ",e.jsx(t.InlineMath,{math:"l"}),", position ",e.jsx(t.InlineMath,{math:"i"}),". High probe accuracy indicates the information is linearly accessible. The probe should be ",e.jsx("em",{children:"simple"})," (linear) to avoid learning the task itself rather than detecting pre-existing representations."]})]}),e.jsx(R,{}),e.jsxs(x,{title:"What Probes Reveal About LLMs",children:[e.jsx("p",{children:"Linear probes on GPT-2 and LLaMA have uncovered:"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsx("li",{children:"Part-of-speech: 97%+ accuracy from early layers (layer 2-3)"}),e.jsx("li",{children:"Syntactic parse trees: recoverable from middle layers"}),e.jsx("li",{children:"Factual knowledge (entity types, relationships): peaks in middle layers"}),e.jsx("li",{children:"Next-token prediction: only accurate from final layers"}),e.jsx("li",{children:"Pattern: low layers = syntax, middle = semantics, high = task-specific"})]})]}),e.jsx(p,{title:"Linear Probing and Activation Patching",code:`import torch
import torch.nn as nn

class LinearProbe(nn.Module):
    """Linear probe for testing information in representations."""
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, hidden_states):
        return self.linear(hidden_states.detach())  # Detach!

def activation_patching(model, clean_input, corrupt_input, layer, position):
    """Measure causal importance of activation at (layer, position).

    1. Run model on clean input, save all activations
    2. Run model on corrupt input, save activation at target (layer, pos)
    3. Run model on clean input but replace target activation with corrupt one
    4. Measure change in output logits
    """
    # Get clean activations and output
    clean_acts = {}
    def save_hook(name):
        def hook(module, input, output):
            clean_acts[name] = output.detach().clone()
        return hook

    # Register hooks on each layer (pseudocode)
    # ... hooks = [model.layers[i].register_hook(save_hook(f"layer_{i}"))]

    # Get corrupt activation at target location
    # corrupt_act = run_model(corrupt_input).activations[layer][position]

    # Patch: replace clean activation with corrupt one at (layer, position)
    # patched_output = run_model_with_patch(clean_input, layer, position, corrupt_act)

    # Measure effect: large change = this activation is important
    # effect = (clean_output - patched_output).norm()

    print("Activation patching workflow:")
    print("1. Clean run -> save activations + output logits")
    print("2. Corrupt run -> save target activation")
    print("3. Patched run -> replace one activation, measure output change")
    print("4. Large change = causally important component")

activation_patching(None, None, None, layer=10, position=15)`}),e.jsx(g,{type:"note",title:"Probing Limitations",children:e.jsxs("p",{children:["Probing has important caveats: (1) high probe accuracy does not mean the model ",e.jsx("em",{children:"uses"}),'that information — it might be an artifact, (2) low probe accuracy does not mean the information is absent — it might be encoded nonlinearly, (3) probe complexity matters — an MLP probe can "learn" to extract information not linearly present. Activation patching provides stronger causal evidence than probing alone.']})})]})}const me=Object.freeze(Object.defineProperty({__proto__:null,default:F},Symbol.toStringTag,{value:"Module"}));function E(){const[i,o]=c.useState(768),[n,s]=c.useState(32),a=i*n,r=Math.round(a*.005);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Sparse Autoencoder Dimensions"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3 flex-wrap",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["d_model: ",i,e.jsx("input",{type:"range",min:256,max:4096,step:256,value:i,onChange:l=>o(parseInt(l.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Expansion: ",n,"x",e.jsx("input",{type:"range",min:4,max:128,step:4,value:n,onChange:l=>s(parseInt(l.target.value)),className:"w-28 accent-violet-500"})]})]}),e.jsxs("div",{className:"grid grid-cols-3 gap-3 text-sm text-center",children:[e.jsxs("div",{className:"p-2 rounded bg-violet-50 dark:bg-violet-900/20",children:[e.jsx("p",{className:"text-violet-700 dark:text-violet-300 font-medium",children:"SAE Features"}),e.jsx("p",{className:"font-bold",children:a.toLocaleString()})]}),e.jsxs("div",{className:"p-2 rounded bg-violet-50 dark:bg-violet-900/20",children:[e.jsx("p",{className:"text-violet-700 dark:text-violet-300 font-medium",children:"Active per Input"}),e.jsxs("p",{className:"font-bold",children:["~",r," (",(r/a*100).toFixed(1),"%)"]})]}),e.jsxs("div",{className:"p-2 rounded bg-violet-50 dark:bg-violet-900/20",children:[e.jsx("p",{className:"text-violet-700 dark:text-violet-300 font-medium",children:"Sparsity"}),e.jsxs("p",{className:"font-bold",children:[(100-r/a*100).toFixed(1),"%"]})]})]}),e.jsx("p",{className:"mt-2 text-xs text-gray-500 text-center",children:"Each input activates only a tiny fraction of the feature dictionary, ensuring monosemantic features."})]})}function H(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Sparse autoencoders (SAEs) decompose neural network activations into a large set of interpretable, monosemantic features. By learning an overcomplete dictionary with sparsity constraints, SAEs untangle the superposition that makes individual neurons hard to interpret."}),e.jsxs(h,{title:"Sparse Autoencoder for Interpretability",children:[e.jsxs("p",{children:["Given an activation vector ",e.jsx(t.InlineMath,{math:"h \\in \\mathbb{R}^d"}),", the SAE learns an encoder-decoder pair with a sparsity penalty:"]}),e.jsx(t.BlockMath,{math:"f = \\text{ReLU}(W_{\\text{enc}}(h - b_d) + b_e), \\quad \\hat{h} = W_{\\text{dec}} f + b_d"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = \\|h - \\hat{h}\\|^2 + \\lambda \\|f\\|_1"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"f \\in \\mathbb{R}^{D}"})," with ",e.jsx(t.InlineMath,{math:"D \\gg d"})," is the sparse feature vector. The L1 penalty on ",e.jsx(t.InlineMath,{math:"f"})," encourages most features to be zero for any given input, yielding monosemantic features."]})]}),e.jsx(E,{}),e.jsxs(x,{title:"Discovered SAE Features (Claude/GPT-2)",children:[e.jsx("p",{children:"SAEs trained on language models have discovered interpretable features for:"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsx("li",{children:'Specific concepts: "Golden Gate Bridge", "DNA sequences", "Python code"'}),e.jsx("li",{children:'Abstract patterns: "start of a list item", "the answer is about to be stated"'}),e.jsx("li",{children:'Safety-relevant: "deceptive reasoning", "refusing harmful requests"'}),e.jsx("li",{children:'Linguistic: "past tense verbs", "words ending in -tion"'}),e.jsx("li",{children:"These features are more interpretable than individual neurons (95%+ vs ~30%)"})]})]}),e.jsx(p,{title:"Training a Sparse Autoencoder",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    """Sparse autoencoder for mechanistic interpretability."""
    def __init__(self, d_model, d_sae, k_sparse=None):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_sae)
        self.decoder = nn.Linear(d_sae, d_model, bias=True)
        # Normalize decoder columns to unit norm
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
        self.k_sparse = k_sparse  # If set, use top-k instead of L1

    def forward(self, h):
        # Subtract decoder bias (centering)
        h_centered = h - self.decoder.bias
        # Encode to sparse features
        f = self.encoder(h_centered)
        if self.k_sparse:
            # TopK activation: keep only top-k features
            topk_vals, topk_idx = f.topk(self.k_sparse, dim=-1)
            f = torch.zeros_like(f).scatter(-1, topk_idx, F.relu(topk_vals))
        else:
            f = F.relu(f)
        # Decode
        h_hat = self.decoder(f)
        return h_hat, f

    def loss(self, h, lambda_l1=5e-3):
        h_hat, f = self.forward(h)
        recon_loss = (h - h_hat).pow(2).mean()
        sparsity_loss = f.abs().mean()
        return recon_loss + lambda_l1 * sparsity_loss, recon_loss, sparsity_loss

# Train on random activations (demo)
sae = SparseAutoencoder(d_model=768, d_sae=768*32)
h = torch.randn(128, 768)  # batch of activations
total_loss, recon, sparse = sae.loss(h)
_, features = sae(h)
active = (features > 0).float().mean()
print(f"Reconstruction loss: {recon.item():.4f}")
print(f"Sparsity loss: {sparse.item():.4f}")
print(f"Active features: {active.item()*100:.2f}% ({int(active.item()*768*32)} of {768*32})")`}),e.jsx(g,{type:"note",title:"TopK SAEs and Scaling",children:e.jsx("p",{children:"Recent work replaces the L1 penalty with a TopK activation function, directly enforcing exactly K active features per input. This avoids the reconstruction-sparsity tradeoff of L1 and scales better to very large dictionaries (millions of features). Anthropic's work on Claude found that scaling SAE width reveals increasingly fine-grained features following a power law."})})]})}const he=Object.freeze(Object.defineProperty({__proto__:null,default:H},Symbol.toStringTag,{value:"Module"}));function W(){const[i,o]=c.useState("few_shot"),n={few_shot:{name:"Few-Shot CoT",desc:"Provide examples with step-by-step solutions in the prompt. The model mimics the reasoning pattern.",example:"Q: Roger has 5 balls. He buys 2 cans of 3. How many? A: He started with 5. 2 cans of 3 = 6. 5 + 6 = 11. The answer is 11.",improvement:"+15-25% on math/reasoning benchmarks"},zero_shot:{name:"Zero-Shot CoT",desc:`Simply append "Let's think step by step" to the prompt. Surprisingly effective without examples.`,example:"Q: [question] A: Let's think step by step...",improvement:"+10-15% on average, no examples needed"},self_refine:{name:"Self-Refinement",desc:"Generate an initial answer, then critique and revise it. Multiple rounds of refinement possible.",example:'Initial answer -> "Wait, let me check..." -> Revised answer -> "Yes, this is correct"',improvement:"+5-10% additional over basic CoT"}},s=n[i];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Chain-of-Thought Methods"}),e.jsx("div",{className:"flex gap-2 mb-3 flex-wrap",children:Object.entries(n).map(([a,r])=>e.jsx("button",{onClick:()=>o(a),className:`px-3 py-1 rounded-lg text-sm transition ${i===a?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:r.name},a))}),e.jsxs("div",{className:"p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20 text-sm space-y-2",children:[e.jsx("p",{className:"text-gray-600 dark:text-gray-400",children:s.desc}),e.jsx("p",{className:"text-xs font-mono bg-white dark:bg-gray-800 p-2 rounded",children:s.example}),e.jsx("p",{className:"text-xs text-violet-600 dark:text-violet-400 font-medium",children:s.improvement})]})]})}function K(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:'Chain-of-thought (CoT) prompting transforms reasoning problems by having the model generate intermediate steps before the final answer. This converts single-step prediction into multi-step computation, effectively giving the model "thinking time."'}),e.jsxs(h,{title:"Chain-of-Thought as Computation",children:[e.jsxs("p",{children:["Standard prediction directly maps input to answer: ",e.jsx(t.InlineMath,{math:"p(a|q)"}),". CoT introduces intermediate reasoning tokens ",e.jsx(t.InlineMath,{math:"r_1, \\ldots, r_n"}),":"]}),e.jsx(t.BlockMath,{math:"p(a|q) = \\sum_{r_1, \\ldots, r_n} p(r_1|q) \\cdot p(r_2|q, r_1) \\cdots p(a|q, r_1, \\ldots, r_n)"}),e.jsx("p",{className:"mt-2",children:"Each reasoning token adds computation. A model generating 100 reasoning tokens performs ~100x more FLOPs than direct answering. This trades inference compute for accuracy."})]}),e.jsx(W,{}),e.jsxs(x,{title:"When Does CoT Help?",children:[e.jsx("p",{children:"CoT is most effective for multi-step reasoning tasks:"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Math:"})," GSM8K +35% (PaLM 540B), MATH +15%"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Logic:"})," Symbolic reasoning, constraint satisfaction"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Common sense:"})," Multi-hop questions requiring world knowledge"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Minimal help:"})," Factual recall, sentiment analysis, simple classification"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Key condition:"})," Model must be large enough (~100B+) to generate coherent reasoning"]})]})]}),e.jsx(p,{title:"Implementing Chain-of-Thought with Verification",code:`def chain_of_thought_with_verification(llm, question, max_retries=3):
    """Generate CoT reasoning with self-verification.

    1. Generate step-by-step solution
    2. Ask model to verify each step
    3. If verification fails, regenerate from the error
    """
    for attempt in range(max_retries):
        # Step 1: Generate CoT
        cot_prompt = f"""Solve step by step:
Q: {question}
A: Let me work through this carefully."""
        reasoning = llm.generate(cot_prompt)

        # Step 2: Verify
        verify_prompt = f"""Verify this solution step by step.
Question: {question}
Solution: {reasoning}

Check each step. Is the final answer correct? Reply YES or NO with explanation."""
        verification = llm.generate(verify_prompt)

        if "YES" in verification.upper():
            return reasoning, attempt + 1

        # Step 3: Self-correct
        print(f"Attempt {attempt + 1} failed verification, retrying...")

    return reasoning, max_retries  # Return best attempt

# Compute tokens used
def estimate_cot_tokens(direct_tokens=5, cot_tokens=150, verification_tokens=100):
    """Compare compute for direct vs CoT with verification."""
    direct_flops = direct_tokens
    cot_flops = cot_tokens + verification_tokens
    print(f"Direct answer: ~{direct_tokens} tokens")
    print(f"CoT + verify: ~{cot_tokens + verification_tokens} tokens")
    print(f"Compute multiplier: {cot_flops / direct_flops:.0f}x")
    print(f"Accuracy improvement: typically 20-40% on reasoning tasks")

estimate_cot_tokens()`}),e.jsx(g,{type:"note",title:"Reasoning Models (o1, R1)",children:e.jsxs("p",{children:["Models like OpenAI's o1 and DeepSeek R1 are trained to produce long internal reasoning chains before answering. Unlike prompting-based CoT, these models learn ",e.jsx("em",{children:"when"})," and",e.jsx("em",{children:"how"})," to reason through reinforcement learning. They often outperform larger models that use standard prompting, demonstrating that test-time compute can substitute for model scale."]})})]})}const pe=Object.freeze(Object.defineProperty({__proto__:null,default:K},Symbol.toStringTag,{value:"Module"}));function V(){const[i,o]=c.useState(.5),[n,s]=c.useState(16),r=1-Math.pow(1-i,n),l=Math.min(.99,i+.15*Math.log2(n));return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Best-of-N vs Majority Voting"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3 flex-wrap",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Base accuracy: ",(i*100).toFixed(0),"%",e.jsx("input",{type:"range",min:.1,max:.9,step:.05,value:i,onChange:d=>o(parseFloat(d.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["N samples: ",n,e.jsx("input",{type:"range",min:1,max:128,step:1,value:n,onChange:d=>s(parseInt(d.target.value)),className:"w-28 accent-violet-500"})]})]}),e.jsxs("div",{className:"grid grid-cols-3 gap-3 text-sm text-center",children:[e.jsxs("div",{className:"p-2 rounded bg-gray-100 dark:bg-gray-800",children:[e.jsx("p",{className:"text-gray-500 font-medium",children:"Single Sample"}),e.jsxs("p",{className:"text-lg font-bold",children:[(i*100).toFixed(0),"%"]})]}),e.jsxs("div",{className:"p-2 rounded bg-violet-50 dark:bg-violet-900/20",children:[e.jsxs("p",{className:"text-violet-700 dark:text-violet-300 font-medium",children:["Majority Vote (N=",n,")"]}),e.jsxs("p",{className:"text-lg font-bold",children:[(l*100).toFixed(1),"%"]})]}),e.jsxs("div",{className:"p-2 rounded bg-violet-100 dark:bg-violet-900/40",children:[e.jsxs("p",{className:"text-violet-700 dark:text-violet-300 font-medium",children:["Oracle Best-of-",n]}),e.jsxs("p",{className:"text-lg font-bold",children:[(r*100).toFixed(1),"%"]})]})]}),e.jsx("p",{className:"mt-2 text-xs text-gray-500 text-center",children:"Oracle BoN assumes a perfect verifier. Real BoN performance depends on verifier quality."})]})}function G(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Search and verification strategies sample multiple candidate solutions and use a reward model or verifier to select the best one. This decouples generation quality from selection quality, enabling significant accuracy improvements at inference time."}),e.jsxs(h,{title:"Best-of-N with Reward Model",children:[e.jsxs("p",{children:["Generate ",e.jsx(t.InlineMath,{math:"N"})," candidate solutions and select the one with the highest reward model score:"]}),e.jsx(t.BlockMath,{math:"\\hat{a} = \\arg\\max_{a_i} R(q, a_i), \\quad a_i \\sim p_\\theta(\\cdot | q), \\quad i = 1, \\ldots, N"}),e.jsxs("p",{className:"mt-2",children:["With an oracle verifier and base accuracy ",e.jsx(t.InlineMath,{math:"p"}),", the probability that at least one of ",e.jsx(t.InlineMath,{math:"N"})," samples is correct is:"]}),e.jsx(t.BlockMath,{math:"P(\\text{at least one correct}) = 1 - (1-p)^N"})]}),e.jsx(V,{}),e.jsxs(x,{title:"Process Reward Models (PRM) vs Outcome Reward Models (ORM)",children:[e.jsx("p",{children:"Two types of reward models for verifying solutions:"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"ORM:"})," Scores only the final answer. Cheap to train but can be fooled by lucky wrong reasoning."]}),e.jsxs("li",{children:[e.jsx("strong",{children:"PRM:"})," Scores each intermediate step. More robust but requires step-level labels."]}),e.jsx("li",{children:"On MATH: PRM + BoN-1860 achieves 78.2% vs ORM + BoN-1860 at 72.4%"}),e.jsx("li",{children:"PRM enables step-level search (beam search over reasoning steps)"})]})]}),e.jsx(p,{title:"Best-of-N with Process Reward Model",code:`import random
import math

def best_of_n_with_prm(generator, prm, question, N=64):
    """Best-of-N sampling with process reward model scoring.

    Args:
        generator: function(question) -> (steps, final_answer)
        prm: function(question, steps) -> step_scores
        question: input question
        N: number of samples
    """
    candidates = []
    for _ in range(N):
        steps, answer = generator(question)
        step_scores = prm(question, steps)
        # PRM score = product of step correctness probabilities
        # (or min, to catch any bad step)
        total_score = min(step_scores)  # Conservative: worst step
        candidates.append((answer, total_score, steps))

    # Select highest-scoring candidate
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]

# Simulate search scaling
def simulate_bon_scaling(base_acc=0.5, verifier_acc=0.85, max_n=256):
    """Show how accuracy scales with N samples and verifier quality."""
    print(f"Base accuracy: {base_acc*100:.0f}%, Verifier accuracy: {verifier_acc*100:.0f}%")
    print(f"{'N':>6} | {'Oracle BoN':>10} | {'Real BoN':>10} | {'Majority':>10}")
    print("-" * 45)
    for n in [1, 4, 16, 64, 256]:
        if n > max_n:
            break
        oracle = 1 - (1 - base_acc)**n
        # Real BoN: limited by verifier imperfection
        real = oracle * verifier_acc + (1 - oracle) * (1 - verifier_acc) * base_acc
        # Majority: improves with sqrt(N) roughly
        majority = min(0.99, base_acc + 0.15 * math.log2(max(n, 1)))
        print(f"{n:>6} | {oracle*100:>9.1f}% | {real*100:>9.1f}% | {majority*100:>9.1f}%")

simulate_bon_scaling()`}),e.jsx(g,{type:"note",title:"MCTS for LLM Reasoning",children:e.jsx("p",{children:"Monte Carlo Tree Search (MCTS), the technique behind AlphaGo, is being applied to LLM reasoning. Each node in the tree is a partial reasoning chain, and the tree is expanded by generating next steps and scoring them with a value model. This enables exploration of diverse reasoning strategies while pruning unpromising paths early."})})]})}const xe=Object.freeze(Object.defineProperty({__proto__:null,default:G},Symbol.toStringTag,{value:"Module"}));function $(){const[i,o]=c.useState(2),[n,s]=c.useState(7),a=Math.pow(10,i),r=.3+.12*Math.log10(n),l=Math.min(.95,r+.08*Math.log10(a)),d=n*Math.pow(a,.5);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Inference Compute Scaling"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3 flex-wrap",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Model: ",n,"B params",e.jsx("input",{type:"range",min:1,max:70,step:1,value:n,onChange:u=>s(parseInt(u.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Inference compute: ",a,"x",e.jsx("input",{type:"range",min:0,max:4,step:.5,value:i,onChange:u=>o(parseFloat(u.target.value)),className:"w-28 accent-violet-500"})]})]}),e.jsxs("div",{className:"grid grid-cols-3 gap-3 text-sm text-center",children:[e.jsxs("div",{className:"p-2 rounded bg-gray-100 dark:bg-gray-800",children:[e.jsx("p",{className:"text-gray-500 font-medium",children:"Base Performance"}),e.jsxs("p",{className:"font-bold",children:[(r*100).toFixed(1),"%"]})]}),e.jsxs("div",{className:"p-2 rounded bg-violet-50 dark:bg-violet-900/20",children:[e.jsxs("p",{className:"text-violet-700 dark:text-violet-300 font-medium",children:["With ",a,"x Compute"]}),e.jsxs("p",{className:"font-bold",children:[(l*100).toFixed(1),"%"]})]}),e.jsxs("div",{className:"p-2 rounded bg-violet-50 dark:bg-violet-900/20",children:[e.jsx("p",{className:"text-violet-700 dark:text-violet-300 font-medium",children:"Equivalent Dense Model"}),e.jsxs("p",{className:"font-bold",children:["~",d.toFixed(0),"B"]})]})]})]})}function U(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Inference scaling laws describe how model performance improves with additional test-time compute. This creates a new dimension for scaling: instead of (or in addition to) training larger models, spend more compute at inference to achieve better results."}),e.jsxs(h,{title:"Inference Scaling Law",children:[e.jsx("p",{children:"Performance on reasoning tasks follows a power law in inference compute:"}),e.jsx(t.BlockMath,{math:"P(C_{\\text{infer}}) = P_0 + k \\cdot C_{\\text{infer}}^{\\gamma}"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"C_{\\text{infer}}"})," includes tokens generated for reasoning, number of samples, and search compute. Empirically, ",e.jsx(t.InlineMath,{math:"\\gamma \\approx 0.2\\text{-}0.5"})," depending on the task and method. This means 10x more inference compute yields 60-300% relative improvement."]})]}),e.jsx($,{}),e.jsxs(x,{title:"Compute-Equivalent Scaling",children:[e.jsx("p",{children:"A small model with extensive search can match a much larger model:"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsx("li",{children:"7B model with 256 samples + PRM matches 70B model with 1 sample on MATH"}),e.jsx("li",{children:"Cost comparison: 7B x 256 = 1792B FLOPs vs 70B x 1 = 70B FLOPs (7B wins on accuracy per FLOP)"}),e.jsx("li",{children:"Wait — 7B x 256 > 70B? Yes, but the cost is purely sequential generation, highly parallelizable"}),e.jsx("li",{children:"Latency vs throughput: parallel BoN has same latency as single sample"})]})]}),e.jsx(p,{title:"Comparing Training vs Inference Scaling",code:`import math

def training_scaled_performance(params_b, task_baseline=0.3):
    """Performance from training scaling (Chinchilla-like)."""
    return task_baseline + 0.12 * math.log10(params_b)

def inference_scaled_performance(params_b, inference_multiplier, task_baseline=0.3):
    """Performance from inference compute scaling."""
    base = training_scaled_performance(params_b, task_baseline)
    bonus = 0.08 * math.log10(max(inference_multiplier, 1))
    return min(0.95, base + bonus)

def total_cost(params_b, inference_multiplier, num_queries=1000):
    """Total cost in relative FLOPs per query."""
    return params_b * inference_multiplier * num_queries

# Find: what's more cost-effective for 80% accuracy?
target = 0.80
print(f"Target accuracy: {target*100:.0f}%")
print(f"{'Strategy':>35} | {'Accuracy':>8} | {'Cost/query':>10}")
print("-" * 60)

strategies = [
    ("70B, 1x inference", 70, 1),
    ("7B, 64x inference (BoN-64)", 7, 64),
    ("7B, 256x inference (BoN-256)", 7, 256),
    ("405B, 1x inference", 405, 1),
    ("70B, 16x inference (BoN-16)", 70, 16),
]

for name, params, mult in strategies:
    perf = inference_scaled_performance(params, mult)
    cost = params * mult
    print(f"{name:>35} | {perf*100:>7.1f}% | {cost:>8.0f}B")

print("\\nKey insight: small model + search can be more cost-effective than large model")`}),e.jsx(g,{type:"note",title:"The Two Scaling Axes",children:e.jsxs("p",{children:["Deep learning now has two independent scaling axes: ",e.jsx("strong",{children:"training compute"}),"(bigger models, more data) and ",e.jsx("strong",{children:"inference compute"})," (longer reasoning, more samples, search). The optimal allocation between these axes depends on the use case — high-volume, simple tasks favor training scaling, while hard, rare queries favor inference scaling. Models like o1 represent a shift toward the inference axis."]})})]})}const ge=Object.freeze(Object.defineProperty({__proto__:null,default:U},Symbol.toStringTag,{value:"Module"}));function Q(){const[i,o]=c.useState(0),n=[{phase:"Observe",desc:"Encode current observation into latent state z_t",color:"bg-violet-100 dark:bg-violet-900/20"},{phase:"Imagine",desc:"Use learned dynamics model to predict future: z_{t+1} = f(z_t, a_t)",color:"bg-violet-200 dark:bg-violet-900/30"},{phase:"Evaluate",desc:"Predict reward from imagined state: r_{t+1} = R(z_{t+1})",color:"bg-violet-300 dark:bg-violet-900/40"},{phase:"Plan",desc:"Search over action sequences in imagination to maximize expected reward",color:"bg-violet-400 dark:bg-violet-800/40"},{phase:"Act",desc:"Execute best action from planning, observe real outcome, update model",color:"bg-violet-500/20 dark:bg-violet-700/30"}],s=n[i];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"World Model Planning Loop"}),e.jsx("div",{className:"flex gap-1 mb-3",children:n.map((a,r)=>e.jsx("button",{onClick:()=>o(r),className:`flex-1 px-2 py-1 rounded text-xs transition ${i===r?"bg-violet-500 text-white":"bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"}`,children:a.phase},r))}),e.jsxs("div",{className:`p-3 rounded-lg ${s.color} text-sm`,children:[e.jsx("p",{className:"font-medium text-gray-700 dark:text-gray-300",children:s.phase}),e.jsx("p",{className:"text-gray-600 dark:text-gray-400 mt-1",children:s.desc})]}),e.jsx("div",{className:"flex mt-2 gap-1",children:n.map((a,r)=>e.jsx("div",{className:`h-1 flex-1 rounded ${r<=i?"bg-violet-500":"bg-gray-200 dark:bg-gray-700"}`},r))})]})}function X(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:'World models learn a compressed representation of environment dynamics, enabling agents to plan by "imagining" future states. This approach is sample-efficient because the agent can practice in its learned model rather than the real environment.'}),e.jsxs(h,{title:"World Model Components",children:[e.jsx("p",{children:"A world model consists of three learned components operating in latent space:"}),e.jsx(t.BlockMath,{math:"\\text{Encoder: } z_t = q(o_t), \\quad \\text{Dynamics: } z_{t+1} = f(z_t, a_t), \\quad \\text{Reward: } r_t = R(z_t)"}),e.jsx("p",{className:"mt-2",children:"The agent can unroll the dynamics model to simulate trajectories:"}),e.jsx(t.BlockMath,{math:"\\hat{\\tau} = (z_t, a_t, \\hat{z}_{t+1}, \\hat{r}_{t+1}, a_{t+1}, \\hat{z}_{t+2}, \\hat{r}_{t+2}, \\ldots)"}),e.jsx("p",{className:"mt-1",children:"Planning is then optimization over action sequences in this imagined trajectory."})]}),e.jsx(Q,{}),e.jsxs(x,{title:"DreamerV3: Universal World Model",children:[e.jsx("p",{children:"DreamerV3 (Hafner et al., 2023) learns world models that transfer across diverse domains:"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsx("li",{children:"First algorithm to collect diamonds in Minecraft from scratch (no human data)"}),e.jsx("li",{children:"Same hyperparameters work across Atari, DMC, DMLab, and Minecraft"}),e.jsx("li",{children:"Uses a Recurrent State Space Model (RSSM) with discrete latent variables"}),e.jsx("li",{children:"Trains actor and critic entirely in imagination (no real-env policy rollouts)"})]})]}),e.jsx(p,{title:"Simplified World Model with Latent Dynamics",code:`import torch
import torch.nn as nn

class WorldModel(nn.Module):
    """Simplified world model with encoder, dynamics, and reward prediction."""
    def __init__(self, obs_dim=64, latent_dim=128, action_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, latent_dim))
        self.dynamics = nn.GRUCell(action_dim, latent_dim)
        self.reward_head = nn.Linear(latent_dim, 1)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, obs_dim))

    def encode(self, obs):
        return self.encoder(obs)

    def imagine_step(self, z, action):
        z_next = self.dynamics(action, z)
        reward = self.reward_head(z_next)
        return z_next, reward

    def imagine_trajectory(self, z0, actions):
        """Unroll dynamics model for planning."""
        z = z0
        rewards = []
        states = [z]
        for a in actions:
            z, r = self.imagine_step(z, a)
            rewards.append(r)
            states.append(z)
        return torch.stack(states), torch.cat(rewards)

# Planning in imagination
model = WorldModel()
obs = torch.randn(1, 64)
z = model.encode(obs)

# Plan 10 steps ahead
actions = [torch.randn(1, 4) for _ in range(10)]
imagined_states, imagined_rewards = model.imagine_trajectory(z, actions)
print(f"Imagined {len(actions)} steps: states {imagined_states.shape}")
print(f"Expected return: {imagined_rewards.sum().item():.3f}")`}),e.jsx(g,{type:"note",title:"Video Generation as World Models",children:e.jsx("p",{children:"Large video generation models (Sora, Genie) can be viewed as world models that learn physics and dynamics from internet video. They predict future visual frames conditioned on actions or text, potentially enabling robots and game agents to learn from vast video data without environment interaction. The boundary between generative models and world models is rapidly blurring."})})]})}const ue=Object.freeze(Object.defineProperty({__proto__:null,default:X},Symbol.toStringTag,{value:"Module"}));function Y(){const[i,o]=c.useState("foundation"),n={classical:{name:"Classical (2015-2019)",approach:"Task-specific RL policies trained in simulation",transfer:"Sim-to-real gap is a major challenge",examples:"OpenAI hand manipulation, locomotion policies"},language:{name:"Language-Guided (2020-2023)",approach:"Language models as planners for robot actions",transfer:"Natural language bridges sim and real domains",examples:"SayCan, Code-as-Policies, RT-2"},foundation:{name:"Foundation Models (2023+)",approach:"Pretrained on diverse robot data, fine-tune for specific tasks",transfer:"Cross-embodiment generalization",examples:"RT-X, Octo, pi0"}},s=n[i];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Embodied AI Evolution"}),e.jsx("div",{className:"flex gap-2 mb-3 flex-wrap",children:Object.entries(n).map(([a,r])=>e.jsx("button",{onClick:()=>o(a),className:`px-3 py-1 rounded-lg text-sm transition ${i===a?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:r.name},a))}),e.jsxs("div",{className:"p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20 text-sm space-y-1",children:[e.jsxs("p",{children:[e.jsx("strong",{children:"Approach:"})," ",s.approach]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Transfer:"})," ",s.transfer]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Examples:"})," ",s.examples]})]})]})}function J(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Foundation models are transforming robotics by providing robots with broad knowledge about the physical world, natural language understanding, and general-purpose reasoning. The goal is a single model that can control diverse robots across diverse tasks."}),e.jsxs(h,{title:"Vision-Language-Action (VLA) Models",children:[e.jsx("p",{children:"VLA models extend multimodal LLMs to output robot actions, creating an end-to-end policy:"}),e.jsx(t.BlockMath,{math:"\\pi(a_t | o_t, l) = \\text{VLA}(\\text{image}_t, \\text{language\\_instruction}, \\text{robot\\_state}_t)"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"a_t"})," is the robot action (e.g., end-effector pose), ",e.jsx(t.InlineMath,{math:"o_t"})," is the visual observation, and ",e.jsx(t.InlineMath,{math:"l"})," is the task instruction. The model is typically a pretrained VLM fine-tuned on robot demonstrations with action tokens added to the vocabulary."]})]}),e.jsx(Y,{}),e.jsxs(x,{title:"RT-2: Vision-Language-Action Model",children:[e.jsx("p",{children:"RT-2 (Google DeepMind) fine-tunes a 55B VLM (PaLI-X) on robot data:"}),e.jsxs("ul",{className:"list-disc list-inside mt-2 space-y-1",children:[e.jsx("li",{children:'Actions represented as text tokens: "1 128 91 241 1 128 147" (7-DOF)'}),e.jsx("li",{children:'Inherits reasoning from VLM pretraining (can handle "pick up the extinct animal")'}),e.jsx("li",{children:"3x improvement on unseen objects compared to RT-1 (task-specific model)"}),e.jsx("li",{children:"Emergent capabilities: multi-step reasoning about objects, spatial relationships"})]})]}),e.jsx(p,{title:"Vision-Language-Action Policy (Simplified)",code:`import torch
import torch.nn as nn

class SimpleVLAPolicy(nn.Module):
    """Simplified VLA policy: image + language -> robot action."""
    def __init__(self, vision_dim=1024, lang_dim=768, action_dim=7):
        super().__init__()
        # Pretrained encoders (frozen in practice)
        self.vision_proj = nn.Linear(vision_dim, 512)
        self.lang_proj = nn.Linear(lang_dim, 512)
        # Action prediction head (trained on robot data)
        self.action_head = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, action_dim),  # 7-DOF: xyz + rotation + gripper
            nn.Tanh(),  # Normalize actions to [-1, 1]
        )

    def forward(self, image_features, language_features):
        v = self.vision_proj(image_features)
        l = self.lang_proj(language_features)
        combined = torch.cat([v, l], dim=-1)
        return self.action_head(combined)

def collect_demonstration(env, policy, instruction):
    """Collect expert demonstration for VLA training."""
    obs = env.reset()
    trajectory = []
    for step in range(100):
        action = policy(obs, instruction)  # Expert or teleoperation
        next_obs, reward, done = env.step(action)
        trajectory.append({
            "image": obs["image"],
            "instruction": instruction,
            "action": action,
        })
        obs = next_obs
        if done:
            break
    return trajectory

# Simulate
vla = SimpleVLAPolicy()
image_feat = torch.randn(1, 1024)   # From ViT
lang_feat = torch.randn(1, 768)     # From language model
action = vla(image_feat, lang_feat)
print(f"Predicted action (7-DOF): {action.squeeze().tolist()[:4]}...")
print(f"Action dim meanings: [dx, dy, dz, rx, ry, rz, gripper]")`}),e.jsx(g,{type:"note",title:"The Data Challenge in Robotics",children:e.jsx("p",{children:"Robot data is orders of magnitude scarcer than internet text/images. The Open X-Embodiment dataset (RT-X) aggregates data from 22 robots across 21 institutions — still only ~1M trajectories vs trillions of text tokens. Simulation, data augmentation, and cross-embodiment transfer learning are critical for bridging this data gap."})})]})}const fe=Object.freeze(Object.defineProperty({__proto__:null,default:J},Symbol.toStringTag,{value:"Module"}));function Z(){const[i,o]=c.useState("alignment"),n={alignment:{name:"AI Safety & Alignment",desc:"Ensuring AI systems behave according to human values and intentions, especially as capabilities increase. Includes reward hacking, goal misgeneralization, deceptive alignment, and scalable oversight.",difficulty:"Critical",timeframe:"Urgent (needed before AGI)"},generalization:{name:"Robust Generalization",desc:"Current models fail on distributional shift, adversarial inputs, and novel combinations of known concepts. True out-of-distribution generalization remains elusive.",difficulty:"Fundamental",timeframe:"Decades-long research program"},reasoning:{name:"Genuine Reasoning",desc:"Do LLMs truly reason or pattern-match? Can they handle truly novel problems that require logical deduction rather than retrieval of similar training examples?",difficulty:"Open debate",timeframe:"Active research area"},efficiency:{name:"Sample Efficiency",desc:"Humans learn from far fewer examples than neural networks. Achieving human-level sample efficiency would transform the field and reduce compute requirements.",difficulty:"Hard",timeframe:"5-15 years"},understanding:{name:"Understanding DNNs",desc:"Why do overparameterized networks generalize? What determines the inductive biases of different architectures? Can we formally characterize what networks learn?",difficulty:"Theoretical",timeframe:"Ongoing fundamental research"}},s=n[i];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Open Problems in Deep Learning"}),e.jsx("div",{className:"flex gap-1 mb-3 flex-wrap",children:Object.entries(n).map(([a,r])=>e.jsx("button",{onClick:()=>o(a),className:`px-2 py-1 rounded-lg text-xs transition ${i===a?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:r.name},a))}),e.jsxs("div",{className:"p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20 text-sm space-y-1",children:[e.jsx("p",{className:"text-gray-600 dark:text-gray-400",children:s.desc}),e.jsxs("div",{className:"flex gap-4 mt-2 text-xs text-gray-500",children:[e.jsxs("span",{children:["Difficulty: ",e.jsx("strong",{children:s.difficulty})]}),e.jsxs("span",{children:["Timeframe: ",e.jsx("strong",{children:s.timeframe})]})]})]})]})}function ee(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Despite remarkable progress, deep learning faces fundamental unsolved challenges in safety, generalization, reasoning, and theoretical understanding. These open problems define the research frontier and will shape the future of the field."}),e.jsxs(h,{title:"The Alignment Problem",children:[e.jsx("p",{children:"As AI systems become more capable, ensuring they act according to human intentions becomes critical. The alignment problem has several formal aspects:"}),e.jsx(t.BlockMath,{math:"\\text{Outer alignment: } R_{\\text{specified}} \\approx R_{\\text{intended}}"}),e.jsx(t.BlockMath,{math:"\\text{Inner alignment: } R_{\\text{learned}} \\approx R_{\\text{specified}}"}),e.jsx("p",{className:"mt-2",children:"Outer alignment asks whether we can correctly specify what we want. Inner alignment asks whether the trained model actually optimizes for the specified objective, even in novel situations."})]}),e.jsx(Z,{}),e.jsx(v,{title:"Scaling Alone May Not Suffice",children:e.jsx("p",{children:"Several fundamental problems are unlikely to be solved by scaling alone: adversarial robustness (adversarial examples persist at all scales), formal reasoning (LLMs still fail at novel logic puzzles), and alignment (larger models may be harder to align). New paradigms, architectures, or training methods may be needed."})}),e.jsx(x,{title:"Key Research Directions",children:e.jsxs("ul",{className:"list-disc list-inside space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Mechanistic interpretability:"})," Understanding what networks compute, enabling debugging and alignment"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Constitutional AI / RLHF:"})," Scalable techniques for aligning model behavior with human values"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Neurosymbolic methods:"})," Combining neural networks with formal logic for reliable reasoning"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Continual learning:"})," Models that learn from new data without catastrophic forgetting"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Energy efficiency:"})," Neuromorphic computing, spiking networks, and analog hardware"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Multimodal grounding:"})," Connecting language to real-world experience and embodiment"]})]})}),e.jsx(p,{title:"Measuring Alignment: Reward Hacking Detection",code:`import torch

def detect_reward_hacking(proxy_rewards, true_rewards, threshold=0.5):
    """Detect when optimizing a proxy reward diverges from true reward.

    Goodhart's Law: "When a measure becomes a target,
    it ceases to be a good measure."

    Args:
        proxy_rewards: rewards from the learned reward model
        true_rewards: rewards from human evaluation (expensive)
        threshold: correlation threshold for alarm
    """
    # Compute correlation between proxy and true rewards
    proxy = torch.tensor(proxy_rewards, dtype=torch.float)
    true = torch.tensor(true_rewards, dtype=torch.float)

    correlation = torch.corrcoef(torch.stack([proxy, true]))[0, 1]

    # Check for Goodhart's Law violation
    # High proxy reward but low true reward = reward hacking
    mean_proxy = proxy.mean().item()
    mean_true = true.mean().item()

    print(f"Proxy-True correlation: {correlation:.3f}")
    print(f"Mean proxy reward: {mean_proxy:.3f}")
    print(f"Mean true reward: {mean_true:.3f}")

    if correlation < threshold:
        print("WARNING: Low correlation — possible reward hacking!")
        print("The model may be exploiting proxy reward without achieving true goal.")
    else:
        print("Rewards appear aligned (but vigilance is still needed).")

# Simulate: model finds exploit in proxy reward
proxy_rewards = [0.9, 0.85, 0.92, 0.95, 0.88, 0.91]  # looks good
true_rewards =  [0.7, 0.3, 0.4, 0.2, 0.5, 0.3]        # actually bad
detect_reward_hacking(proxy_rewards, true_rewards)`}),e.jsx(g,{type:"note",title:"The Road Ahead",children:e.jsx("p",{children:'Deep learning has achieved extraordinary results, but significant challenges remain. The field is at an inflection point — moving from "can we scale?" to "can we build safe, reliable, and efficient AI systems?" Progress on these open problems will determine whether AI fulfills its transformative potential responsibly.'})})]})}const ye=Object.freeze(Object.defineProperty({__proto__:null,default:ee},Symbol.toStringTag,{value:"Module"}));export{se as a,re as b,oe as c,le as d,ce as e,de as f,me as g,he as h,pe as i,xe as j,ge as k,ue as l,fe as m,ye as n,ie as s};

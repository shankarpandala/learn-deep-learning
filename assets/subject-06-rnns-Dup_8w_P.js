import{j as e,r as m}from"./vendor-DpISuAX6.js";import{r as t}from"./vendor-katex-CbWCYdth.js";import{D as h,E as x,P as p,N as f,T as y,W as j}from"./subject-01-foundations-D0A1VJsr.js";function b(){const[a,c]=m.useState(0);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsxs("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:["Unrolled RNN (",a+1," time steps)"]}),e.jsx("div",{className:"flex items-center gap-4 mb-3",children:e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Steps",e.jsx("input",{type:"range",min:0,max:4-1,step:1,value:a,onChange:i=>c(parseInt(i.target.value)),className:"w-32 accent-violet-500"}),a+1]})}),e.jsxs("svg",{width:460,height:180,className:"mx-auto block",children:[Array.from({length:a+1},(i,s)=>{const n=60+s*100,o=60,g=150,_=20;return e.jsxs("g",{children:[e.jsx("rect",{x:n-25,y:o-20,width:50,height:40,rx:6,fill:"#8b5cf6",opacity:.85}),e.jsxs("text",{x:n,y:o+5,textAnchor:"middle",fill:"white",fontSize:12,fontWeight:"bold",children:["h_",s]}),e.jsxs("text",{x:n,y:g,textAnchor:"middle",fill:"#6b7280",fontSize:11,children:["x_",s]}),e.jsx("line",{x1:n,y1:g-10,x2:n,y2:o+20,stroke:"#8b5cf6",strokeWidth:1.5,markerEnd:"url(#arr)"}),e.jsxs("text",{x:n,y:_,textAnchor:"middle",fill:"#6b7280",fontSize:11,children:["y_",s]}),e.jsx("line",{x1:n,y1:o-20,x2:n,y2:_+6,stroke:"#8b5cf6",strokeWidth:1.5,markerEnd:"url(#arr)"}),s<a&&e.jsx("line",{x1:n+25,y1:o,x2:n+75,y2:o,stroke:"#a78bfa",strokeWidth:2,markerEnd:"url(#arr)"})]},s)}),e.jsx("defs",{children:e.jsx("marker",{id:"arr",markerWidth:8,markerHeight:6,refX:8,refY:3,orient:"auto",children:e.jsx("path",{d:"M0,0 L8,3 L0,6 Z",fill:"#8b5cf6"})})})]})]})}function v(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Recurrent Neural Networks process sequential data by maintaining a hidden state that captures information from previous time steps, making them naturally suited for tasks involving temporal or ordered data such as text, speech, and time series."}),e.jsxs(h,{title:"Recurrent Neural Network",children:[e.jsxs("p",{children:["An RNN computes a hidden state ",e.jsx(t.InlineMath,{math:"h_t"})," at each time step ",e.jsx(t.InlineMath,{math:"t"})," via:"]}),e.jsx(t.BlockMath,{math:"h_t = \\tanh(W_{hh}\\, h_{t-1} + W_{xh}\\, x_t + b_h)"}),e.jsx(t.BlockMath,{math:"y_t = W_{hy}\\, h_t + b_y"}),e.jsxs("p",{className:"mt-2",children:["The same weight matrices ",e.jsx(t.InlineMath,{math:"W_{hh}, W_{xh}, W_{hy}"})," are shared across all time steps, giving RNNs a fixed parameter count regardless of sequence length."]})]}),e.jsx(b,{}),e.jsxs(x,{title:"Hidden State Dimensions",children:[e.jsxs("p",{children:["For input dimension ",e.jsx(t.InlineMath,{math:"d = 50"})," and hidden size ",e.jsx(t.InlineMath,{math:"h = 128"}),":"]}),e.jsx(t.BlockMath,{math:"W_{xh} \\in \\mathbb{R}^{128 \\times 50},\\quad W_{hh} \\in \\mathbb{R}^{128 \\times 128},\\quad W_{hy} \\in \\mathbb{R}^{|V| \\times 128}"}),e.jsxs("p",{children:["Total recurrent parameters: ",e.jsx(t.InlineMath,{math:"128 \\times 50 + 128 \\times 128 = 22{,}784"}),"."]})]}),e.jsx(p,{title:"Vanilla RNN in PyTorch",code:`import torch
import torch.nn as nn

# Single-layer RNN: input_size=50, hidden_size=128
rnn = nn.RNN(input_size=50, hidden_size=128, batch_first=True)

# Batch of 8 sequences, each length 20, feature dim 50
x = torch.randn(8, 20, 50)
h0 = torch.zeros(1, 8, 128)  # initial hidden state

output, h_n = rnn(x, h0)
print(f"Output shape: {output.shape}")   # (8, 20, 128)
print(f"Final hidden: {h_n.shape}")      # (1, 8, 128)

# Manual single step
W_xh = rnn.weight_ih_l0  # (4*128, 50) for RNN it's (128, 50)
W_hh = rnn.weight_hh_l0  # (128, 128)
print(f"Params: {sum(p.numel() for p in rnn.parameters()):,}")`}),e.jsx(f,{type:"note",title:"Weight Sharing is Key",children:e.jsxs("p",{children:["Unlike feedforward networks that have separate parameters per layer, an RNN",e.jsx("strong",{children:" reuses the same weights"})," at every time step. This weight sharing acts as an inductive bias for temporal invariance and keeps the model compact, but it also makes training via backpropagation through time challenging due to vanishing or exploding gradients."]})}),e.jsx(h,{title:"Hidden State as Memory",children:e.jsxs("p",{children:["The hidden state ",e.jsx(t.InlineMath,{math:"h_t"})," is a compressed representation of the entire input history ",e.jsx(t.InlineMath,{math:"(x_1, x_2, \\ldots, x_t)"}),". In practice, vanilla RNNs struggle to remember information beyond roughly 10-20 time steps due to the vanishing gradient problem, motivating architectures like LSTM and GRU."]})}),e.jsx(x,{title:"Common RNN Patterns",children:e.jsxs("p",{children:["RNNs can be configured in several input-output patterns:",e.jsx("strong",{children:" One-to-many"})," (image captioning),",e.jsx("strong",{children:" many-to-one"})," (sentiment classification),",e.jsx("strong",{children:" many-to-many"})," (machine translation, language modeling). The same core recurrence equation applies in all cases; only the input feeding and output tapping differ."]})})]})}const te=Object.freeze(Object.defineProperty({__proto__:null,default:v},Symbol.toStringTag,{value:"Module"}));function k(){const[a,c]=m.useState(5);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Gradient Flow Through Time"}),e.jsx("div",{className:"flex items-center gap-4 mb-3",children:e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Steps: ",a,e.jsx("input",{type:"range",min:2,max:8,step:1,value:a,onChange:r=>c(parseInt(r.target.value)),className:"w-32 accent-violet-500"})]})}),e.jsxs("svg",{width:440,height:100,className:"mx-auto block",children:[Array.from({length:a},(r,i)=>{const s=40+i*(360/(a-1||1)),n=Math.pow(.7,a-1-i);return e.jsxs("g",{children:[e.jsx("circle",{cx:s,cy:50,r:18,fill:"#8b5cf6",opacity:Math.max(n,.15)}),e.jsxs("text",{x:s,y:55,textAnchor:"middle",fill:"white",fontSize:10,fontWeight:"bold",children:["t=",i]}),i<a-1&&e.jsx("line",{x1:s+18,y1:50,x2:s+360/(a-1)-18,y2:50,stroke:"#a78bfa",strokeWidth:2,strokeDasharray:"4,3",markerEnd:"url(#garr)"})]},i)}),e.jsxs("text",{x:220,y:95,textAnchor:"middle",fill:"#6b7280",fontSize:10,children:["Gradient magnitude decays ~0.7^",a-1," = ",Math.pow(.7,a-1).toFixed(4)]}),e.jsx("defs",{children:e.jsx("marker",{id:"garr",markerWidth:7,markerHeight:5,refX:7,refY:2.5,orient:"auto",children:e.jsx("path",{d:"M0,0 L7,2.5 L0,5 Z",fill:"#a78bfa"})})})]})]})}function N(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Backpropagation Through Time (BPTT) is the standard algorithm for computing gradients in RNNs. It unrolls the recurrence across time and applies the chain rule, but this introduces unique challenges around vanishing and exploding gradients."}),e.jsxs(h,{title:"Backpropagation Through Time",children:[e.jsxs("p",{children:["The total loss over a sequence of length ",e.jsx(t.InlineMath,{math:"T"})," is:"]}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = \\sum_{t=1}^{T} \\ell_t(y_t, \\hat{y}_t)"}),e.jsxs("p",{className:"mt-2",children:["The gradient of ",e.jsx(t.InlineMath,{math:"\\mathcal{L}"})," w.r.t. ",e.jsx(t.InlineMath,{math:"W_{hh}"})," requires summing contributions from each step:"]}),e.jsx(t.BlockMath,{math:"\\frac{\\partial \\mathcal{L}}{\\partial W_{hh}} = \\sum_{t=1}^{T} \\sum_{k=1}^{t} \\frac{\\partial \\ell_t}{\\partial h_t} \\left(\\prod_{j=k+1}^{t} \\frac{\\partial h_j}{\\partial h_{j-1}}\\right) \\frac{\\partial h_k}{\\partial W_{hh}}"})]}),e.jsx(k,{}),e.jsxs(y,{title:"Vanishing / Exploding Gradients",id:"bptt-gradient-bound",children:[e.jsx("p",{children:"The Jacobian product satisfies:"}),e.jsx(t.BlockMath,{math:"\\left\\|\\prod_{j=k+1}^{t} \\frac{\\partial h_j}{\\partial h_{j-1}}\\right\\| \\leq \\|W_{hh}\\|^{t-k} \\cdot \\gamma^{t-k}"}),e.jsxs("p",{children:["where ",e.jsx(t.InlineMath,{math:"\\gamma = \\max|\\tanh'(z)| \\le 1"}),". If the spectral radius of ",e.jsx(t.InlineMath,{math:"W_{hh}"})," is less than 1, gradients vanish exponentially; if greater than 1, they explode."]})]}),e.jsxs(h,{title:"Truncated BPTT",children:[e.jsxs("p",{children:["Instead of backpropagating through the full sequence, truncated BPTT limits the backward pass to ",e.jsx(t.InlineMath,{math:"k"})," steps:"]}),e.jsx(t.BlockMath,{math:"\\frac{\\partial \\mathcal{L}}{\\partial W_{hh}} \\approx \\sum_{t=1}^{T} \\sum_{j=\\max(1, t-k)}^{t} \\frac{\\partial \\ell_t}{\\partial h_t} \\left(\\prod_{i=j+1}^{t} \\frac{\\partial h_i}{\\partial h_{i-1}}\\right) \\frac{\\partial h_j}{\\partial W_{hh}}"}),e.jsx("p",{className:"mt-2",children:"This trades off long-range dependency modeling for computational efficiency and gradient stability."})]}),e.jsx(p,{title:"BPTT with Truncation in PyTorch",code:`import torch
import torch.nn as nn

rnn = nn.RNN(input_size=32, hidden_size=64, batch_first=True)
linear = nn.Linear(64, 10)
loss_fn = nn.CrossEntropyLoss()

x = torch.randn(4, 100, 32)        # long sequence
targets = torch.randint(0, 10, (4, 100))

# Truncated BPTT with k=20 steps
k = 20
h = torch.zeros(1, 4, 64)
optimizer = torch.optim.Adam(list(rnn.parameters()) + list(linear.parameters()), lr=1e-3)

for start in range(0, 100, k):
    chunk = x[:, start:start+k, :]
    h = h.detach()  # stop gradient flow beyond truncation window
    out, h = rnn(chunk, h)
    logits = linear(out.reshape(-1, 64))
    loss = loss_fn(logits, targets[:, start:start+k].reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Step {start}-{start+k}: loss={loss.item():.4f}")`}),e.jsx(j,{title:"Exploding Gradients",children:e.jsxs("p",{children:["When gradients explode, a single update can catastrophically change all parameters. Always monitor gradient norms during RNN training and apply ",e.jsx("strong",{children:"gradient clipping"})," as a safeguard. Truncated BPTT alone does not prevent exploding gradients."]})}),e.jsx(f,{type:"note",title:"BPTT Computational Cost",children:e.jsxs("p",{children:["Full BPTT requires ",e.jsx(t.InlineMath,{math:"O(T)"})," memory to store all intermediate hidden states. Truncated BPTT with window ",e.jsx(t.InlineMath,{math:"k"})," reduces this to ",e.jsx(t.InlineMath,{math:"O(k)"}),", making it practical for very long sequences. Modern frameworks like PyTorch handle the unrolling and gradient computation automatically."]})})]})}const ae=Object.freeze(Object.defineProperty({__proto__:null,default:N},Symbol.toStringTag,{value:"Module"}));function w(){const[a,c]=m.useState("many-to-one"),l=400,d=140,r={"many-to-one":{inputs:[0,1,2,3],outputs:[3],label:"Sequence Classification"},"one-to-many":{inputs:[0],outputs:[0,1,2,3],label:"Sequence Generation"},"many-to-many":{inputs:[0,1,2,3],outputs:[0,1,2,3],label:"Language Modeling"}},i=r[a];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:i.label}),e.jsx("div",{className:"flex items-center gap-3 mb-3 flex-wrap",children:Object.keys(r).map(s=>e.jsx("button",{onClick:()=>c(s),className:`px-3 py-1 rounded-lg text-sm ${a===s?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:r[s].label},s))}),e.jsx("svg",{width:l,height:d,className:"mx-auto block",children:[0,1,2,3].map(s=>{const n=60+s*90;return e.jsxs("g",{children:[e.jsx("rect",{x:n-22,y:45,width:44,height:34,rx:5,fill:"#8b5cf6",opacity:.8}),e.jsxs("text",{x:n,y:67,textAnchor:"middle",fill:"white",fontSize:11,fontWeight:"bold",children:["h",s]}),i.inputs.includes(s)&&e.jsxs(e.Fragment,{children:[e.jsxs("text",{x:n,y:130,textAnchor:"middle",fill:"#6b7280",fontSize:10,children:["x",s]}),e.jsx("line",{x1:n,y1:118,x2:n,y2:79,stroke:"#a78bfa",strokeWidth:1.5})]}),i.outputs.includes(s)&&e.jsxs(e.Fragment,{children:[e.jsxs("text",{x:n,y:22,textAnchor:"middle",fill:"#6b7280",fontSize:10,children:["y",s]}),e.jsx("line",{x1:n,y1:45,x2:n,y2:28,stroke:"#a78bfa",strokeWidth:1.5})]}),s<3&&e.jsx("line",{x1:n+22,y1:62,x2:n+68,y2:62,stroke:"#c4b5fd",strokeWidth:1.5})]},s)})})]})}function T(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"RNNs support a flexible set of input-output configurations, making them applicable to a wide range of sequence tasks including classification, generation, and language modeling."}),e.jsxs(h,{title:"Sequence Classification (Many-to-One)",children:[e.jsx("p",{children:"The RNN processes an entire input sequence and produces a single output from the final hidden state:"}),e.jsx(t.BlockMath,{math:"\\hat{y} = \\text{softmax}(W_y \\, h_T + b_y)"}),e.jsx("p",{className:"mt-2",children:"Common applications: sentiment analysis, spam detection, document classification."})]}),e.jsx(w,{}),e.jsxs(h,{title:"Language Modeling (Many-to-Many)",children:[e.jsx("p",{children:"At each time step the model predicts the next token given all previous tokens:"}),e.jsx(t.BlockMath,{math:"P(x_{t+1} | x_1, \\ldots, x_t) = \\text{softmax}(W_y \\, h_t)"}),e.jsx("p",{className:"mt-2",children:"The training objective is to minimize the cross-entropy loss, which is equivalent to maximizing the log-likelihood of the data."}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = -\\frac{1}{T}\\sum_{t=1}^{T} \\log P(x_{t+1} | x_{\\le t})"})]}),e.jsxs(x,{title:"Perplexity",children:[e.jsx("p",{children:"Language model quality is measured by perplexity:"}),e.jsx(t.BlockMath,{math:"\\text{PPL} = \\exp\\!\\left(-\\frac{1}{T}\\sum_{t=1}^{T}\\log P(x_{t+1}|x_{\\le t})\\right)"}),e.jsx("p",{children:"A perplexity of 100 means the model is as uncertain as choosing uniformly among 100 tokens."})]}),e.jsx(p,{title:"Character-Level Language Model",code:`import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        e = self.embed(x)
        out, h = self.rnn(e, h)
        logits = self.head(out)
        return logits, h

# Example: vocab of 26 letters + space
model = CharRNN(vocab_size=27)
x = torch.randint(0, 27, (4, 50))  # batch=4, seq_len=50
logits, h = model(x)
print(f"Logits: {logits.shape}")  # (4, 50, 27)

# Greedy generation
def generate(model, seed, length=100):
    model.eval()
    tokens = [seed]
    h = None
    for _ in range(length):
        x = torch.tensor([[tokens[-1]]])
        logits, h = model(x, h)
        tokens.append(logits[0, -1].argmax().item())
    return tokens`}),e.jsx(f,{type:"note",title:"Sequence Generation (One-to-Many)",children:e.jsxs("p",{children:["In generation tasks like image captioning, a single input (e.g., a CNN feature vector) is fed as the initial hidden state, and the RNN autoregressively produces output tokens.",e.jsx("strong",{children:" Teacher forcing"})," is commonly used during training: the ground-truth token at step ",e.jsx(t.InlineMath,{math:"t"})," is fed as input at step ",e.jsx(t.InlineMath,{math:"t+1"}),", rather than the model's own prediction."]})}),e.jsx(x,{title:"Practical Tip: Sampling Strategies",children:e.jsxs("p",{children:["During generation, greedy decoding always picks the most likely token. Alternatives include",e.jsx("strong",{children:" temperature scaling"})," (",e.jsx(t.InlineMath,{math:"p_i \\propto \\exp(\\text{logit}_i / \\tau)"}),") and",e.jsx("strong",{children:" top-k sampling"}),", which truncates the distribution to the ",e.jsx(t.InlineMath,{math:"k"})," most likely tokens before sampling."]})})]})}const se=Object.freeze(Object.defineProperty({__proto__:null,default:T},Symbol.toStringTag,{value:"Module"}));function M(){const[a,c]=m.useState(.8),[l,d]=m.useState(.5),[r,i]=m.useState(.6),[s,n]=m.useState(1),o=a*s+l*r,_=.7*Math.tanh(o);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Interactive LSTM Gate Demo"}),e.jsxs("div",{className:"grid grid-cols-2 gap-3 mb-4",children:[e.jsxs("label",{className:"flex flex-col gap-1 text-sm text-gray-600 dark:text-gray-400",children:["Forget gate (f): ",a.toFixed(2),e.jsx("input",{type:"range",min:0,max:1,step:.01,value:a,onChange:u=>c(parseFloat(u.target.value)),className:"accent-violet-500"})]}),e.jsxs("label",{className:"flex flex-col gap-1 text-sm text-gray-600 dark:text-gray-400",children:["Input gate (i): ",l.toFixed(2),e.jsx("input",{type:"range",min:0,max:1,step:.01,value:l,onChange:u=>d(parseFloat(u.target.value)),className:"accent-violet-500"})]}),e.jsxs("label",{className:"flex flex-col gap-1 text-sm text-gray-600 dark:text-gray-400",children:["Candidate (g): ",r.toFixed(2),e.jsx("input",{type:"range",min:-1,max:1,step:.01,value:r,onChange:u=>i(parseFloat(u.target.value)),className:"accent-violet-500"})]}),e.jsxs("label",{className:"flex flex-col gap-1 text-sm text-gray-600 dark:text-gray-400",children:["Previous cell (c_prev): ",s.toFixed(2),e.jsx("input",{type:"range",min:-2,max:2,step:.01,value:s,onChange:u=>n(parseFloat(u.target.value)),className:"accent-violet-500"})]})]}),e.jsxs("div",{className:"flex gap-6 justify-center text-sm",children:[e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/30 px-4 py-2 text-center",children:[e.jsx("div",{className:"text-violet-700 dark:text-violet-300 font-semibold",children:"New Cell"}),e.jsx("div",{className:"text-lg font-mono",children:o.toFixed(4)})]}),e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/30 px-4 py-2 text-center",children:[e.jsx("div",{className:"text-violet-700 dark:text-violet-300 font-semibold",children:"Hidden"}),e.jsx("div",{className:"text-lg font-mono",children:_.toFixed(4)})]})]})]})}function S(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Long Short-Term Memory networks address the vanishing gradient problem by introducing a gated cell state that can maintain information over long time spans. The gating mechanism learns when to store, update, and output information."}),e.jsxs(h,{title:"LSTM Equations",children:[e.jsxs("p",{children:["Given input ",e.jsx(t.InlineMath,{math:"x_t"})," and previous states ",e.jsx(t.InlineMath,{math:"h_{t-1}, c_{t-1}"}),":"]}),e.jsx(t.BlockMath,{math:"f_t = \\sigma(W_f [h_{t-1}, x_t] + b_f) \\quad \\text{(forget gate)}"}),e.jsx(t.BlockMath,{math:"i_t = \\sigma(W_i [h_{t-1}, x_t] + b_i) \\quad \\text{(input gate)}"}),e.jsx(t.BlockMath,{math:"\\tilde{c}_t = \\tanh(W_c [h_{t-1}, x_t] + b_c) \\quad \\text{(candidate)}"}),e.jsx(t.BlockMath,{math:"c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t \\quad \\text{(cell update)}"}),e.jsx(t.BlockMath,{math:"o_t = \\sigma(W_o [h_{t-1}, x_t] + b_o) \\quad \\text{(output gate)}"}),e.jsx(t.BlockMath,{math:"h_t = o_t \\odot \\tanh(c_t)"})]}),e.jsx(M,{}),e.jsx(x,{title:"Cell State as a Highway",children:e.jsxs("p",{children:["The cell state update ",e.jsx(t.InlineMath,{math:"c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t"})," acts like a highway: when ",e.jsx(t.InlineMath,{math:"f_t \\approx 1"})," and ",e.jsx(t.InlineMath,{math:"i_t \\approx 0"}),", the cell state passes through unchanged, allowing gradients to flow across many time steps without decay."]})}),e.jsx(p,{title:"LSTM in PyTorch",code:`import torch
import torch.nn as nn

lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)

x = torch.randn(8, 30, 64)  # batch=8, seq=30, features=64
h0 = torch.zeros(2, 8, 128)  # 2 layers
c0 = torch.zeros(2, 8, 128)

output, (h_n, c_n) = lstm(x, (h0, c0))
print(f"Output: {output.shape}")       # (8, 30, 128)
print(f"Hidden: {h_n.shape}")          # (2, 8, 128)
print(f"Cell:   {c_n.shape}")          # (2, 8, 128)
print(f"Params: {sum(p.numel() for p in lstm.parameters()):,}")
# LSTM has 4x parameters of vanilla RNN (4 gate weight matrices)`}),e.jsx(f,{type:"note",title:"Why 4x Parameters?",children:e.jsxs("p",{children:["An LSTM with hidden size ",e.jsx(t.InlineMath,{math:"h"})," and input size ",e.jsx(t.InlineMath,{math:"d"})," has four gate matrices, each of size ",e.jsx(t.InlineMath,{math:"(h+d) \\times h"}),", giving total recurrent parameters ",e.jsx(t.InlineMath,{math:"4 \\cdot h \\cdot (h + d) + 4h"})," (including biases). This is exactly 4 times a vanilla RNN of the same hidden size."]})}),e.jsxs(x,{title:"Gate Values in Practice",children:[e.jsx("p",{children:"During training on language modeling tasks, forget gates typically learn values close to 1 (remembering most information), while input and output gates show more variation. The forget gate bias is commonly initialized to 1.0 (Gers et al., 2000) to encourage information flow early in training."}),e.jsx(t.BlockMath,{math:"f_t \\approx 0.9,\\quad i_t \\in [0.1, 0.8],\\quad o_t \\in [0.3, 0.9]"})]})]})}const ne=Object.freeze(Object.defineProperty({__proto__:null,default:S},Symbol.toStringTag,{value:"Module"}));function z(){const[a,c]=m.useState("standard"),l={standard:{title:"Standard LSTM",gates:3,params:"4h(h+d)",note:"Separate forget, input, output gates with independent candidate."},peephole:{title:"Peephole LSTM",gates:3,params:"4h(h+d)+3h^2",note:"Gates peek at the cell state directly for more informed gating."},coupled:{title:"Coupled Gate LSTM",gates:2,params:"3h(h+d)",note:"Input gate = 1 - forget gate. Fewer parameters, enforces trade-off."},gru:{title:"GRU",gates:2,params:"3h(h+d)",note:"No separate cell state. Merges cell and hidden state into one."}},d=l[a];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"LSTM Variant Comparison"}),e.jsx("div",{className:"flex items-center gap-2 mb-4 flex-wrap",children:Object.keys(l).map(r=>e.jsx("button",{onClick:()=>c(r),className:`px-3 py-1 rounded-lg text-sm ${a===r?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:l[r].title},r))}),e.jsxs("div",{className:"grid grid-cols-3 gap-4 text-center text-sm",children:[e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/30 px-3 py-2",children:[e.jsx("div",{className:"text-violet-700 dark:text-violet-300 font-semibold",children:"Gates"}),e.jsx("div",{className:"text-lg font-mono",children:d.gates})]}),e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/30 px-3 py-2",children:[e.jsx("div",{className:"text-violet-700 dark:text-violet-300 font-semibold",children:"Parameters"}),e.jsx("div",{className:"text-lg font-mono",children:d.params})]}),e.jsx("div",{className:"col-span-3 text-left text-gray-600 dark:text-gray-400 mt-1",children:d.note})]})]})}function L(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Several LSTM variants modify the gating mechanism to improve performance or reduce computational cost. Understanding these variants helps select the right architecture for a given task."}),e.jsxs(h,{title:"Peephole Connections",children:[e.jsxs("p",{children:["Peephole LSTMs allow the gates to access the cell state ",e.jsx(t.InlineMath,{math:"c_{t-1}"})," directly:"]}),e.jsx(t.BlockMath,{math:"f_t = \\sigma(W_f [h_{t-1}, x_t] + W_{pf} \\odot c_{t-1} + b_f)"}),e.jsx(t.BlockMath,{math:"i_t = \\sigma(W_i [h_{t-1}, x_t] + W_{pi} \\odot c_{t-1} + b_i)"}),e.jsx(t.BlockMath,{math:"o_t = \\sigma(W_o [h_{t-1}, x_t] + W_{po} \\odot c_t + b_o)"}),e.jsxs("p",{className:"mt-2",children:["The diagonal weight matrices ",e.jsx(t.InlineMath,{math:"W_{pf}, W_{pi}, W_{po}"})," add",e.jsx(t.InlineMath,{math:"3h"})," extra parameters, letting the gates make more informed decisions."]})]}),e.jsxs(h,{title:"Coupled Forget-Input Gate",children:[e.jsx("p",{children:"Instead of independent forget and input gates, use a single gate:"}),e.jsx(t.BlockMath,{math:"c_t = f_t \\odot c_{t-1} + (1 - f_t) \\odot \\tilde{c}_t"}),e.jsx("p",{className:"mt-2",children:"This couples forgetting and updating: the cell can only write new information to the extent it forgets old information, reducing parameters by one gate matrix."})]}),e.jsx(z,{}),e.jsxs(x,{title:"GRU vs LSTM at a Glance",children:[e.jsxs("p",{children:["The GRU merges forget and input gates into an ",e.jsx("strong",{children:"update gate"})," and eliminates the separate cell state:"]}),e.jsx(t.BlockMath,{math:"z_t = \\sigma(W_z [h_{t-1}, x_t])"}),e.jsx(t.BlockMath,{math:"h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t"}),e.jsx("p",{children:"GRU has ~75% of the LSTM parameters and often matches LSTM performance on shorter sequences."})]}),e.jsx(p,{title:"GRU vs LSTM Parameter Count",code:`import torch.nn as nn

input_size, hidden_size = 64, 256

lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
gru = nn.GRU(input_size, hidden_size, batch_first=True)

lstm_params = sum(p.numel() for p in lstm.parameters())
gru_params = sum(p.numel() for p in gru.parameters())

print(f"LSTM params: {lstm_params:,}")    # 4 * 256 * (256+64) + 4*256
print(f"GRU  params: {gru_params:,}")     # 3 * 256 * (256+64) + 3*256
print(f"GRU / LSTM:  {gru_params/lstm_params:.2%}")  # ~75%

# Greff et al. (2017) LSTM variant ablation:
# - Forget gate and output activation are most critical
# - Peephole connections provide marginal benefit
# - Coupled gates perform comparably to standard LSTM`}),e.jsx(f,{type:"note",title:"Which Variant to Choose?",children:e.jsxs("p",{children:["Large-scale studies (Greff et al., 2017; Jozefowicz et al., 2015) found that ",e.jsx("strong",{children:"no variant consistently outperforms the standard LSTM"}),". The forget gate bias initialization (set to 1.0) matters more than architectural changes. Start with a standard LSTM or GRU and only try variants if you have a specific bottleneck."]})}),e.jsx(x,{title:"Practical Recommendation",children:e.jsx("p",{children:"Begin with a standard LSTM with forget bias = 1.0. If you need fewer parameters or faster training, switch to a GRU. Only explore peephole or coupled-gate variants if you have evidence from ablation studies on your specific task that they help."})})]})}const re=Object.freeze(Object.defineProperty({__proto__:null,default:L},Symbol.toStringTag,{value:"Module"}));function R(){const[a,c]=m.useState(5),l=[1.2,3.5,8,15,2.1,50,.8,7.3];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Gradient Clipping Visualization"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Max norm: ",a.toFixed(1),e.jsx("input",{type:"range",min:1,max:20,step:.5,value:a,onChange:d=>c(parseFloat(d.target.value)),className:"w-40 accent-violet-500"})]}),e.jsx("div",{className:"flex items-end gap-2 h-32",children:l.map((d,r)=>{const i=Math.min(d,a),s=2.2;return e.jsxs("div",{className:"flex flex-col items-center gap-1 flex-1",children:[e.jsx("span",{className:"text-xs text-gray-500",children:i.toFixed(1)}),e.jsxs("div",{className:"w-full flex flex-col items-center",children:[e.jsx("div",{style:{height:`${i*s}px`},className:`w-6 rounded-t ${d>a?"bg-violet-400":"bg-violet-600"}`}),d>a&&e.jsx("div",{style:{height:`${(d-i)*s}px`},className:"w-6 bg-red-200 dark:bg-red-900/40 border-t border-dashed border-red-400"})]})]},r)})}),e.jsx("p",{className:"text-xs text-gray-500 mt-2 text-center",children:"Violet = kept, red dashed = clipped away"})]})}function B(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Training LSTMs effectively requires careful attention to gradient management, weight initialization, and hyperparameter choices. These practical techniques are essential for stable, fast convergence."}),e.jsxs(h,{title:"Gradient Clipping",children:[e.jsxs("p",{children:["Gradient clipping rescales the gradient when its norm exceeds a threshold ",e.jsx(t.InlineMath,{math:"\\theta"}),":"]}),e.jsx(t.BlockMath,{math:"g \\leftarrow \\begin{cases} g & \\text{if } \\|g\\| \\leq \\theta \\\\ \\theta \\cdot \\frac{g}{\\|g\\|} & \\text{if } \\|g\\| > \\theta \\end{cases}"}),e.jsx("p",{className:"mt-2",children:"This preserves gradient direction while bounding its magnitude, preventing the catastrophic parameter updates caused by exploding gradients."})]}),e.jsx(R,{}),e.jsxs(y,{title:"Forget Gate Bias Initialization",id:"forget-bias-init",children:[e.jsxs("p",{children:["Initializing the forget gate bias to a positive value (typically 1.0 or 2.0) ensures that ",e.jsx(t.InlineMath,{math:"f_t \\approx 1"})," at the start of training, which allows gradients to flow through the cell state early in training:"]}),e.jsx(t.BlockMath,{math:"b_f \\leftarrow 1.0 \\implies f_t = \\sigma(Wh + Wx + 1.0) \\approx 0.73"}),e.jsx("p",{children:"This simple trick, proposed by Gers et al. (2000), significantly improves LSTM training stability and is now standard practice."})]}),e.jsx(p,{title:"LSTM Training with Best Practices",code:`import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2,
                           dropout=0.3, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

        # Best practice: initialize forget gate bias to 1.0
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)  # forget gate bias
            elif 'weight' in name:
                nn.init.orthogonal_(param)  # orthogonal init for stability

    def forward(self, x):
        e = self.embed(x)
        out, (h_n, _) = self.lstm(e)
        return self.fc(self.dropout(h_n[-1]))

model = LSTMClassifier(10000, 128, 256, 5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop with gradient clipping
for epoch in range(5):
    x = torch.randint(0, 10000, (32, 50))
    y = torch.randint(0, 5, (32,))
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    print(f"Epoch {epoch}: loss={loss.item():.4f}")`}),e.jsx(j,{title:"Common LSTM Training Pitfalls",children:e.jsxs("p",{children:[e.jsx("strong",{children:"1. Forgetting to clip gradients"})," leads to NaN losses.",e.jsx("strong",{children:" 2. Default zero bias"})," for forget gates causes information loss early in training.",e.jsx("strong",{children:" 3. Too-high learning rate"})," with Adam can destabilize LSTMs more than feedforward networks due to the recurrent dynamics."]})}),e.jsx(f,{type:"note",title:"Learning Rate Scheduling",children:e.jsxs("p",{children:["LSTMs benefit from learning rate warmup (linearly increasing over the first ~1000 steps) followed by cosine or step decay. A common recipe: start with ",e.jsx(t.InlineMath,{math:"\\text{lr} = 10^{-3}"})," for Adam or ",e.jsx(t.InlineMath,{math:"\\text{lr} = 1.0"})," for SGD with gradient clipping at 0.25-5.0."]})})]})}const ie=Object.freeze(Object.defineProperty({__proto__:null,default:B},Symbol.toStringTag,{value:"Module"}));function W(){const[a,c]=m.useState(.6),[l,d]=m.useState(.8),[r,i]=m.useState(1),n=Math.tanh(.5+l*r),o=(1-a)*r+a*n;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"GRU Gate Explorer"}),e.jsxs("div",{className:"grid grid-cols-3 gap-3 mb-4",children:[e.jsxs("label",{className:"flex flex-col gap-1 text-sm text-gray-600 dark:text-gray-400",children:["Update gate (z): ",a.toFixed(2),e.jsx("input",{type:"range",min:0,max:1,step:.01,value:a,onChange:g=>c(parseFloat(g.target.value)),className:"accent-violet-500"})]}),e.jsxs("label",{className:"flex flex-col gap-1 text-sm text-gray-600 dark:text-gray-400",children:["Reset gate (r): ",l.toFixed(2),e.jsx("input",{type:"range",min:0,max:1,step:.01,value:l,onChange:g=>d(parseFloat(g.target.value)),className:"accent-violet-500"})]}),e.jsxs("label",{className:"flex flex-col gap-1 text-sm text-gray-600 dark:text-gray-400",children:["Previous h: ",r.toFixed(2),e.jsx("input",{type:"range",min:-2,max:2,step:.01,value:r,onChange:g=>i(parseFloat(g.target.value)),className:"accent-violet-500"})]})]}),e.jsxs("div",{className:"flex gap-6 justify-center text-sm",children:[e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/30 px-4 py-2 text-center",children:[e.jsx("div",{className:"text-violet-700 dark:text-violet-300 font-semibold",children:"Candidate"}),e.jsx("div",{className:"text-lg font-mono",children:n.toFixed(4)})]}),e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/30 px-4 py-2 text-center",children:[e.jsx("div",{className:"text-violet-700 dark:text-violet-300 font-semibold",children:"New h_t"}),e.jsx("div",{className:"text-lg font-mono",children:o.toFixed(4)})]}),e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/30 px-4 py-2 text-center",children:[e.jsx("div",{className:"text-violet-700 dark:text-violet-300 font-semibold",children:"% from prev"}),e.jsxs("div",{className:"text-lg font-mono",children:[((1-a)*100).toFixed(0),"%"]})]})]})]})}function q(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"The Gated Recurrent Unit (GRU), introduced by Cho et al. (2014), simplifies the LSTM architecture by merging the cell and hidden states into a single state vector and using only two gates instead of three."}),e.jsxs(h,{title:"GRU Equations",children:[e.jsxs("p",{children:["Given input ",e.jsx(t.InlineMath,{math:"x_t"})," and previous hidden state ",e.jsx(t.InlineMath,{math:"h_{t-1}"}),":"]}),e.jsx(t.BlockMath,{math:"z_t = \\sigma(W_z [h_{t-1}, x_t] + b_z) \\quad \\text{(update gate)}"}),e.jsx(t.BlockMath,{math:"r_t = \\sigma(W_r [h_{t-1}, x_t] + b_r) \\quad \\text{(reset gate)}"}),e.jsx(t.BlockMath,{math:"\\tilde{h}_t = \\tanh(W_h [r_t \\odot h_{t-1}, x_t] + b_h) \\quad \\text{(candidate)}"}),e.jsx(t.BlockMath,{math:"h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t"})]}),e.jsx(W,{}),e.jsxs(x,{title:"Gate Intuition",children:[e.jsxs("p",{children:[e.jsx("strong",{children:"Update gate"})," ",e.jsx(t.InlineMath,{math:"z_t"}),": Controls how much new information to mix in. When ",e.jsx(t.InlineMath,{math:"z_t \\to 0"}),", the hidden state is copied unchanged (like an LSTM with ",e.jsx(t.InlineMath,{math:"f_t = 1"}),"). When ",e.jsx(t.InlineMath,{math:"z_t \\to 1"}),", the state is fully replaced by the candidate."]}),e.jsxs("p",{className:"mt-2",children:[e.jsx("strong",{children:"Reset gate"})," ",e.jsx(t.InlineMath,{math:"r_t"}),": Controls how much past state to expose when computing the candidate. When ",e.jsx(t.InlineMath,{math:"r_t \\to 0"}),", the candidate ignores the past, allowing the GRU to drop irrelevant history."]})]}),e.jsx(p,{title:"GRU Implementation from Scratch",code:`import torch
import torch.nn as nn

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Update gate
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        # Reset gate
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        # Candidate
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h_prev):
        combined = torch.cat([h_prev, x], dim=-1)
        z = torch.sigmoid(self.W_z(combined))
        r = torch.sigmoid(self.W_r(combined))
        combined_r = torch.cat([r * h_prev, x], dim=-1)
        h_candidate = torch.tanh(self.W_h(combined_r))
        h_new = (1 - z) * h_prev + z * h_candidate
        return h_new

# Test
cell = GRUCell(64, 128)
h = torch.zeros(8, 128)
for t in range(20):
    x_t = torch.randn(8, 64)
    h = cell(x_t, h)
print(f"Final h: {h.shape}")  # (8, 128)`}),e.jsx(f,{type:"note",title:"GRU Simplicity Advantage",children:e.jsx("p",{children:"With no separate cell state and fewer gates, the GRU is faster to compute per step and has ~25% fewer parameters than an LSTM. This makes GRUs particularly attractive for smaller datasets or when inference speed is critical. The GRU achieves comparable performance to the LSTM on many benchmarks."})})]})}const oe=Object.freeze(Object.defineProperty({__proto__:null,default:q},Symbol.toStringTag,{value:"Module"}));function I(){const[a,c]=m.useState("all"),l=[{feature:"Gates",lstm:"3 (forget, input, output)",gru:"2 (update, reset)",category:"arch"},{feature:"State vectors",lstm:"h_t and c_t",gru:"h_t only",category:"arch"},{feature:"Parameters (h=256, d=64)",lstm:"~328K",gru:"~246K",category:"cost"},{feature:"Speed (relative)",lstm:"1.0x",gru:"~1.3x faster",category:"cost"},{feature:"Long-range deps",lstm:"Excellent",gru:"Good",category:"perf"},{feature:"Small datasets",lstm:"May overfit",gru:"Better generalization",category:"perf"},{feature:"Speech recognition",lstm:"Preferred",gru:"Comparable",category:"perf"},{feature:"Machine translation",lstm:"Preferred",gru:"Comparable",category:"perf"}],d=a==="all"?l:l.filter(r=>r.category===a);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"LSTM vs GRU Comparison"}),e.jsx("div",{className:"flex gap-2 mb-3 flex-wrap",children:[["all","All"],["arch","Architecture"],["cost","Cost"],["perf","Performance"]].map(([r,i])=>e.jsx("button",{onClick:()=>c(r),className:`px-3 py-1 rounded-lg text-sm ${a===r?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:i},r))}),e.jsxs("table",{className:"w-full text-sm",children:[e.jsx("thead",{children:e.jsxs("tr",{className:"border-b border-gray-200 dark:border-gray-700",children:[e.jsx("th",{className:"text-left py-2 text-gray-600 dark:text-gray-400",children:"Feature"}),e.jsx("th",{className:"text-left py-2 text-violet-600 dark:text-violet-400",children:"LSTM"}),e.jsx("th",{className:"text-left py-2 text-violet-600 dark:text-violet-400",children:"GRU"})]})}),e.jsx("tbody",{children:d.map((r,i)=>e.jsxs("tr",{className:"border-b border-gray-100 dark:border-gray-800",children:[e.jsx("td",{className:"py-2 text-gray-700 dark:text-gray-300 font-medium",children:r.feature}),e.jsx("td",{className:"py-2 text-gray-600 dark:text-gray-400",children:r.lstm}),e.jsx("td",{className:"py-2 text-gray-600 dark:text-gray-400",children:r.gru})]},i))})]})]})}function P(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"The choice between LSTM and GRU depends on the task, dataset size, and computational constraints. Empirical evidence shows neither consistently dominates the other, but understanding their trade-offs helps make informed decisions."}),e.jsx(I,{}),e.jsx(y,{title:"Empirical Findings (Chung et al., 2014)",id:"lstm-gru-empirical",children:e.jsx("p",{children:"Across music modeling, speech signal modeling, and NLP tasks, the GRU and LSTM achieve comparable performance. The GRU tends to converge faster due to fewer parameters. However, on tasks requiring very long-range dependencies (sequences of 100+ steps), the LSTM's separate cell state provides a modest advantage."})}),e.jsxs(x,{title:"Decision Framework",children:[e.jsxs("p",{children:["Use ",e.jsx("strong",{children:"LSTM"})," when:"]}),e.jsxs("ul",{className:"list-disc ml-6 mt-1 space-y-1",children:[e.jsx("li",{children:"Sequences are very long (hundreds of steps)"}),e.jsx("li",{children:"You have sufficient compute and data"}),e.jsx("li",{children:"The task requires fine-grained memory control"})]}),e.jsxs("p",{className:"mt-2",children:["Use ",e.jsx("strong",{children:"GRU"})," when:"]}),e.jsxs("ul",{className:"list-disc ml-6 mt-1 space-y-1",children:[e.jsx("li",{children:"Speed and efficiency matter"}),e.jsx("li",{children:"The dataset is small (fewer parameters = less overfitting)"}),e.jsx("li",{children:"Sequences are moderate length"})]})]}),e.jsx(p,{title:"Benchmarking LSTM vs GRU",code:`import torch
import torch.nn as nn
import time

def benchmark(model, x, n_runs=100):
    # Warmup
    for _ in range(10):
        model(x)
    torch.cuda.synchronize() if x.is_cuda else None
    start = time.time()
    for _ in range(n_runs):
        model(x)
    torch.cuda.synchronize() if x.is_cuda else None
    return (time.time() - start) / n_runs * 1000  # ms

input_size, hidden_size, seq_len = 64, 256, 100
lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)

x = torch.randn(32, seq_len, input_size)

lstm_time = benchmark(lstm, x)
gru_time = benchmark(gru, x)

print(f"LSTM: {lstm_time:.2f} ms/batch")
print(f"GRU:  {gru_time:.2f} ms/batch")
print(f"GRU speedup: {lstm_time/gru_time:.2f}x")

# Typical result: GRU is 1.2-1.4x faster than LSTM`}),e.jsx(f,{type:"note",title:"Modern Perspective",children:e.jsxs("p",{children:["With the rise of Transformers, the LSTM vs GRU debate has become less central. However, both remain highly relevant for on-device inference, streaming applications, and tasks where the ",e.jsx(t.InlineMath,{math:"O(n^2)"})," attention cost of Transformers is prohibitive. In practice, ",e.jsx("strong",{children:"try both and pick based on validation performance"}),"."]})})]})}const le=Object.freeze(Object.defineProperty({__proto__:null,default:P},Symbol.toStringTag,{value:"Module"}));function C(){const[a,c]=m.useState(100),l=[{name:"LSTM",base:1,parallel:!1},{name:"GRU",base:.75,parallel:!1},{name:"SRU",base:.4,parallel:!0},{name:"QRNN",base:.35,parallel:!0}],d=a*1;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Relative Throughput"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Sequence length: ",a,e.jsx("input",{type:"range",min:10,max:500,step:10,value:a,onChange:r=>c(parseInt(r.target.value)),className:"w-40 accent-violet-500"})]}),e.jsx("div",{className:"space-y-2",children:l.map(r=>{const i=r.parallel?r.base*Math.log2(a+1):r.base*a,s=Math.min(i/d*100,100);return e.jsxs("div",{className:"flex items-center gap-3",children:[e.jsx("span",{className:"w-14 text-sm text-gray-600 dark:text-gray-400 font-mono",children:r.name}),e.jsx("div",{className:"flex-1 bg-gray-100 dark:bg-gray-800 rounded h-5 overflow-hidden",children:e.jsx("div",{className:"h-full bg-violet-500 rounded",style:{width:`${s}%`}})}),e.jsx("span",{className:"text-xs text-gray-500 w-16 text-right",children:r.parallel?"parallel":"sequential"})]},r.name)})})]})}function G(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Standard RNNs, LSTMs, and GRUs are inherently sequential, limiting GPU utilization. Efficient recurrent variants like SRU and QRNN restructure the computation to maximize parallelism while retaining sequential modeling capability."}),e.jsxs(h,{title:"Simple Recurrent Unit (SRU)",children:[e.jsx("p",{children:"The SRU (Lei et al., 2018) separates gating from hidden state computation:"}),e.jsx(t.BlockMath,{math:"\\tilde{x}_t = W x_t"}),e.jsx(t.BlockMath,{math:"f_t = \\sigma(W_f x_t + b_f)"}),e.jsx(t.BlockMath,{math:"c_t = f_t \\odot c_{t-1} + (1 - f_t) \\odot \\tilde{x}_t"}),e.jsx(t.BlockMath,{math:"h_t = r_t \\odot \\tanh(c_t) + (1 - r_t) \\odot x_t"}),e.jsxs("p",{className:"mt-2",children:["The key insight: ",e.jsx(t.InlineMath,{math:"W x_t"})," and ",e.jsx(t.InlineMath,{math:"W_f x_t"})," depend only on the input and can be computed in parallel across all time steps. Only the lightweight element-wise recurrence is sequential."]})]}),e.jsxs(h,{title:"Quasi-Recurrent Neural Network (QRNN)",children:[e.jsx("p",{children:"The QRNN (Bradbury et al., 2017) applies convolutions across time, then a minimal recurrence:"}),e.jsx(t.BlockMath,{math:"Z = \\tanh(W_z * X), \\quad F = \\sigma(W_f * X)"}),e.jsx(t.BlockMath,{math:"c_t = f_t \\odot c_{t-1} + (1 - f_t) \\odot z_t"}),e.jsxs("p",{className:"mt-2",children:["The convolution ",e.jsx(t.InlineMath,{math:"*"})," captures local temporal patterns in parallel, while the element-wise gated pooling propagates information sequentially. This is 2-17x faster than LSTMs in practice."]})]}),e.jsx(C,{}),e.jsxs(x,{title:"Parallelism Comparison",children:[e.jsxs("p",{children:["For a sequence of length ",e.jsx(t.InlineMath,{math:"T"})," with hidden size ",e.jsx(t.InlineMath,{math:"h"}),":"]}),e.jsxs("ul",{className:"list-disc ml-6 mt-1 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"LSTM/GRU"}),": ",e.jsx(t.InlineMath,{math:"O(T)"})," sequential matrix multiplications of ",e.jsx(t.InlineMath,{math:"O(h^2)"})]}),e.jsxs("li",{children:[e.jsx("strong",{children:"SRU"}),": One batched matmul ",e.jsx(t.InlineMath,{math:"O(Th^2)"})," parallel, then ",e.jsx(t.InlineMath,{math:"O(T)"})," element-wise ops"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"QRNN"}),": One convolution ",e.jsx(t.InlineMath,{math:"O(Thk)"})," parallel, then ",e.jsx(t.InlineMath,{math:"O(T)"})," element-wise ops"]})]})]}),e.jsx(p,{title:"SRU-like Efficient Recurrence",code:`import torch
import torch.nn as nn

class SimpleSRU(nn.Module):
    """Simplified SRU showing the parallel/sequential split."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W = nn.Linear(input_size, hidden_size)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.W_r = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        # Parallel across time: matrix multiplications
        x_tilde = self.W(x)            # (B, T, H)
        f = torch.sigmoid(self.W_f(x)) # (B, T, H)
        r = torch.sigmoid(self.W_r(x)) # (B, T, H)

        # Sequential: lightweight element-wise recurrence
        B, T, H = x_tilde.shape
        c = torch.zeros(B, H, device=x.device)
        outputs = []
        for t in range(T):
            c = f[:, t] * c + (1 - f[:, t]) * x_tilde[:, t]
            h = r[:, t] * torch.tanh(c) + (1 - r[:, t]) * x[:, t, :H]
            outputs.append(h)
        return torch.stack(outputs, dim=1)

model = SimpleSRU(64, 128)
x = torch.randn(8, 100, 64)
out = model(x)
print(f"Output: {out.shape}")  # (8, 100, 128)`}),e.jsx(f,{type:"note",title:"When to Use Efficient RNN Variants",children:e.jsxs("p",{children:["SRU and QRNN shine when you need RNN-like sequential modeling but cannot afford the latency of standard LSTMs. They are especially useful for long sequences on GPUs. However, for most modern NLP tasks, Transformers have supplanted these architectures. Efficient RNNs remain relevant in ",e.jsx("strong",{children:"streaming"}),", ",e.jsx("strong",{children:"edge deployment"}),", and ",e.jsx("strong",{children:"low-latency"})," applications."]})})]})}const de=Object.freeze(Object.defineProperty({__proto__:null,default:G},Symbol.toStringTag,{value:"Module"}));function A(){const[a,c]=m.useState(3),l=440;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Bidirectional RNN"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Time steps: ",a+1,e.jsx("input",{type:"range",min:2,max:5,step:1,value:a,onChange:r=>c(parseInt(r.target.value)),className:"w-32 accent-violet-500"})]}),e.jsxs("svg",{width:l,height:200,className:"mx-auto block",children:[e.jsxs("defs",{children:[e.jsx("marker",{id:"biArr",markerWidth:7,markerHeight:5,refX:7,refY:2.5,orient:"auto",children:e.jsx("path",{d:"M0,0 L7,2.5 L0,5 Z",fill:"#8b5cf6"})}),e.jsx("marker",{id:"biArrB",markerWidth:7,markerHeight:5,refX:7,refY:2.5,orient:"auto",children:e.jsx("path",{d:"M0,0 L7,2.5 L0,5 Z",fill:"#f97316"})})]}),Array.from({length:a+1},(r,i)=>{const s=50+i*(340/a);return e.jsxs("g",{children:[e.jsx("rect",{x:s-20,y:50,width:40,height:28,rx:4,fill:"#8b5cf6",opacity:.8}),e.jsx("text",{x:s,y:69,textAnchor:"middle",fill:"white",fontSize:9,fontWeight:"bold",children:"fwd"}),e.jsx("rect",{x:s-20,y:100,width:40,height:28,rx:4,fill:"#f97316",opacity:.8}),e.jsx("text",{x:s,y:119,textAnchor:"middle",fill:"white",fontSize:9,fontWeight:"bold",children:"bwd"}),e.jsxs("text",{x:s,y:178,textAnchor:"middle",fill:"#6b7280",fontSize:10,children:["x_",i]}),e.jsx("line",{x1:s,y1:168,x2:s,y2:128,stroke:"#9ca3af",strokeWidth:1}),e.jsx("line",{x1:s,y1:168,x2:s,y2:78,stroke:"#9ca3af",strokeWidth:1}),e.jsxs("text",{x:s,y:30,textAnchor:"middle",fill:"#6b7280",fontSize:10,children:["y_",i]}),e.jsx("line",{x1:s,y1:50,x2:s,y2:36,stroke:"#9ca3af",strokeWidth:1}),i<a&&e.jsx("line",{x1:s+20,y1:64,x2:s+340/a-20,y2:64,stroke:"#8b5cf6",strokeWidth:1.5,markerEnd:"url(#biArr)"}),i>0&&e.jsx("line",{x1:s-20,y1:114,x2:s-340/a+20,y2:114,stroke:"#f97316",strokeWidth:1.5,markerEnd:"url(#biArrB)"})]},i)}),e.jsx("text",{x:l-10,y:69,textAnchor:"end",fill:"#8b5cf6",fontSize:9,children:"forward"}),e.jsx("text",{x:l-10,y:119,textAnchor:"end",fill:"#f97316",fontSize:9,children:"backward"})]})]})}function F(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Bidirectional RNNs process sequences in both forward and backward directions, allowing each output to depend on both past and future context. This is especially powerful for tasks where the full sequence is available at inference time."}),e.jsxs(h,{title:"Bidirectional RNN",children:[e.jsx("p",{children:"A bidirectional RNN runs two separate hidden state sequences:"}),e.jsx(t.BlockMath,{math:"\\overrightarrow{h_t} = f(W_{\\rightarrow}[\\overrightarrow{h_{t-1}}, x_t])"}),e.jsx(t.BlockMath,{math:"\\overleftarrow{h_t} = f(W_{\\leftarrow}[\\overleftarrow{h_{t+1}}, x_t])"}),e.jsx("p",{className:"mt-2",children:"The output at each step concatenates both directions:"}),e.jsx(t.BlockMath,{math:"h_t = [\\overrightarrow{h_t}; \\overleftarrow{h_t}] \\in \\mathbb{R}^{2d}"})]}),e.jsx(A,{}),e.jsx(x,{title:"Named Entity Recognition",children:e.jsx("p",{children:'In the sentence "Paris is the capital of France", recognizing "Paris" as a location benefits from seeing "capital of France" to the right. A forward-only RNN at position 0 has no future context. A BiRNN at position 0 sees both the word itself and a backward summary of the full sentence.'})}),e.jsx(p,{title:"Bidirectional LSTM in PyTorch",code:`import torch
import torch.nn as nn

bilstm = nn.LSTM(
    input_size=64,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
    bidirectional=True  # key argument
)

x = torch.randn(8, 30, 64)
output, (h_n, c_n) = bilstm(x)

print(f"Output: {output.shape}")   # (8, 30, 256) = 2 * hidden_size
print(f"Hidden: {h_n.shape}")      # (4, 8, 128) = 2*num_layers x batch x hidden

# Extract final forward and backward hidden states
h_forward = h_n[-2]   # last layer, forward
h_backward = h_n[-1]  # last layer, backward
h_combined = torch.cat([h_forward, h_backward], dim=-1)
print(f"Combined: {h_combined.shape}")  # (8, 256)

# For classification, use the combined representation
classifier = nn.Linear(256, 10)
logits = classifier(h_combined)
print(f"Logits: {logits.shape}")  # (8, 10)`}),e.jsx(j,{title:"Cannot Use for Autoregressive Generation",children:e.jsxs("p",{children:["Bidirectional RNNs require the full input sequence at inference time. They are",e.jsx("strong",{children:" not suitable for autoregressive generation"})," (e.g., language modeling, text generation), where tokens are produced one at a time and future context is unavailable. Use unidirectional RNNs for generation tasks."]})}),e.jsx(f,{type:"note",title:"Applications",children:e.jsx("p",{children:"BiRNNs are widely used in NER, POS tagging, sentiment analysis, machine translation encoders, and speech recognition. ELMo (Peters et al., 2018) uses a deep bidirectional LSTM to produce contextual word embeddings that became foundational for transfer learning in NLP."})})]})}const ce=Object.freeze(Object.defineProperty({__proto__:null,default:F},Symbol.toStringTag,{value:"Module"}));function U(){const[a,c]=m.useState(3),l=400,d=220,r=4;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsxs("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:["Stacked RNN (",a," layers)"]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Layers: ",a,e.jsx("input",{type:"range",min:1,max:4,step:1,value:a,onChange:i=>c(parseInt(i.target.value)),className:"w-32 accent-violet-500"})]}),e.jsxs("svg",{width:l,height:d,className:"mx-auto block",children:[Array.from({length:a},(i,s)=>{const n=d-40-s*45,o=.5+s*(.5/(a-1||1));return Array.from({length:r},(g,_)=>{const u=50+_*90;return e.jsxs("g",{children:[e.jsx("rect",{x:u-18,y:n-14,width:36,height:28,rx:4,fill:"#8b5cf6",opacity:o}),e.jsxs("text",{x:u,y:n+4,textAnchor:"middle",fill:"white",fontSize:8,children:["L",s]}),_<r-1&&e.jsx("line",{x1:u+18,y1:n,x2:u+72,y2:n,stroke:"#a78bfa",strokeWidth:1,opacity:.6}),s>0&&e.jsx("line",{x1:u,y1:n+14,x2:u,y2:n+31,stroke:"#c4b5fd",strokeWidth:1,opacity:.6})]},`${s}-${_}`)})}),Array.from({length:r},(i,s)=>e.jsxs("text",{x:50+s*90,y:d-8,textAnchor:"middle",fill:"#6b7280",fontSize:10,children:["t=",s]},s)),e.jsx("text",{x:l/2,y:14,textAnchor:"middle",fill:"#6b7280",fontSize:10,children:"output layer"})]})]})}function O(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Stacking multiple RNN layers creates a deep recurrent architecture where each layer processes the output sequence of the layer below, enabling the network to learn hierarchical temporal representations."}),e.jsxs(h,{title:"Stacked (Multi-Layer) RNN",children:[e.jsxs("p",{children:["For a stack of ",e.jsx(t.InlineMath,{math:"L"})," layers, the hidden state at layer ",e.jsx(t.InlineMath,{math:"l"})," and time ",e.jsx(t.InlineMath,{math:"t"})," is:"]}),e.jsx(t.BlockMath,{math:"h_t^{(l)} = f(W^{(l)} [h_{t-1}^{(l)}, h_t^{(l-1)}] + b^{(l)})"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"h_t^{(0)} = x_t"})," is the input. Each layer has its own set of parameters, and the output of the top layer ",e.jsx(t.InlineMath,{math:"h_t^{(L)}"})," is used for predictions."]})]}),e.jsx(U,{}),e.jsxs(y,{title:"Residual Connections for Deep RNNs",id:"residual-rnn",children:[e.jsx("p",{children:"For deep stacks (3+ layers), residual connections prevent gradient degradation:"}),e.jsx(t.BlockMath,{math:"h_t^{(l)} = f(W^{(l)} [h_{t-1}^{(l)}, h_t^{(l-1)}]) + h_t^{(l-1)}"}),e.jsx("p",{children:"This ensures that gradients can flow directly from the output to early layers, analogous to residual connections in deep feedforward networks. The identity shortcut makes it easy for the network to learn an identity mapping at each layer."})]}),e.jsx(x,{title:"Typical Depth Guidelines",children:e.jsxs("p",{children:[e.jsx("strong",{children:"2 layers"}),": Standard choice for most tasks, significant improvement over 1 layer.",e.jsx("strong",{children:" 3-4 layers"}),": Used in machine translation and speech recognition.",e.jsx("strong",{children:" 8+ layers"}),": Rare for RNNs; usually requires residual connections, layer normalization, and careful initialization. Google's NMT system used 8 LSTM layers with residuals."]})}),e.jsx(p,{title:"Deep LSTM with Residual Connections",code:`import torch
import torch.nn as nn

class ResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=4, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.layers.append(nn.LSTM(in_dim, hidden_size, batch_first=True))
            self.norms.append(nn.LayerNorm(hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(input_size, hidden_size)  # for first residual

    def forward(self, x):
        for i, (lstm, norm) in enumerate(zip(self.layers, self.norms)):
            out, _ = lstm(x)
            out = norm(out)
            out = self.dropout(out)
            if i == 0:
                x = out + self.proj(x)  # project input to hidden dim
            else:
                x = out + x  # residual connection
        return x

model = ResidualLSTM(64, 256, num_layers=4)
x = torch.randn(8, 50, 64)
out = model(x)
print(f"Output: {out.shape}")  # (8, 50, 256)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")`}),e.jsx(f,{type:"note",title:"Layer Normalization in RNNs",children:e.jsxs("p",{children:["Unlike batch normalization, ",e.jsx("strong",{children:"layer normalization"})," normalizes across the feature dimension within each time step, making it natural for variable-length sequences. Applied after each recurrent layer, it stabilizes training of deep RNNs and allows higher learning rates. It has become the default normalization for recurrent architectures."]})})]})}const he=Object.freeze(Object.defineProperty({__proto__:null,default:O},Symbol.toStringTag,{value:"Module"}));function E(){const[a,c]=m.useState(!0),l=460,d=160,r=["le","chat","est","noir"],i=["the","cat","is","black"];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Encoder-Decoder Architecture"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:[e.jsx("input",{type:"checkbox",checked:a,onChange:s=>c(s.target.checked),className:"accent-violet-500"}),"Show context vector"]}),e.jsxs("svg",{width:l,height:d,className:"mx-auto block",children:[r.map((s,n)=>{const o=30+n*55;return e.jsxs("g",{children:[e.jsx("rect",{x:o-20,y:50,width:40,height:28,rx:4,fill:"#8b5cf6",opacity:.8}),e.jsx("text",{x:o,y:68,textAnchor:"middle",fill:"white",fontSize:8,fontWeight:"bold",children:"enc"}),e.jsx("text",{x:o,y:100,textAnchor:"middle",fill:"#6b7280",fontSize:9,children:s}),n<3&&e.jsx("line",{x1:o+20,y1:64,x2:o+35,y2:64,stroke:"#a78bfa",strokeWidth:1.5})]},`e${n}`)}),a&&e.jsxs(e.Fragment,{children:[e.jsx("circle",{cx:230,cy:64,r:14,fill:"#7c3aed",opacity:.9}),e.jsx("text",{x:230,y:68,textAnchor:"middle",fill:"white",fontSize:8,fontWeight:"bold",children:"c"}),e.jsx("line",{x1:195,y1:64,x2:216,y2:64,stroke:"#a78bfa",strokeWidth:2})]}),i.map((s,n)=>{const o=270+n*50;return e.jsxs("g",{children:[e.jsx("rect",{x:o-20,y:50,width:40,height:28,rx:4,fill:"#f97316",opacity:.8}),e.jsx("text",{x:o,y:68,textAnchor:"middle",fill:"white",fontSize:8,fontWeight:"bold",children:"dec"}),e.jsx("text",{x:o,y:28,textAnchor:"middle",fill:"#6b7280",fontSize:9,children:s}),n<3&&e.jsx("line",{x1:o+20,y1:64,x2:o+30,y2:64,stroke:"#fdba74",strokeWidth:1.5}),a&&e.jsx("line",{x1:230,y1:78,x2:o,y2:50,stroke:"#c4b5fd",strokeWidth:.7,strokeDasharray:"3,2",opacity:.5})]},`d${n}`)}),e.jsx("text",{x:100,y:140,textAnchor:"middle",fill:"#8b5cf6",fontSize:10,children:"Encoder"}),e.jsx("text",{x:345,y:140,textAnchor:"middle",fill:"#f97316",fontSize:10,children:"Decoder"})]})]})}function D(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"The encoder-decoder framework maps variable-length input sequences to variable-length output sequences through a fixed-size context vector, forming the basis of sequence-to-sequence (seq2seq) models."}),e.jsxs(h,{title:"Encoder-Decoder Framework",children:[e.jsxs("p",{children:["The ",e.jsx("strong",{children:"encoder"})," reads the input sequence and produces a context vector:"]}),e.jsx(t.BlockMath,{math:"h_t^{\\text{enc}} = f_{\\text{enc}}(x_t, h_{t-1}^{\\text{enc}}), \\quad c = h_T^{\\text{enc}}"}),e.jsxs("p",{className:"mt-2",children:["The ",e.jsx("strong",{children:"decoder"})," generates the output sequence conditioned on ",e.jsx(t.InlineMath,{math:"c"}),":"]}),e.jsx(t.BlockMath,{math:"h_t^{\\text{dec}} = f_{\\text{dec}}(y_{t-1}, h_{t-1}^{\\text{dec}}, c)"}),e.jsx(t.BlockMath,{math:"P(y_t | y_{<t}, x) = \\text{softmax}(W_o h_t^{\\text{dec}})"})]}),e.jsx(E,{}),e.jsx(j,{title:"Information Bottleneck",children:e.jsxs("p",{children:["The entire source sentence is compressed into a single fixed-size vector ",e.jsx(t.InlineMath,{math:"c"}),". For long sequences, this bottleneck causes information loss and degraded performance. This limitation directly motivated the invention of ",e.jsx("strong",{children:"attention mechanisms"}),", which allow the decoder to selectively access all encoder hidden states."]})}),e.jsxs(x,{title:"Machine Translation Pipeline",children:[e.jsx("p",{children:'For translating "le chat est noir" to "the cat is black":'}),e.jsxs("ol",{className:"list-decimal ml-6 mt-1 space-y-1",children:[e.jsx("li",{children:"Encoder processes each French token, updating hidden state"}),e.jsxs("li",{children:["Final encoder state ",e.jsx(t.InlineMath,{math:"c"})," summarizes the French sentence"]}),e.jsxs("li",{children:["Decoder receives ",e.jsx(t.InlineMath,{math:"c"})," as initial state and a start token"]}),e.jsx("li",{children:"At each step, decoder predicts next English token and feeds it back as input"})]})]}),e.jsx(p,{title:"Encoder-Decoder in PyTorch",code:`import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        embedded = self.embed(src)
        outputs, (h, c) = self.lstm(embedded)
        return outputs, h, c

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt, h, c):
        embedded = self.embed(tgt)
        output, (h, c) = self.lstm(embedded, (h, c))
        prediction = self.fc(output)
        return prediction, h, c

enc = Encoder(5000, 128, 256)
dec = Decoder(6000, 128, 256)

src = torch.randint(0, 5000, (4, 15))
tgt = torch.randint(0, 6000, (4, 12))

enc_out, h, c = enc(src)
output, _, _ = dec(tgt, h, c)
print(f"Decoder output: {output.shape}")  # (4, 12, 6000)`}),e.jsx(f,{type:"note",title:"Historical Significance",children:e.jsx("p",{children:"The encoder-decoder framework (Cho et al., 2014; Sutskever et al., 2014) was a breakthrough for neural machine translation, replacing phrase-based statistical systems. Combined with attention (Bahdanau et al., 2015), it became the dominant NMT approach until Transformers arrived. The architectural pattern remains foundational in modern sequence models."})})]})}const me=Object.freeze(Object.defineProperty({__proto__:null,default:D},Symbol.toStringTag,{value:"Module"}));function H(){const[a,c]=m.useState(.5),l=6,d=["<s>","the","cat","sat","down","."],r=["<s>","a","cat","sit","down",","];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Teacher Forcing Ratio"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["TF ratio: ",a.toFixed(2),e.jsx("input",{type:"range",min:0,max:1,step:.05,value:a,onChange:i=>c(parseFloat(i.target.value)),className:"w-40 accent-violet-500"})]}),e.jsx("div",{className:"flex gap-1 justify-center",children:Array.from({length:l},(i,s)=>{const n=s/l<a;return e.jsxs("div",{className:"flex flex-col items-center gap-1",children:[e.jsx("div",{className:`px-2 py-1 rounded text-xs font-mono ${n?"bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-300":"bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300"}`,children:n?d[s]:r[s]}),e.jsx("span",{className:"text-xs text-gray-400",children:n?"truth":"pred"})]},s)})}),e.jsx("p",{className:"text-xs text-center mt-2 text-gray-500",children:"Violet = ground truth input (teacher forcing), Orange = model's own prediction"})]})}function V(){const[a,c]=m.useState(3),l=[{tokens:["the","cat"],score:-.8},{tokens:["a","cat"],score:-1.2},{tokens:["the","dog"],score:-1.5},{tokens:["a","dog"],score:-2.1},{tokens:["my","cat"],score:-2.4}];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsxs("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:["Beam Search (width=",a,")"]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Beam width:",e.jsx("input",{type:"range",min:1,max:5,step:1,value:a,onChange:d=>c(parseInt(d.target.value)),className:"w-32 accent-violet-500"})]}),e.jsx("div",{className:"space-y-1",children:l.slice(0,a).map((d,r)=>e.jsxs("div",{className:"flex items-center gap-3",children:[e.jsx("span",{className:"font-mono text-sm text-violet-600 dark:text-violet-400 w-28",children:d.tokens.join(" ")}),e.jsx("div",{className:"flex-1 bg-gray-100 dark:bg-gray-800 rounded h-4 overflow-hidden",children:e.jsx("div",{className:"h-full bg-violet-500 rounded",style:{width:`${Math.exp(d.score)*100}%`}})}),e.jsx("span",{className:"text-xs text-gray-500 w-12 text-right",children:d.score.toFixed(2)})]},r))})]})}function $(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Sequence-to-sequence models combine an encoder and decoder to handle tasks where both input and output are variable-length sequences. Key training and inference techniques include teacher forcing and beam search."}),e.jsxs(h,{title:"Teacher Forcing",children:[e.jsxs("p",{children:["During training, the decoder receives the ",e.jsx("strong",{children:"ground-truth"})," previous token as input rather than its own prediction:"]}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = -\\sum_{t=1}^{T} \\log P(y_t^* | y_1^*, \\ldots, y_{t-1}^*, c)"}),e.jsxs("p",{className:"mt-2",children:["This stabilizes training but creates a mismatch between training (seeing perfect inputs) and inference (seeing its own potentially erroneous predictions), known as ",e.jsx("strong",{children:"exposure bias"}),"."]})]}),e.jsx(H,{}),e.jsxs(h,{title:"Beam Search",children:[e.jsxs("p",{children:["At inference, beam search maintains the top-",e.jsx(t.InlineMath,{math:"k"})," most likely partial sequences at each step:"]}),e.jsx(t.BlockMath,{math:"\\text{score}(y_{1:t}) = \\sum_{i=1}^{t} \\log P(y_i | y_{<i}, c)"}),e.jsxs("p",{className:"mt-2",children:["Length normalization prevents bias toward shorter sequences:",e.jsx(t.InlineMath,{math:"\\text{score}_{\\text{norm}} = \\frac{1}{T^\\alpha} \\sum_{t} \\log P(y_t)"})," with ",e.jsx(t.InlineMath,{math:"\\alpha \\approx 0.6"}),"."]})]}),e.jsx(V,{}),e.jsx(p,{title:"Seq2Seq with Teacher Forcing",code:`import torch
import torch.nn as nn
import random

class Seq2Seq(nn.Module):
    def __init__(self, enc_vocab, dec_vocab, embed_dim, hidden_dim):
        super().__init__()
        self.enc_embed = nn.Embedding(enc_vocab, embed_dim)
        self.dec_embed = nn.Embedding(dec_vocab, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, dec_vocab)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        _, (h, c) = self.encoder(self.enc_embed(src))
        B, T = tgt.shape
        outputs = torch.zeros(B, T, self.fc.out_features, device=src.device)
        dec_input = tgt[:, 0:1]  # <sos> token

        for t in range(T):
            out, (h, c) = self.decoder(self.dec_embed(dec_input), (h, c))
            outputs[:, t:t+1] = self.fc(out)
            if t < T - 1:
                use_tf = random.random() < teacher_forcing_ratio
                dec_input = tgt[:, t+1:t+2] if use_tf else outputs[:, t].argmax(-1, keepdim=True)
        return outputs

model = Seq2Seq(5000, 6000, 128, 256)
src = torch.randint(0, 5000, (4, 15))
tgt = torch.randint(0, 6000, (4, 12))
out = model(src, tgt, teacher_forcing_ratio=0.5)
print(f"Output: {out.shape}")  # (4, 12, 6000)`}),e.jsx(j,{title:"Exposure Bias",children:e.jsx("p",{children:"Scheduled sampling (Bengio et al., 2015) gradually decreases the teacher forcing ratio during training, easing the transition from training to inference. However, it can slow convergence. Alternative approaches include sequence-level training with REINFORCE or minimum risk training."})})]})}const xe=Object.freeze(Object.defineProperty({__proto__:null,default:$},Symbol.toStringTag,{value:"Module"}));function Y(){const[a,c]=m.useState("bahdanau"),l=["le","chat","noir","dort"],d=["the","black","cat","sleeps"],s=a==="bahdanau"?[[.85,.05,.05,.05],[.05,.1,.8,.05],[.05,.75,.1,.1],[.05,.05,.05,.85]]:[[.8,.1,.05,.05],[.05,.08,.82,.05],[.08,.72,.12,.08],[.03,.05,.07,.85]];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Attention Alignment"}),e.jsxs("div",{className:"flex gap-2 mb-3",children:[e.jsx("button",{onClick:()=>c("bahdanau"),className:`px-3 py-1 rounded-lg text-sm ${a==="bahdanau"?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:"Bahdanau (additive)"}),e.jsx("button",{onClick:()=>c("luong"),className:`px-3 py-1 rounded-lg text-sm ${a==="luong"?"bg-violet-500 text-white":"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"}`,children:"Luong (multiplicative)"})]}),e.jsx("div",{className:"flex justify-center",children:e.jsxs("div",{className:"inline-grid gap-0.5",style:{gridTemplateColumns:`60px repeat(${l.length}, 48px)`},children:[e.jsx("div",{}),l.map((n,o)=>e.jsx("div",{className:"text-center text-xs text-gray-500 font-mono",children:n},o)),d.map((n,o)=>e.jsxs(e.Fragment,{children:[e.jsx("div",{className:"text-right pr-2 text-xs text-gray-500 font-mono leading-8",children:n},`l${o}`),s[o].map((g,_)=>e.jsx("div",{className:"w-12 h-8 rounded flex items-center justify-center text-xs font-mono",style:{backgroundColor:`rgba(139, 92, 246, ${g})`,color:g>.5?"white":"#6b7280"},children:g.toFixed(2)},`${o}-${_}`))]}))]})}),e.jsx("p",{className:"text-xs text-center mt-2 text-gray-500",children:"Source (columns) vs Target (rows). Darker = higher attention weight."})]})}function Q(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Attention mechanisms allow the decoder to dynamically focus on different parts of the encoder output at each decoding step, overcoming the information bottleneck of fixed-size context vectors."}),e.jsxs(h,{title:"Bahdanau Attention (Additive)",children:[e.jsx("p",{children:"The alignment score uses a learned feedforward network:"}),e.jsx(t.BlockMath,{math:"e_{t,i} = v^T \\tanh(W_s\\, s_{t-1} + W_h\\, h_i)"}),e.jsx(t.BlockMath,{math:"\\alpha_{t,i} = \\frac{\\exp(e_{t,i})}{\\sum_j \\exp(e_{t,j})}"}),e.jsx(t.BlockMath,{math:"c_t = \\sum_i \\alpha_{t,i}\\, h_i"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"s_{t-1}"})," is the decoder state and ",e.jsx(t.InlineMath,{math:"h_i"})," are encoder outputs. The context ",e.jsx(t.InlineMath,{math:"c_t"})," changes at every decoder step."]})]}),e.jsxs(h,{title:"Luong Attention (Multiplicative)",children:[e.jsx("p",{children:"Luong attention computes alignment with a simpler dot product or bilinear form:"}),e.jsx(t.BlockMath,{math:"e_{t,i} = s_t^T W_a\\, h_i \\quad \\text{(general)}"}),e.jsx(t.BlockMath,{math:"e_{t,i} = s_t^T h_i \\quad \\text{(dot)}"}),e.jsxs("p",{className:"mt-2",children:["Luong attention uses the current decoder state ",e.jsx(t.InlineMath,{math:"s_t"})," (not ",e.jsx(t.InlineMath,{math:"s_{t-1}"}),"), and applies attention after the decoder RNN step rather than before."]})]}),e.jsx(Y,{}),e.jsx(y,{title:"Attention as Soft Alignment",id:"soft-alignment",children:e.jsxs("p",{children:["The attention weights ",e.jsx(t.InlineMath,{math:"\\alpha_{t,i}"})," form a probability distribution over source positions. This acts as a differentiable, soft version of word alignment used in statistical MT. The model learns these alignments end-to-end without explicit alignment supervision."]})}),e.jsx(p,{title:"Bahdanau Attention Module",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs):
        # decoder_state: (B, 1, H), encoder_outputs: (B, S, H)
        scores = self.v(torch.tanh(
            self.W_s(decoder_state) + self.W_h(encoder_outputs)
        ))  # (B, S, 1)
        weights = F.softmax(scores.squeeze(-1), dim=-1)  # (B, S)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)  # (B, 1, H)
        return context, weights

attn = BahdanauAttention(256)
dec_state = torch.randn(4, 1, 256)
enc_out = torch.randn(4, 20, 256)
ctx, weights = attn(dec_state, enc_out)
print(f"Context: {ctx.shape}")    # (4, 1, 256)
print(f"Weights: {weights.shape}")  # (4, 20)
print(f"Weights sum: {weights.sum(-1)}")  # all 1.0`}),e.jsx(f,{type:"note",title:"Impact of Attention",children:e.jsxs("p",{children:["Attention improved BLEU scores on WMT translation by 2-5 points and enabled training on longer sentences. It also provides interpretability through attention weight visualization. The concept of attention evolved into the ",e.jsx("strong",{children:"self-attention"})," mechanism at the core of Transformers, which apply attention within a single sequence rather than across encoder-decoder pairs."]})})]})}const pe=Object.freeze(Object.defineProperty({__proto__:null,default:Q},Symbol.toStringTag,{value:"Module"}));function X(){const[a,c]=m.useState(.3),l=["John","went","to","New","York"],d={he:.3,went:.2,to:.15,the:.1,a:.05},r={John:.5,went:.1,to:.1,New:.2,York:.1},i={};for(const[n,o]of Object.entries(d))i[n]=(i[n]||0)+a*o;for(const[n,o]of Object.entries(r))i[n]=(i[n]||0)+(1-a)*o;const s=Object.entries(i).sort((n,o)=>o[1]-n[1]);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Copy Mechanism in Action"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["p_gen: ",a.toFixed(2)," (generate)  |  1-p_gen: ",(1-a).toFixed(2)," (copy)",e.jsx("input",{type:"range",min:0,max:1,step:.05,value:a,onChange:n=>c(parseFloat(n.target.value)),className:"w-32 accent-violet-500"})]}),e.jsxs("div",{className:"flex gap-2 mb-3 flex-wrap",children:[e.jsx("span",{className:"text-xs text-gray-500",children:"Source:"}),l.map((n,o)=>e.jsx("span",{className:"px-2 py-0.5 rounded bg-violet-100 dark:bg-violet-900/40 text-violet-700 dark:text-violet-300 text-xs font-mono",children:n},o))]}),e.jsx("div",{className:"space-y-1",children:s.slice(0,6).map(([n,o])=>e.jsxs("div",{className:"flex items-center gap-2",children:[e.jsx("span",{className:"w-14 text-xs font-mono text-gray-600 dark:text-gray-400 text-right",children:n}),e.jsx("div",{className:"flex-1 bg-gray-100 dark:bg-gray-800 rounded h-4 overflow-hidden",children:e.jsx("div",{className:"h-full bg-violet-500 rounded",style:{width:`${o*100}%`}})}),e.jsx("span",{className:"text-xs text-gray-500 w-10 text-right",children:o.toFixed(3)})]},n))})]})}function K(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Copy mechanisms and pointer networks extend seq2seq models to handle rare words, named entities, and structured outputs by allowing the decoder to copy tokens directly from the source sequence."}),e.jsxs(h,{title:"Pointer Network",children:[e.jsx("p",{children:"A pointer network (Vinyals et al., 2015) uses attention as a pointer to select elements from the input sequence:"}),e.jsx(t.BlockMath,{math:"P(y_t = x_i) = \\alpha_{t,i} = \\frac{\\exp(e_{t,i})}{\\sum_j \\exp(e_{t,j})}"}),e.jsx("p",{className:"mt-2",children:"Unlike standard seq2seq which outputs from a fixed vocabulary, pointer networks can output any element from the variable-length input, making them ideal for combinatorial optimization problems (e.g., sorting, convex hull, TSP)."})]}),e.jsxs(h,{title:"Copy Mechanism (Pointer-Generator)",children:[e.jsxs("p",{children:["The pointer-generator network (See et al., 2017) combines generation and copying using a soft switch ",e.jsx(t.InlineMath,{math:"p_{\\text{gen}}"}),":"]}),e.jsx(t.BlockMath,{math:"p_{\\text{gen}} = \\sigma(w_c^T c_t + w_s^T s_t + w_x^T x_t + b)"}),e.jsx(t.BlockMath,{math:"P(w) = p_{\\text{gen}} \\, P_{\\text{vocab}}(w) + (1 - p_{\\text{gen}}) \\sum_{i: x_i = w} \\alpha_{t,i}"}),e.jsx("p",{className:"mt-2",children:"This allows the model to generate from the vocabulary or copy from the source via the attention distribution."})]}),e.jsx(X,{}),e.jsxs(x,{title:"Use Cases",children:[e.jsx("p",{children:"Copy mechanisms are critical for:"}),e.jsxs("ul",{className:"list-disc ml-6 mt-1 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Summarization"}),": copying factual details (names, numbers) from the article"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Dialogue"}),": repeating entities mentioned by the user"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Code generation"}),": copying variable names from the context"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Data-to-text"}),": faithfully reproducing values from structured data"]})]})]}),e.jsx(p,{title:"Pointer-Generator Network",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.vocab_proj = nn.Linear(hidden_dim, vocab_size)
        self.p_gen_linear = nn.Linear(hidden_dim * 2 + hidden_dim, 1)

    def forward(self, dec_state, enc_outputs, src_ids):
        # dec_state: (B, H), enc_outputs: (B, S, H), src_ids: (B, S)
        B, S, H = enc_outputs.shape

        # Attention
        dec_exp = dec_state.unsqueeze(1).expand(-1, S, -1)
        scores = self.attn(torch.cat([dec_exp, enc_outputs], -1)).squeeze(-1)
        attn_weights = F.softmax(scores, dim=-1)  # (B, S)

        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)

        # p_gen switch
        p_gen = torch.sigmoid(self.p_gen_linear(
            torch.cat([context, dec_state, dec_state], -1)  # simplified
        ))  # (B, 1)

        # Vocab distribution
        vocab_dist = F.softmax(self.vocab_proj(dec_state), dim=-1) * p_gen

        # Copy distribution
        copy_dist = torch.zeros(B, self.vocab_size, device=dec_state.device)
        copy_dist.scatter_add_(1, src_ids, attn_weights * (1 - p_gen))

        return vocab_dist + copy_dist, attn_weights

model = PointerGenerator(vocab_size=10000, hidden_dim=256)
dec_h = torch.randn(4, 256)
enc_out = torch.randn(4, 20, 256)
src = torch.randint(0, 10000, (4, 20))
probs, attn = model(dec_h, enc_out, src)
print(f"Output dist: {probs.shape}")  # (4, 10000)
print(f"Sum: {probs.sum(-1)}")  # ~1.0`}),e.jsxs(y,{title:"Coverage Mechanism",id:"coverage-mechanism",children:[e.jsx("p",{children:"To prevent repetition in generation, a coverage vector tracks cumulative attention:"}),e.jsx(t.BlockMath,{math:"\\text{cov}_t = \\sum_{t'=0}^{t-1} \\alpha_{t'}"}),e.jsx("p",{children:"A coverage loss penalizes re-attending to already-covered positions:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_{\\text{cov}} = \\sum_i \\min(\\alpha_{t,i}, \\text{cov}_{t,i})"})]}),e.jsx(f,{type:"note",title:"Legacy and Modern Influence",children:e.jsx("p",{children:"While Transformers have largely replaced RNN-based seq2seq, the copy mechanism concept lives on in retrieval-augmented generation and tool-use paradigms in large language models. The idea of selectively copying from a source remains fundamental in modern NLP architectures."})})]})}const ge=Object.freeze(Object.defineProperty({__proto__:null,default:K},Symbol.toStringTag,{value:"Module"}));export{ae as a,se as b,ne as c,re as d,ie as e,oe as f,le as g,de as h,ce as i,he as j,me as k,xe as l,pe as m,ge as n,te as s};

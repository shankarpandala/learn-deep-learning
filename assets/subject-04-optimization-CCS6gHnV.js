import{j as e,r as _}from"./vendor-DpISuAX6.js";import{r as t}from"./vendor-katex-CbWCYdth.js";import{D as v,E as L,P as S,W as I,N as z,T as W,a as A}from"./subject-01-foundations-D0A1VJsr.js";function q(){const[a,j]=_.useState(.9),[l,h]=_.useState(0),i=400,s=250,p=[];let m=3.5,x=3,o=0,c=0;const d=.02;for(let k=0;k<=60;k++){p.push({x:m,y:x});const N=2*m,M=20*x;o=a*o+N,c=a*c+M,m-=d*o,x-=d*c}const r=[];let n=3.5,g=3;for(let k=0;k<=60;k++)r.push({x:n,y:g}),n-=d*2*n,g-=d*20*g;const u=i/8,f=s/7,y=i/2,b=s/2,T=Math.min(l,p.length-1);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Momentum vs Vanilla SGD"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3 flex-wrap",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["β = ",a.toFixed(2),e.jsx("input",{type:"range",min:0,max:.99,step:.01,value:a,onChange:k=>j(parseFloat(k.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Step: ",l,e.jsx("input",{type:"range",min:0,max:60,step:1,value:l,onChange:k=>h(parseInt(k.target.value)),className:"w-28 accent-violet-500"})]})]}),e.jsxs("svg",{width:i,height:s,className:"mx-auto block",children:[e.jsx("line",{x1:0,y1:b,x2:i,y2:b,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("line",{x1:y,y1:0,x2:y,y2:s,stroke:"#d1d5db",strokeWidth:.5}),r.slice(0,T+1).map((k,N,M)=>N>0&&e.jsx("line",{x1:y+M[N-1].x*u,y1:b-M[N-1].y*f,x2:y+k.x*u,y2:b-k.y*f,stroke:"#9ca3af",strokeWidth:1.5,opacity:.6},`v-${N}`)),p.slice(0,T+1).map((k,N,M)=>N>0&&e.jsx("line",{x1:y+M[N-1].x*u,y1:b-M[N-1].y*f,x2:y+k.x*u,y2:b-k.y*f,stroke:"#8b5cf6",strokeWidth:2},`m-${N}`)),e.jsx("circle",{cx:y+p[T].x*u,cy:b-p[T].y*f,r:4,fill:"#8b5cf6"}),e.jsx("circle",{cx:y+r[T].x*u,cy:b-r[T].y*f,r:4,fill:"#9ca3af"})]}),e.jsxs("div",{className:"mt-2 flex justify-center gap-6 text-xs",children:[e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-violet-500"})," Momentum"]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-gray-400"})," Vanilla SGD"]})]})]})}function O(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Momentum accelerates SGD by accumulating an exponentially decaying moving average of past gradients, allowing the optimizer to build up velocity in consistent gradient directions and dampen oscillations."}),e.jsxs(v,{title:"Classical Momentum",children:[e.jsx(t.BlockMath,{math:"v_t = \\beta \\, v_{t-1} + \\nabla_\\theta \\mathcal{L}(\\theta_{t-1})"}),e.jsx(t.BlockMath,{math:"\\theta_t = \\theta_{t-1} - \\alpha \\, v_t"}),e.jsxs("p",{className:"mt-2",children:["Here ",e.jsx(t.InlineMath,{math:"\\beta \\in [0,1)"})," is the momentum coefficient (typically 0.9) and ",e.jsx(t.InlineMath,{math:"v_t"})," is the velocity vector."]})]}),e.jsx(q,{}),e.jsxs(L,{title:"Exponential Moving Average Intuition",children:[e.jsxs("p",{children:["Expanding the velocity recursion for ",e.jsx(t.InlineMath,{math:"\\beta = 0.9"}),":"]}),e.jsx(t.BlockMath,{math:"v_t = g_t + 0.9\\,g_{t-1} + 0.81\\,g_{t-2} + 0.729\\,g_{t-3} + \\cdots"}),e.jsxs("p",{children:["The effective window is roughly ",e.jsx(t.InlineMath,{math:"1/(1-\\beta) = 10"})," steps. Gradients from more than ~10 steps ago contribute negligibly."]})]}),e.jsx(S,{title:"Momentum SGD in PyTorch",code:`import torch
import torch.optim as optim
import torch.nn as nn

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

# Classical momentum with β=0.9
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training step
x = torch.randn(32, 784)
target = torch.randint(0, 10, (32,))
loss = nn.CrossEntropyLoss()(model(x), target)
loss.backward()
optimizer.step()
optimizer.zero_grad()

print(f"Loss: {loss.item():.4f}")`}),e.jsx(I,{title:"Momentum Can Overshoot",children:e.jsxs("p",{children:["High momentum values (",e.jsx(t.InlineMath,{math:"\\beta > 0.95"}),") can cause the optimizer to overshoot minima, especially with large learning rates. If training becomes unstable, try reducing ",e.jsx(t.InlineMath,{math:"\\beta"})," or the learning rate."]})}),e.jsx(z,{type:"note",title:"Why Momentum Works",children:e.jsx("p",{children:"In ravine-shaped loss landscapes (common in deep learning), gradients oscillate across the narrow dimension and are consistent along the long dimension. Momentum cancels out oscillations and amplifies the consistent direction, leading to faster convergence."})})]})}const ye=Object.freeze(Object.defineProperty({__proto__:null,default:O},Symbol.toStringTag,{value:"Module"}));function P(){const[a,j]=_.useState(0),l=400,h=250,i=l/2,s=h/2,p=40,m=40,x=.9,o=.02,c=[];let d=3,r=3,n=0,g=0;for(let w=0;w<=50;w++){c.push({x:d,y:r});const R=2*d,B=18*r;n=x*n+R,g=x*g+B,d-=o*n,r-=o*g}const u=[];let f=3,y=3,b=0,T=0;for(let w=0;w<=50;w++){u.push({x:f,y});const R=f-o*x*b,B=y-o*x*T,C=2*R,G=18*B;b=x*b+C,T=x*T+G,f-=o*b,y-=o*T}const k=Math.min(a,50),N=w=>i+w*p,M=w=>s-w*m;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Nesterov vs Classical Momentum"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Step: ",a,e.jsx("input",{type:"range",min:0,max:50,step:1,value:a,onChange:w=>j(parseInt(w.target.value)),className:"w-36 accent-violet-500"})]}),e.jsxs("svg",{width:l,height:h,className:"mx-auto block",children:[e.jsx("line",{x1:0,y1:s,x2:l,y2:s,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("line",{x1:i,y1:0,x2:i,y2:h,stroke:"#d1d5db",strokeWidth:.5}),c.slice(0,k+1).map((w,R,B)=>R>0&&e.jsx("line",{x1:N(B[R-1].x),y1:M(B[R-1].y),x2:N(w.x),y2:M(w.y),stroke:"#9ca3af",strokeWidth:1.5},`c-${R}`)),u.slice(0,k+1).map((w,R,B)=>R>0&&e.jsx("line",{x1:N(B[R-1].x),y1:M(B[R-1].y),x2:N(w.x),y2:M(w.y),stroke:"#8b5cf6",strokeWidth:2},`n-${R}`)),e.jsx("circle",{cx:N(c[k].x),cy:M(c[k].y),r:4,fill:"#9ca3af"}),e.jsx("circle",{cx:N(u[k].x),cy:M(u[k].y),r:4,fill:"#8b5cf6"}),e.jsx("circle",{cx:N(0),cy:M(0),r:5,fill:"#f97316",opacity:.6})]}),e.jsxs("div",{className:"mt-2 flex justify-center gap-6 text-xs",children:[e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-violet-500"})," Nesterov"]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-gray-400"})," Classical"]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-2 h-2 rounded-full bg-orange-500"})," Minimum"]})]})]})}function D(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Nesterov Accelerated Gradient (NAG) improves on classical momentum by evaluating the gradient at a “look-ahead” position, providing a corrective factor that reduces overshooting."}),e.jsxs(v,{title:"Nesterov Accelerated Gradient",children:[e.jsx(t.BlockMath,{math:"v_t = \\beta \\, v_{t-1} + \\nabla_\\theta \\mathcal{L}(\\theta_{t-1} - \\alpha \\beta \\, v_{t-1})"}),e.jsx(t.BlockMath,{math:"\\theta_t = \\theta_{t-1} - \\alpha \\, v_t"}),e.jsxs("p",{className:"mt-2",children:["The key difference: the gradient is computed at the anticipated future position ",e.jsx(t.InlineMath,{math:"\\theta - \\alpha \\beta v"})," rather than the current position."]})]}),e.jsx(P,{}),e.jsxs(W,{title:"Convergence Advantage",id:"nag-convergence",children:[e.jsxs("p",{children:["For ",e.jsx(t.InlineMath,{math:"L"}),"-smooth convex functions, Nesterov momentum achieves:"]}),e.jsx(t.BlockMath,{math:"f(\\theta_t) - f(\\theta^*) \\leq O\\!\\left(\\frac{1}{t^2}\\right)"}),e.jsxs("p",{children:["compared to ",e.jsx(t.InlineMath,{math:"O(1/t)"})," for classical gradient descent, making it an optimal first-order method by Nesterov's lower bound."]})]}),e.jsx(L,{title:"Look-Ahead Intuition",children:e.jsx("p",{children:"Think of a ball rolling downhill with momentum. Classical momentum checks the slope at the current position. Nesterov first rolls the ball forward by its momentum, then checks the slope at the new position. This look-ahead acts as a correction that prevents overshooting valleys."})}),e.jsx(S,{title:"Nesterov Momentum in PyTorch",code:`import torch
import torch.optim as optim
import torch.nn as nn

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

# Nesterov momentum — just add nesterov=True
optimizer = optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9, nesterov=True
)

x = torch.randn(32, 784)
target = torch.randint(0, 10, (32,))
loss = nn.CrossEntropyLoss()(model(x), target)
loss.backward()
optimizer.step()
optimizer.zero_grad()
print(f"Loss: {loss.item():.4f}")`}),e.jsx(z,{type:"note",title:"Practical Recommendations",children:e.jsx("p",{children:"In practice, Nesterov momentum gives a modest but consistent improvement over classical momentum. It is the default choice for SGD in many frameworks. When using adaptive methods like Adam, the Nesterov variant (NAdam) can also be beneficial."})})]})}const fe=Object.freeze(Object.defineProperty({__proto__:null,default:D},Symbol.toStringTag,{value:"Module"}));function F(){const[a,j]=_.useState("convex"),h=[{setting:"convex",method:"GD",rate:"O(1/t)",optimal:!1},{setting:"convex",method:"Nesterov GD",rate:"O(1/t²)",optimal:!0},{setting:"convex",method:"SGD",rate:"O(1/√t)",optimal:!0},{setting:"strongly-convex",method:"GD",rate:"O(exp(-t/κ))",optimal:!1},{setting:"strongly-convex",method:"Nesterov GD",rate:"O(exp(-t/√κ))",optimal:!0},{setting:"strongly-convex",method:"SGD",rate:"O(1/t)",optimal:!0},{setting:"non-convex",method:"SGD",rate:"O(1/√t) to ε-stationary",optimal:!1}].filter(i=>i.setting===a);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-2 text-base font-bold text-gray-800 dark:text-gray-200",children:"Convergence Rate Comparison"}),e.jsx("div",{className:"flex gap-2 mb-3",children:["convex","strongly-convex","non-convex"].map(i=>e.jsx("button",{onClick:()=>j(i),className:`px-3 py-1 rounded text-xs font-medium transition-colors ${a===i?"bg-violet-500 text-white":"bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"}`,children:i},i))}),e.jsxs("table",{className:"w-full text-sm",children:[e.jsx("thead",{children:e.jsxs("tr",{className:"text-left text-gray-500 dark:text-gray-400",children:[e.jsx("th",{className:"pb-1",children:"Method"}),e.jsx("th",{className:"pb-1",children:"Rate"}),e.jsx("th",{className:"pb-1",children:"Optimal?"})]})}),e.jsx("tbody",{children:h.map((i,s)=>e.jsxs("tr",{className:"border-t border-gray-100 dark:border-gray-800",children:[e.jsx("td",{className:"py-1 text-gray-700 dark:text-gray-300",children:i.method}),e.jsx("td",{className:"py-1 font-mono text-violet-600 dark:text-violet-400",children:i.rate}),e.jsx("td",{className:"py-1",children:i.optimal?"✓":"—"})]},s))})]})]})}function V(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Understanding convergence rates helps us choose optimizers and set expectations. The rates differ significantly between convex, strongly convex, and non-convex settings."}),e.jsxs(v,{title:"Key Assumptions",children:[e.jsx("p",{children:"Convergence proofs typically require:"}),e.jsx(t.BlockMath,{math:"L\\text{-smooth: } \\|\\nabla f(x) - \\nabla f(y)\\| \\leq L\\|x - y\\|"}),e.jsx(t.BlockMath,{math:"\\mu\\text{-strongly convex: } f(y) \\geq f(x) + \\nabla f(x)^T(y-x) + \\frac{\\mu}{2}\\|y-x\\|^2"}),e.jsxs("p",{className:"mt-2",children:["The condition number ",e.jsx(t.InlineMath,{math:"\\kappa = L/\\mu"})," governs how hard the problem is."]})]}),e.jsx(F,{}),e.jsxs(W,{title:"SGD Convergence (Convex Case)",id:"sgd-convex-rate",children:[e.jsxs("p",{children:["For an ",e.jsx(t.InlineMath,{math:"L"}),"-smooth convex function with bounded variance",e.jsx(t.InlineMath,{math:"\\sigma^2"}),", SGD with step size ",e.jsx(t.InlineMath,{math:"\\alpha_t = \\alpha_0 / \\sqrt{t}"})," satisfies:"]}),e.jsx(t.BlockMath,{math:"\\mathbb{E}[f(\\bar{\\theta}_T)] - f(\\theta^*) \\leq O\\!\\left(\\frac{\\|\\theta_0 - \\theta^*\\|^2}{T} + \\frac{\\sigma}{\\sqrt{T}}\\right)"})]}),e.jsxs(A,{title:"Sketch: SGD Convergence Bound",children:[e.jsxs("p",{children:["Starting from ",e.jsx(t.InlineMath,{math:"L"}),"-smoothness and taking expectations over stochastic gradients:"]}),e.jsx(t.BlockMath,{math:"\\mathbb{E}[\\|\\theta_{t+1} - \\theta^*\\|^2] \\leq \\|\\theta_t - \\theta^*\\|^2 - 2\\alpha_t(f(\\theta_t) - f(\\theta^*)) + \\alpha_t^2 \\sigma^2"}),e.jsxs("p",{children:["Summing over ",e.jsx(t.InlineMath,{math:"t = 0, \\ldots, T-1"})," and rearranging with decreasing step sizes yields the ",e.jsx(t.InlineMath,{math:"O(1/\\sqrt{T})"})," rate."]})]}),e.jsx(L,{title:"Practical Implication",children:e.jsxs("p",{children:["The ",e.jsx(t.InlineMath,{math:"O(1/\\sqrt{T})"})," rate means that to halve the error, you need 4x more iterations. To go from ",e.jsx(t.InlineMath,{math:"10^{-2}"})," to ",e.jsx(t.InlineMath,{math:"10^{-4}"})," error requires 10,000x more steps — motivating better optimizers and schedules."]})}),e.jsx(S,{title:"Tracking Convergence Empirically",code:`import torch
import torch.nn as nn

# Simple convex problem: linear regression
torch.manual_seed(42)
X = torch.randn(200, 10)
w_true = torch.randn(10, 1)
y = X @ w_true + 0.1 * torch.randn(200, 1)

w = torch.randn(10, 1, requires_grad=True)
losses = []
for t in range(1, 501):
    loss = ((X @ w - y) ** 2).mean()
    loss.backward()
    losses.append(loss.item())
    with torch.no_grad():
        w -= (0.1 / t**0.5) * w.grad   # decaying lr
        w.grad.zero_()

print(f"Final loss: {losses[-1]:.6f}")
print(f"Loss ratio (step 125 vs 500): {losses[124]/losses[-1]:.2f}")
# Expect ratio ~2 for O(1/sqrt(t)) convergence`}),e.jsx(z,{type:"note",title:"Non-Convex Reality",children:e.jsxs("p",{children:["Deep learning losses are non-convex, so we can only guarantee convergence to stationary points (",e.jsx(t.InlineMath,{math:"\\|\\nabla f\\| \\leq \\epsilon"}),"). In practice, SGD noise helps escape saddle points, and most local minima in overparameterized networks generalize well."]})})]})}const je=Object.freeze(Object.defineProperty({__proto__:null,default:V},Symbol.toStringTag,{value:"Module"}));function H(){const[a,j]=_.useState(30),[l,h]=_.useState(.9),i=400,s=200,p=[],m=[];let x=0,o=0;const c=.1;for(let g=1;g<=60;g++){const u=1+.3*Math.sin(g*.5);x+=u*u,o=l*o+(1-l)*u*u,p.push(c/(Math.sqrt(x)+1e-8)),m.push(c/(Math.sqrt(o)+1e-8))}const d=Math.max(...m,...p)*1.1,r=i/62,n=s/d;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Effective Learning Rate Over Time"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3 flex-wrap",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Steps: ",a,e.jsx("input",{type:"range",min:5,max:60,step:1,value:a,onChange:g=>j(parseInt(g.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["ρ = ",l.toFixed(2),e.jsx("input",{type:"range",min:.5,max:.999,step:.01,value:l,onChange:g=>h(parseFloat(g.target.value)),className:"w-28 accent-violet-500"})]})]}),e.jsxs("svg",{width:i,height:s,className:"mx-auto block",children:[e.jsx("line",{x1:0,y1:s-1,x2:i,y2:s-1,stroke:"#d1d5db",strokeWidth:.5}),p.slice(0,a).map((g,u,f)=>u>0&&e.jsx("line",{x1:u*r,y1:s-f[u-1]*n,x2:(u+1)*r,y2:s-g*n,stroke:"#9ca3af",strokeWidth:1.8},`a-${u}`)),m.slice(0,a).map((g,u,f)=>u>0&&e.jsx("line",{x1:u*r,y1:s-f[u-1]*n,x2:(u+1)*r,y2:s-g*n,stroke:"#8b5cf6",strokeWidth:2},`r-${u}`))]}),e.jsxs("div",{className:"mt-2 flex justify-center gap-6 text-xs",children:[e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-violet-500"})," RMSProp"]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-gray-400"})," AdaGrad"]})]})]})}function U(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"AdaGrad introduced per-parameter adaptive learning rates by accumulating squared gradients. RMSProp fixes AdaGrad's aggressive decay by using an exponential moving average instead."}),e.jsxs(v,{title:"AdaGrad",children:[e.jsx(t.BlockMath,{math:"G_t = G_{t-1} + g_t \\odot g_t"}),e.jsx(t.BlockMath,{math:"\\theta_t = \\theta_{t-1} - \\frac{\\alpha}{\\sqrt{G_t} + \\epsilon} \\odot g_t"}),e.jsx("p",{className:"mt-2",children:"Each parameter gets its own effective learning rate that decreases as its cumulative gradient grows. Parameters with sparse or small gradients retain larger learning rates."})]}),e.jsx(I,{title:"AdaGrad's Learning Rate Decay",children:e.jsxs("p",{children:["Since ",e.jsx(t.InlineMath,{math:"G_t"})," only grows, the effective learning rate monotonically decreases to zero. For non-convex problems (deep learning), this causes premature convergence — training effectively stops too early."]})}),e.jsxs(v,{title:"RMSProp (Hinton, 2012)",children:[e.jsx(t.BlockMath,{math:"v_t = \\rho \\, v_{t-1} + (1 - \\rho)\\, g_t^2"}),e.jsx(t.BlockMath,{math:"\\theta_t = \\theta_{t-1} - \\frac{\\alpha}{\\sqrt{v_t} + \\epsilon} \\odot g_t"}),e.jsxs("p",{className:"mt-2",children:["The decay factor ",e.jsx(t.InlineMath,{math:"\\rho"})," (typically 0.9 or 0.99) keeps a moving window of recent gradient magnitudes, preventing the denominator from growing unboundedly."]})]}),e.jsx(H,{}),e.jsx(L,{title:"Sparse Features",children:e.jsx("p",{children:"In NLP with one-hot embeddings, common words get frequent gradient updates while rare words get few. AdaGrad/RMSProp automatically give rare words larger effective learning rates, helping them learn from limited data."})}),e.jsx(S,{title:"AdaGrad & RMSProp in PyTorch",code:`import torch
import torch.optim as optim
import torch.nn as nn

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
x = torch.randn(32, 784)
target = torch.randint(0, 10, (32,))

# AdaGrad
opt_ada = optim.Adagrad(model.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss()(model(x), target)
loss.backward(); opt_ada.step(); opt_ada.zero_grad()
print(f"AdaGrad loss: {loss.item():.4f}")

# RMSProp — usually preferred for deep learning
opt_rms = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)
loss = nn.CrossEntropyLoss()(model(x), target)
loss.backward(); opt_rms.step(); opt_rms.zero_grad()
print(f"RMSProp loss: {loss.item():.4f}")`}),e.jsx(z,{type:"note",title:"AdaGrad Still Shines for Sparse Problems",children:e.jsx("p",{children:"Despite its limitations in deep learning, AdaGrad remains the optimizer of choice for sparse problems like recommendation systems and click-through-rate prediction, where its decaying rate naturally handles frequently occurring features."})})]})}const be=Object.freeze(Object.defineProperty({__proto__:null,default:U},Symbol.toStringTag,{value:"Module"}));function E(){const[a,j]=_.useState(.9),l=380,h=180,i=20,s=[],p=[];let m=0;const x=1;for(let d=1;d<=i;d++){m=a*m+(1-a)*x;const r=m/(1-Math.pow(a,d));s.push(m),p.push(r)}const o=l/(i+2),c=h/1.5;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Bias Correction Effect"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["β₁ = ",a.toFixed(2),e.jsx("input",{type:"range",min:.5,max:.999,step:.01,value:a,onChange:d=>j(parseFloat(d.target.value)),className:"w-32 accent-violet-500"})]}),e.jsxs("svg",{width:l,height:h,className:"mx-auto block",children:[e.jsx("line",{x1:o,y1:h-x*c,x2:l,y2:h-x*c,stroke:"#d1d5db",strokeWidth:.8,strokeDasharray:"4,4"}),e.jsx("text",{x:l-60,y:h-x*c-5,fill:"#9ca3af",fontSize:10,children:"true mean"}),s.map((d,r,n)=>r>0&&e.jsx("line",{x1:r*o+o,y1:h-n[r-1]*c,x2:(r+1)*o+o,y2:h-d*c,stroke:"#9ca3af",strokeWidth:1.5},`u-${r}`)),p.map((d,r,n)=>r>0&&e.jsx("line",{x1:r*o+o,y1:h-n[r-1]*c,x2:(r+1)*o+o,y2:h-d*c,stroke:"#8b5cf6",strokeWidth:2},`c-${r}`))]}),e.jsxs("div",{className:"mt-2 flex justify-center gap-6 text-xs",children:[e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-violet-500"})," Bias-corrected"]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-gray-400"})," Uncorrected"]})]})]})}function $(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Adam combines the momentum idea (first moment) with adaptive learning rates (second moment), making it the most widely used optimizer in deep learning. AdamW improves it with properly decoupled weight decay."}),e.jsxs(v,{title:"Adam (Adaptive Moment Estimation)",children:[e.jsx(t.BlockMath,{math:"m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t \\quad \\text{(first moment)}"}),e.jsx(t.BlockMath,{math:"v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2 \\quad \\text{(second moment)}"}),e.jsx(t.BlockMath,{math:"\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\quad \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t} \\quad \\text{(bias correction)}"}),e.jsx(t.BlockMath,{math:"\\theta_t = \\theta_{t-1} - \\frac{\\alpha}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t"})]}),e.jsx(E,{}),e.jsxs(v,{title:"AdamW (Decoupled Weight Decay)",children:[e.jsx(t.BlockMath,{math:"\\theta_t = \\theta_{t-1} - \\alpha\\!\\left(\\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon} + \\lambda\\,\\theta_{t-1}\\right)"}),e.jsx("p",{className:"mt-2",children:"In Adam with L2 regularization, the penalty gradient is also adapted, weakening it for parameters with large gradients. AdamW applies weight decay directly to the parameters, outside the adaptive mechanism, giving proper regularization."})]}),e.jsxs(W,{title:"Default Hyperparameters",id:"adam-defaults",children:[e.jsx("p",{children:"The recommended defaults from the original paper (Kingma & Ba, 2015):"}),e.jsx(t.BlockMath,{math:"\\alpha = 0.001, \\quad \\beta_1 = 0.9, \\quad \\beta_2 = 0.999, \\quad \\epsilon = 10^{-8}"}),e.jsxs("p",{children:["For AdamW, typical weight decay ",e.jsx(t.InlineMath,{math:"\\lambda \\in [0.01, 0.1]"}),". Many LLM training runs use ",e.jsx(t.InlineMath,{math:"\\beta_2 = 0.95"})," for stability."]})]}),e.jsx(L,{title:"Adam vs AdamW on Regularized Models",children:e.jsxs("p",{children:["With ",e.jsx(t.InlineMath,{math:"\\lambda = 0.01"})," weight decay on a Transformer, AdamW typically achieves 1-3% better validation accuracy because the regularization is not distorted by the adaptive scaling. This difference grows with model size."]})}),e.jsx(S,{title:"Adam & AdamW in PyTorch",code:`import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
x, target = torch.randn(32, 784), torch.randint(0, 10, (32,))

# AdamW — preferred for most tasks
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01
)

loss = nn.CrossEntropyLoss()(model(x), target)
loss.backward()
optimizer.step()
optimizer.zero_grad()
print(f"Loss: {loss.item():.4f}")

# Check parameter norms — AdamW regularizes these
norms = [p.norm().item() for p in model.parameters()]
print(f"Param norms: {[f'{n:.3f}' for n in norms]}")`}),e.jsx(I,{title:"Adam Can Generalize Poorly",children:e.jsx("p",{children:"Adam sometimes converges to sharper minima than SGD with momentum, leading to worse generalization. AdamW mitigates this, and combining Adam with learning rate warmup and cosine decay further helps. For vision tasks, SGD+momentum often still wins."})}),e.jsx(z,{type:"note",title:"When to Use Adam vs SGD",children:e.jsxs("p",{children:[e.jsx("strong",{children:"Adam/AdamW"}),": Transformers, NLP, generative models, quick prototyping.",e.jsx("strong",{children:" SGD+momentum"}),": CNNs for image classification (often better generalization with proper tuning). When in doubt, start with AdamW."]})})]})}const ve=Object.freeze(Object.defineProperty({__proto__:null,default:$},Symbol.toStringTag,{value:"Module"}));function Z(){const[a,j]=_.useState(1),l=360,h=160,i=a/(Math.sqrt(a*a)+1e-8),s=Math.sign(a),p=a,m=Math.max(Math.abs(i),Math.abs(s),Math.abs(p))*1.2,x=80,o=30,c=60,d=(h-40)/m,r=[{label:"SGD",value:p,color:"#9ca3af"},{label:"Adam",value:i,color:"#f97316"},{label:"Lion",value:s,color:"#8b5cf6"}];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Update Magnitude Comparison"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Gradient magnitude: ",a.toFixed(1),e.jsx("input",{type:"range",min:.1,max:5,step:.1,value:a,onChange:n=>j(parseFloat(n.target.value)),className:"w-32 accent-violet-500"})]}),e.jsxs("svg",{width:l,height:h,className:"mx-auto block",children:[e.jsx("line",{x1:0,y1:h-20,x2:l,y2:h-20,stroke:"#d1d5db",strokeWidth:.5}),r.map((n,g)=>{const u=c+g*(x+o),f=Math.abs(n.value)*d;return e.jsxs("g",{children:[e.jsx("rect",{x:u,y:h-20-f,width:x,height:f,fill:n.color,rx:4,opacity:.8}),e.jsx("text",{x:u+x/2,y:h-5,textAnchor:"middle",fill:"#6b7280",fontSize:11,children:n.label}),e.jsx("text",{x:u+x/2,y:h-25-f,textAnchor:"middle",fill:n.color,fontSize:10,children:n.value.toFixed(2)})]},n.label)})]})]})}function X(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Recent optimizers push beyond Adam with sign-based updates (Lion), second-order curvature information (Sophia), and other innovations that reduce memory or improve convergence."}),e.jsxs(v,{title:"Lion (EvoLved Sign Momentum)",children:[e.jsx(t.BlockMath,{math:"u_t = \\text{sign}(\\beta_1 m_{t-1} + (1 - \\beta_1) g_t)"}),e.jsx(t.BlockMath,{math:"m_t = \\beta_2 m_{t-1} + (1 - \\beta_2) g_t"}),e.jsx(t.BlockMath,{math:"\\theta_t = \\theta_{t-1} - \\alpha\\,(u_t + \\lambda\\,\\theta_{t-1})"}),e.jsxs("p",{className:"mt-2",children:["Lion uses only the ",e.jsx("strong",{children:"sign"})," of the interpolated momentum, producing uniform magnitude updates. Discovered via program search by Google Brain (2023)."]})]}),e.jsx(Z,{}),e.jsxs(v,{title:"Sophia (Second-Order Clipped)",children:[e.jsx(t.BlockMath,{math:"h_t \\approx \\text{diag}(\\nabla^2 f(\\theta_t)) \\quad \\text{(Hessian diagonal estimate)}"}),e.jsx(t.BlockMath,{math:"\\theta_t = \\theta_{t-1} - \\alpha \\cdot \\text{clip}\\!\\left(\\frac{m_t}{h_t}, \\rho\\right)"}),e.jsx("p",{className:"mt-2",children:"Sophia uses a diagonal Hessian estimate for per-parameter preconditioning, clipped to prevent instability. The Hessian can be estimated via Hutchinson's method."})]}),e.jsxs(W,{title:"Memory Comparison",id:"optimizer-memory",children:[e.jsx("p",{children:"Optimizer state memory per parameter:"}),e.jsx(t.BlockMath,{math:"\\text{SGD: 1 float} \\quad \\text{Adam: 2 floats} \\quad \\text{Lion: 1 float} \\quad \\text{Sophia: 2 floats}"}),e.jsx("p",{children:"Lion saves ~50% optimizer memory vs Adam, which is significant for LLMs where optimizer states can be 2x the model size."})]}),e.jsx(L,{title:"Lion Hyperparameter Tuning",children:e.jsxs("p",{children:["Lion typically requires 3-10x smaller learning rate than Adam. For a model trained with Adam at ",e.jsx(t.InlineMath,{math:"\\alpha = 10^{-4}"}),", try Lion with ",e.jsx(t.InlineMath,{math:"\\alpha = 10^{-5}"})," to ",e.jsx(t.InlineMath,{math:"3 \\times 10^{-5}"}),". Weight decay should be 3-10x larger than Adam's setting."]})}),e.jsx(S,{title:"Implementing Lion from Scratch",code:`import torch
from torch.optim import Optimizer

class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                b1, b2 = group['betas']

                # Sign update with interpolated momentum
                update = torch.sign(exp_avg * b1 + grad * (1 - b1))
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(update, alpha=-group['lr'])

                # Update momentum (different beta for tracking)
                exp_avg.mul_(b2).add_(grad, alpha=1 - b2)

# Usage
model = torch.nn.Linear(128, 10)
opt = Lion(model.parameters(), lr=3e-5, weight_decay=0.1)
print("Lion optimizer created successfully")`}),e.jsx(I,{title:"Modern Optimizers Require Careful Tuning",children:e.jsx("p",{children:"Lion and Sophia do not share Adam's hyperparameter ranges. Directly copying Adam's learning rate will fail. Always do a learning rate sweep when switching optimizers."})}),e.jsx(z,{type:"note",title:"The Optimizer Landscape in 2024+",children:e.jsx("p",{children:"AdamW remains the default for most practitioners. Lion shows promise for large-scale vision and language models with memory constraints. Sophia can be 2x faster for LLM pretraining but adds complexity. Start with AdamW and explore alternatives when needed."})})]})}const _e=Object.freeze(Object.defineProperty({__proto__:null,default:X},Symbol.toStringTag,{value:"Module"}));function K(){const[a,j]=_.useState(10),[l,h]=_.useState("step"),i=400,s=180,p=100,m=1,x=[];for(let r=0;r<p;r++){let n;if(r<a)n=m*(r+1)/a;else if(l==="step")n=r<50?m:r<75?m*.1:m*.01;else{const g=Math.exp(-.03*(r-a));n=m*g}x.push(n)}const o=i/p,c=(s-30)/(m*1.1),d=x.map((r,n)=>`${n===0?"M":"L"}${n*o},${s-20-r*c}`).join(" ");return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Learning Rate Schedule"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3 flex-wrap",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Warmup: ",a,e.jsx("input",{type:"range",min:0,max:30,step:1,value:a,onChange:r=>j(parseInt(r.target.value)),className:"w-28 accent-violet-500"})]}),e.jsx("div",{className:"flex gap-2",children:["step","exponential"].map(r=>e.jsx("button",{onClick:()=>h(r),className:`px-3 py-1 rounded text-xs font-medium ${l===r?"bg-violet-500 text-white":"bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"}`,children:r},r))})]}),e.jsxs("svg",{width:i,height:s,className:"mx-auto block",children:[e.jsx("line",{x1:0,y1:s-20,x2:i,y2:s-20,stroke:"#d1d5db",strokeWidth:.5}),a>0&&e.jsx("line",{x1:a*o,y1:0,x2:a*o,y2:s-20,stroke:"#f97316",strokeWidth:.8,strokeDasharray:"3,3"}),e.jsx("path",{d,fill:"none",stroke:"#8b5cf6",strokeWidth:2}),a>0&&e.jsx("text",{x:a*o+3,y:12,fill:"#f97316",fontSize:9,children:"warmup end"})]})]})}function Q(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Learning rate scheduling is crucial for training stability and final performance. Warmup prevents early instability, while decay ensures convergence to good minima."}),e.jsxs(v,{title:"Linear Warmup",children:[e.jsx(t.BlockMath,{math:"\\alpha_t = \\alpha_{\\max} \\cdot \\frac{t}{T_{\\text{warmup}}}, \\quad t \\leq T_{\\text{warmup}}"}),e.jsx("p",{className:"mt-2",children:"Gradients are noisy and poorly conditioned early in training. Warmup lets statistics in Adam/BatchNorm stabilize before applying the full learning rate."})]}),e.jsxs(v,{title:"Step & Exponential Decay",children:[e.jsx(t.BlockMath,{math:"\\text{Step: } \\alpha_t = \\alpha_0 \\cdot \\gamma^{\\lfloor t / s \\rfloor}"}),e.jsx(t.BlockMath,{math:"\\text{Exponential: } \\alpha_t = \\alpha_0 \\cdot e^{-\\lambda t}"}),e.jsxs("p",{className:"mt-2",children:["Step decay drops the learning rate by factor ",e.jsx(t.InlineMath,{math:"\\gamma"})," every",e.jsx(t.InlineMath,{math:"s"})," epochs. Exponential decay provides smoother reduction."]})]}),e.jsx(K,{}),e.jsx(L,{title:"Why Warmup Helps Adam",children:e.jsxs("p",{children:["At step 1, Adam's bias-corrected second moment ",e.jsx(t.InlineMath,{math:"\\hat{v}_1 = g_1^2"})," is based on a single gradient, making the adaptive ratio highly unreliable. With warmup, by the time the full learning rate is reached, the moment estimates have accumulated enough data to be meaningful."]})}),e.jsx(S,{title:"Warmup + Step Decay Schedule",code:`import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

model = nn.Linear(128, 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

warmup_steps = 1000
total_steps = 50000

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps            # linear warmup
    elif step < 30000:
        return 1.0                             # constant
    elif step < 40000:
        return 0.1                             # first drop
    else:
        return 0.01                            # second drop

scheduler = LambdaLR(optimizer, lr_lambda)

# Training loop pattern
for step in range(100):
    loss = model(torch.randn(8, 128)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

print(f"Final LR: {scheduler.get_last_lr()[0]:.6f}")`}),e.jsx(I,{title:"Step Scheduler Before or After Optimizer?",children:e.jsxs("p",{children:["In PyTorch, always call ",e.jsx("code",{children:"scheduler.step()"})," after ",e.jsx("code",{children:"optimizer.step()"}),". Calling it before can skip the first learning rate value and lead to unexpected behavior."]})}),e.jsx(z,{type:"note",title:"Practical Guidelines",children:e.jsxs("p",{children:[e.jsx("strong",{children:"Warmup duration"}),": 1-5% of total training steps for Transformers, less for CNNs. ",e.jsx("strong",{children:"Decay"}),": step decay at 30% and 60% of training is a classic recipe for image classification. For language models, cosine decay is more common."]})})]})}const ke=Object.freeze(Object.defineProperty({__proto__:null,default:Q},Symbol.toStringTag,{value:"Module"}));function Y(){const[a,j]=_.useState(1),[l,h]=_.useState(0),i=400,s=180,p=120,m=1,x=[],o=Math.floor(p/a);for(let n=0;n<p;n++){const g=n%o,u=l+.5*(m-l)*(1+Math.cos(Math.PI*g/o));x.push(u)}const c=i/p,d=(s-30)/(m*1.1),r=x.map((n,g)=>`${g===0?"M":"L"}${g*c},${s-20-n*d}`).join(" ");return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Cosine Annealing with Warm Restarts"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3 flex-wrap",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Restarts: ",a,e.jsx("input",{type:"range",min:1,max:6,step:1,value:a,onChange:n=>j(parseInt(n.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Min LR: ",l.toFixed(2),e.jsx("input",{type:"range",min:0,max:.3,step:.01,value:l,onChange:n=>h(parseFloat(n.target.value)),className:"w-28 accent-violet-500"})]})]}),e.jsxs("svg",{width:i,height:s,className:"mx-auto block",children:[e.jsx("line",{x1:0,y1:s-20,x2:i,y2:s-20,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("path",{d:r,fill:"none",stroke:"#8b5cf6",strokeWidth:2}),a>1&&Array.from({length:a-1},(n,g)=>e.jsx("line",{x1:(g+1)*o*c,y1:0,x2:(g+1)*o*c,y2:s-20,stroke:"#f97316",strokeWidth:.8,strokeDasharray:"3,3"},g))]}),e.jsxs("div",{className:"mt-2 text-center text-xs text-gray-500 dark:text-gray-400",children:["Period length: ",o," steps ",a>1&&"| Orange lines = restart points"]})]})}function J(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Cosine annealing smoothly decays the learning rate following a cosine curve. Combined with warm restarts (SGDR), it can escape local minima and build snapshot ensembles."}),e.jsxs(v,{title:"Cosine Annealing Schedule",children:[e.jsx(t.BlockMath,{math:"\\alpha_t = \\alpha_{\\min} + \\frac{1}{2}(\\alpha_{\\max} - \\alpha_{\\min})\\left(1 + \\cos\\!\\left(\\frac{\\pi \\, t}{T}\\right)\\right)"}),e.jsxs("p",{className:"mt-2",children:["The learning rate starts at ",e.jsx(t.InlineMath,{math:"\\alpha_{\\max}"}),", smoothly decreases to",e.jsx(t.InlineMath,{math:"\\alpha_{\\min}"})," following a half cosine curve over ",e.jsx(t.InlineMath,{math:"T"})," steps."]})]}),e.jsxs(v,{title:"SGDR: Warm Restarts",children:[e.jsx(t.BlockMath,{math:"\\alpha_t = \\alpha_{\\min} + \\frac{1}{2}(\\alpha_{\\max} - \\alpha_{\\min})\\left(1 + \\cos\\!\\left(\\frac{\\pi \\, T_{\\text{cur}}}{T_i}\\right)\\right)"}),e.jsxs("p",{className:"mt-2",children:["After each period ",e.jsx(t.InlineMath,{math:"T_i"}),", the learning rate jumps back to ",e.jsx(t.InlineMath,{math:"\\alpha_{\\max}"}),". Period lengths can increase with ",e.jsx(t.InlineMath,{math:"T_i = T_0 \\cdot T_{\\text{mult}}^i"}),"."]})]}),e.jsx(Y,{}),e.jsx(W,{title:"Why Cosine Works",id:"cosine-intuition",children:e.jsxs("p",{children:["The cosine schedule spends most of its time at moderate learning rates (the flat part of the cosine near 0 and ",e.jsx(t.InlineMath,{math:"\\pi"}),"). Compared to linear decay, it provides more aggressive early reduction and gentler final convergence, matching the typical optimization landscape of neural networks."]})}),e.jsx(L,{title:"Snapshot Ensembles",children:e.jsx("p",{children:"With warm restarts, save model weights at the end of each cosine cycle (the minimum LR point). Average predictions from these snapshots for a free ensemble that typically improves accuracy by 0.5-1% without extra training cost."})}),e.jsx(S,{title:"Cosine Annealing in PyTorch",code:`import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

model = nn.Linear(128, 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Simple cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# Or with warm restarts (SGDR)
scheduler_wr = CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

lrs = []
for step in range(100):
    optimizer.step()
    scheduler.step()
    lrs.append(scheduler.get_last_lr()[0])

print(f"LR at step 0: {lrs[0]:.6f}")
print(f"LR at step 25: {lrs[25]:.6f}")
print(f"LR at step 49: {lrs[49]:.6f}")`}),e.jsx(z,{type:"note",title:"Cosine is the Default for LLMs",children:e.jsx("p",{children:"Nearly all modern LLM training runs (GPT, LLaMA, etc.) use cosine decay, typically with linear warmup for the first 1-2% of steps and a minimum LR of 10% of the peak rate. This has become the de facto standard schedule."})})]})}const Ne=Object.freeze(Object.defineProperty({__proto__:null,default:J},Symbol.toStringTag,{value:"Module"}));function ee(){const[a,j]=_.useState(.3),l=400,h=180,i=100,s=1,p=25,m=1e4,x=[],o=s/p,c=s/m,d=Math.floor(i*a),r=i-d;for(let f=0;f<i;f++){let y;if(f<d)y=o+(s-o)*(f/d);else{const b=(f-d)/r;y=s-(s-c)*b}x.push(y)}const n=l/i,g=(h-30)/(s*1.1),u=x.map((f,y)=>`${y===0?"M":"L"}${y*n},${h-20-f*g}`).join(" ");return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"One-Cycle Learning Rate Policy"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Warmup fraction: ",a.toFixed(2),e.jsx("input",{type:"range",min:.1,max:.5,step:.05,value:a,onChange:f=>j(parseFloat(f.target.value)),className:"w-32 accent-violet-500"})]}),e.jsxs("svg",{width:l,height:h,className:"mx-auto block",children:[e.jsx("line",{x1:0,y1:h-20,x2:l,y2:h-20,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("line",{x1:d*n,y1:0,x2:d*n,y2:h-20,stroke:"#f97316",strokeWidth:.8,strokeDasharray:"3,3"}),e.jsx("path",{d:u,fill:"none",stroke:"#8b5cf6",strokeWidth:2}),e.jsx("text",{x:d*n+3,y:12,fill:"#f97316",fontSize:9,children:"peak"})]})]})}function te(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Cyclic learning rate policies oscillate the learning rate between bounds. The one-cycle policy, proposed by Leslie Smith, enables super-convergence: training in dramatically fewer iterations."}),e.jsxs(v,{title:"Cyclic Learning Rate",children:[e.jsx(t.BlockMath,{math:"\\alpha_t = \\alpha_{\\min} + (\\alpha_{\\max} - \\alpha_{\\min}) \\cdot \\max(0, 1 - |t / T_{\\text{half}} - 1|)"}),e.jsxs("p",{className:"mt-2",children:["The learning rate linearly increases from ",e.jsx(t.InlineMath,{math:"\\alpha_{\\min}"})," to",e.jsx(t.InlineMath,{math:"\\alpha_{\\max}"}),", then linearly decreases back, repeating cyclically. This triangular wave can help explore the loss landscape."]})]}),e.jsxs(v,{title:"One-Cycle Policy",children:[e.jsx("p",{children:"A single cycle of: warm up to peak LR, then anneal down to a very small value."}),e.jsx(t.BlockMath,{math:"\\text{Phase 1: } \\alpha_{\\min} \\to \\alpha_{\\max} \\quad (\\sim 30\\% \\text{ of training})"}),e.jsx(t.BlockMath,{math:"\\text{Phase 2: } \\alpha_{\\max} \\to \\alpha_{\\min}/10^4 \\quad (\\sim 70\\% \\text{ of training})"})]}),e.jsx(ee,{}),e.jsx(W,{title:"Super-Convergence",id:"super-convergence",children:e.jsx("p",{children:"With the one-cycle policy, certain architectures can be trained in 1/5 to 1/10 of the usual number of epochs. The high learning rate phase acts as regularization (similar to large noise), while the final low LR phase fine-tunes to a sharp minimum."})}),e.jsx(L,{title:"LR Range Test (Smith's Method)",children:e.jsx("p",{children:"To find the optimal max LR: start with a very small LR and exponentially increase it over one epoch. Plot loss vs LR. The optimal max LR is typically where loss is still decreasing but before it diverges — usually one order of magnitude before the minimum."})}),e.jsx(S,{title:"One-Cycle Policy in PyTorch",code:`import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# One-cycle policy
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.1, total_steps=1000,
    pct_start=0.3,        # 30% warmup
    div_factor=25,         # initial_lr = max_lr / 25
    final_div_factor=1e4,  # final_lr = initial_lr / 1e4
    anneal_strategy='cos'
)

# LR Range Test
def lr_range_test(model, data_loader, start_lr=1e-7, end_lr=10, steps=100):
    lrs, losses_list = [], []
    lr = start_lr
    mult = (end_lr / start_lr) ** (1 / steps)
    opt = torch.optim.SGD(model.parameters(), lr=start_lr)
    for i, (x, y) in zip(range(steps), data_loader):
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        opt.step(); opt.zero_grad()
        lrs.append(lr)
        losses_list.append(loss.item())
        lr *= mult
        for pg in opt.param_groups: pg['lr'] = lr
    return lrs, losses_list

print("One-cycle scheduler created with 1000 steps")`}),e.jsx(z,{type:"note",title:"When to Use One-Cycle",children:e.jsx("p",{children:"One-cycle works best with SGD+momentum for CNNs and can dramatically reduce training time. For Transformers with Adam, cosine annealing with warmup is usually preferred. Always run the LR range test first to find the right peak learning rate."})})]})}const we=Object.freeze(Object.defineProperty({__proto__:null,default:te},Symbol.toStringTag,{value:"Module"}));function ae(){const[a,j]=_.useState(8),[l,h]=_.useState(1),[i,s]=_.useState(0),p=380,m=180,x=Array.from({length:a},(y,b)=>2*Math.sin(b*1.3)+3),o=x.reduce((y,b)=>y+b,0)/x.length,c=x.reduce((y,b)=>y+(b-o)**2,0)/x.length,d=x.map(y=>l*(y-o)/Math.sqrt(c+1e-5)+i),r=[...x,...d],n=Math.min(...r)-.5,g=Math.max(...r)+.5,u=p/(a+1),f=(m-30)/(g-n);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Batch Norm Effect"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3 flex-wrap",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["γ = ",l.toFixed(1),e.jsx("input",{type:"range",min:.1,max:3,step:.1,value:l,onChange:y=>h(parseFloat(y.target.value)),className:"w-24 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["β = ",i.toFixed(1),e.jsx("input",{type:"range",min:-2,max:2,step:.1,value:i,onChange:y=>s(parseFloat(y.target.value)),className:"w-24 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Batch: ",a,e.jsx("input",{type:"range",min:4,max:16,step:1,value:a,onChange:y=>j(parseInt(y.target.value)),className:"w-24 accent-violet-500"})]})]}),e.jsxs("svg",{width:p,height:m,className:"mx-auto block",children:[e.jsx("line",{x1:0,y1:m-20,x2:p,y2:m-20,stroke:"#d1d5db",strokeWidth:.5}),x.map((y,b)=>e.jsxs("g",{children:[e.jsx("circle",{cx:(b+.7)*u,cy:m-20-(y-n)*f,r:5,fill:"#9ca3af",opacity:.6}),e.jsx("circle",{cx:(b+.7)*u,cy:m-20-(d[b]-n)*f,r:5,fill:"#8b5cf6"})]},`r-${b}`))]}),e.jsxs("div",{className:"mt-2 flex justify-center gap-6 text-xs",children:[e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-2 h-2 rounded-full bg-gray-400"})," Raw"]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-2 h-2 rounded-full bg-violet-500"})," Normalized"]})]}),e.jsxs("div",{className:"mt-1 text-center text-xs text-gray-500",children:["μ = ",o.toFixed(2),", σ² = ",c.toFixed(2)]})]})}function re(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Batch Normalization (Ioffe & Szegedy, 2015) normalizes activations across the mini-batch, enabling faster training with higher learning rates and reducing sensitivity to initialization."}),e.jsxs(v,{title:"Batch Normalization",children:[e.jsx(t.BlockMath,{math:"\\hat{x}_i = \\frac{x_i - \\mu_B}{\\sqrt{\\sigma_B^2 + \\epsilon}}"}),e.jsx(t.BlockMath,{math:"y_i = \\gamma \\hat{x}_i + \\beta"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"\\mu_B"})," and ",e.jsx(t.InlineMath,{math:"\\sigma_B^2"})," are computed over the mini-batch, and ",e.jsx(t.InlineMath,{math:"\\gamma, \\beta"})," are learned affine parameters."]})]}),e.jsx(ae,{}),e.jsxs(W,{title:"Smoothing Effect",id:"bn-smoothing",children:[e.jsx("p",{children:"Rather than reducing “internal covariate shift” as originally claimed, BatchNorm's main benefit is making the loss landscape significantly smoother:"}),e.jsx(t.BlockMath,{math:"\\|\\nabla \\mathcal{L}_{\\text{BN}}\\| \\leq \\|\\nabla \\mathcal{L}\\| \\cdot \\frac{\\gamma}{\\sqrt{\\sigma_B^2 + \\epsilon}}"}),e.jsx("p",{children:"This allows larger learning rates without divergence (Santurkar et al., 2018)."})]}),e.jsx(L,{title:"Train vs Eval Mode",children:e.jsxs("p",{children:["During training, BN uses mini-batch statistics. During evaluation, it uses running averages accumulated during training. Forgetting to switch modes (",e.jsx("code",{children:"model.eval()"}),") is a common source of bugs that causes inference performance to degrade."]})}),e.jsx(S,{title:"Batch Normalization in PyTorch",code:`import torch
import torch.nn as nn

# BN for fully connected layers
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),  # normalizes across batch dim
    nn.ReLU(),
    nn.Linear(256, 10)
)

# BN for convolutional layers
conv_model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.BatchNorm2d(64),   # normalizes across batch, H, W
    nn.ReLU(),
)

# Training vs evaluation mode matters!
model.train()   # uses batch statistics
x_train = torch.randn(32, 784)
out = model(x_train)

model.eval()    # uses running mean/var
x_test = torch.randn(1, 784)
out = model(x_test)  # works even with batch_size=1
print(f"Output shape: {out.shape}")`}),e.jsx(I,{title:"Small Batch Sizes",children:e.jsx("p",{children:"BatchNorm degrades with small batch sizes (below ~16) because the mini-batch statistics become noisy. For small batches, use GroupNorm or LayerNorm instead. This is particularly relevant for object detection and segmentation models with large inputs."})}),e.jsx(z,{type:"note",title:"BatchNorm's Legacy",children:e.jsx("p",{children:"BatchNorm was transformative for CNNs and remains the default normalization for vision models. However, Transformers and RNNs typically use LayerNorm due to variable sequence lengths and the desire for batch-independent normalization."})})]})}const Me=Object.freeze(Object.defineProperty({__proto__:null,default:re},Symbol.toStringTag,{value:"Module"}));function se(){const[a,j]=_.useState("layer"),l=4,h=6;Array.from({length:l*h},(o,c)=>.3*Math.sin(c*.7)+.5*Math.cos(c*.3));const i=44,s=32,p=2,m={batch:["#8b5cf6","#a78bfa","#c4b5fd","#ddd6fe","#ede9fe","#f5f3ff"],layer:["#8b5cf6","#a78bfa","#c4b5fd","#ddd6fe"]},x=(o,c)=>a==="batch"?m.batch[c%m.batch.length]:a==="layer"?m.layer[o%m.layer.length]:"#8b5cf6";return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-2 text-base font-bold text-gray-800 dark:text-gray-200",children:"Normalization Dimensions"}),e.jsx("div",{className:"flex gap-2 mb-3",children:["batch","layer","instance"].map(o=>e.jsx("button",{onClick:()=>j(o),className:`px-3 py-1 rounded text-xs font-medium ${a===o?"bg-violet-500 text-white":"bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"}`,children:o==="batch"?"BatchNorm":o==="layer"?"LayerNorm":"InstanceNorm"},o))}),e.jsx("div",{className:"flex justify-center",children:e.jsxs("svg",{width:(h+1)*(i+p)+40,height:(l+1)*(s+p)+10,children:[e.jsx("text",{x:0,y:15,fill:"#6b7280",fontSize:10,children:"B \\ C"}),Array.from({length:h},(o,c)=>e.jsxs("text",{x:40+c*(i+p)+i/2,y:15,textAnchor:"middle",fill:"#6b7280",fontSize:9,children:["c",c]},`h-${c}`)),Array.from({length:l},(o,c)=>e.jsxs("g",{children:[e.jsxs("text",{x:15,y:30+c*(s+p)+s/2+4,textAnchor:"middle",fill:"#6b7280",fontSize:9,children:["b",c]}),Array.from({length:h},(d,r)=>e.jsx("rect",{x:40+r*(i+p),y:22+c*(s+p),width:i,height:s,rx:4,fill:x(c,r),opacity:.5,stroke:x(c,r),strokeWidth:1.5},`cell-${c}-${r}`))]},`row-${c}`))]})}),e.jsxs("p",{className:"text-center text-xs text-gray-500 mt-2",children:[a==="batch"&&"Same color = normalized together (across batch, per channel)",a==="layer"&&"Same color = normalized together (across channels, per instance)",a==="instance"&&"Each cell normalized independently (per instance, per channel)"]})]})}function ne(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Layer Normalization (Ba et al., 2016) normalizes across features within each instance, making it independent of batch size. It is the standard normalization for Transformers."}),e.jsxs(v,{title:"Layer Normalization",children:[e.jsx(t.BlockMath,{math:"\\mu = \\frac{1}{H}\\sum_{i=1}^{H} x_i, \\quad \\sigma^2 = \\frac{1}{H}\\sum_{i=1}^{H}(x_i - \\mu)^2"}),e.jsx(t.BlockMath,{math:"y_i = \\gamma \\frac{x_i - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta"}),e.jsxs("p",{className:"mt-2",children:["Statistics are computed over the feature dimension ",e.jsx(t.InlineMath,{math:"H"})," independently for each sample. No dependence on other samples in the batch."]})]}),e.jsx(se,{}),e.jsxs(W,{title:"LayerNorm vs BatchNorm",id:"ln-vs-bn",children:[e.jsx("p",{children:"Key differences that make LayerNorm preferred for sequence models:"}),e.jsx(t.BlockMath,{math:"\\text{BN: normalize over } (B, H, W) \\quad \\text{LN: normalize over } (C)"}),e.jsx("p",{children:"LayerNorm computes identical results regardless of batch size (even 1). It handles variable-length sequences naturally and behaves identically at train and eval time."})]}),e.jsx(L,{title:"Pre-Norm vs Post-Norm Transformers",children:e.jsxs("p",{children:[e.jsx("strong",{children:"Post-Norm"})," (original): ",e.jsx(t.InlineMath,{math:"x + \\text{LN}(\\text{Attn}(x))"}),".",e.jsx("strong",{children:" Pre-Norm"})," (GPT-2+): ",e.jsx(t.InlineMath,{math:"x + \\text{Attn}(\\text{LN}(x))"}),". Pre-Norm is more stable for training deep Transformers because gradients flow through the residual path without being normalized."]})}),e.jsx(S,{title:"Layer Normalization in PyTorch",code:`import torch
import torch.nn as nn

# LayerNorm for a Transformer with d_model=512
ln = nn.LayerNorm(512)

# Works the same regardless of batch size
x1 = torch.randn(1, 10, 512)   # batch=1, seq=10
x32 = torch.randn(32, 10, 512) # batch=32, seq=10

# Identical normalization per-instance
out1 = ln(x1)
out32 = ln(x32)

# Pre-norm Transformer block pattern
class PreNormBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ffn(self.ln2(x))
        return x

block = PreNormBlock(512, 8)
print(f"Output: {block(x32).shape}")`}),e.jsx(z,{type:"note",title:"When to Use Which Norm",children:e.jsxs("p",{children:[e.jsx("strong",{children:"BatchNorm"}),": CNNs with batch size ≥ 16. ",e.jsx("strong",{children:"LayerNorm"}),": Transformers, RNNs, any model needing batch-independent normalization.",e.jsx("strong",{children:" GroupNorm"}),": CNNs with small batch sizes. The trend in modern architectures is toward LayerNorm and its variants (RMSNorm)."]})})]})}const Le=Object.freeze(Object.defineProperty({__proto__:null,default:ne},Symbol.toStringTag,{value:"Module"}));function ie(){const[a,j]=_.useState(2),l=8,h=4,i=36,s=28,p=2,m=["#8b5cf6","#f97316","#10b981","#ef4444","#3b82f6","#f59e0b","#ec4899","#06b6d4"];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Group Normalization Groups"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Groups: ",a,e.jsx("input",{type:"range",min:1,max:8,step:1,value:a,onChange:x=>j(parseInt(x.target.value)),className:"w-28 accent-violet-500"})]}),e.jsx("div",{className:"flex justify-center",children:e.jsxs("svg",{width:l*(i+p)+80,height:h*(s+p)+30,children:[e.jsx("text",{x:5,y:12,fill:"#6b7280",fontSize:9,children:"Channels →"}),e.jsx("text",{x:5,y:24+h*(s+p)/2,fill:"#6b7280",fontSize:9,children:"Spatial ↓"}),Array.from({length:h},(x,o)=>Array.from({length:l},(c,d)=>{const r=Math.floor(d/(l/a));return e.jsx("rect",{x:45+d*(i+p),y:20+o*(s+p),width:i,height:s,rx:3,fill:m[r%m.length],opacity:.45,stroke:m[r%m.length],strokeWidth:1.5},`${o}-${d}`)})),Array.from({length:a},(x,o)=>{const c=l/a,d=o*c;return e.jsxs("text",{x:45+(d+c/2)*(i+p),y:18,textAnchor:"middle",fill:m[o%m.length],fontSize:9,children:["G",o]},`gl-${o}`)})]})}),e.jsxs("p",{className:"text-center text-xs text-gray-500 mt-1",children:[a," group",a>1?"s":""," of ",l/a," channel",l/a>1?"s":""," each.",a===l?" (= Instance Norm)":a===1?" (= Layer Norm)":""]})]})}function oe(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"RMSNorm simplifies LayerNorm by removing the mean centering, while GroupNorm provides a flexible middle ground between BatchNorm and InstanceNorm for convolutional networks."}),e.jsxs(v,{title:"RMSNorm",children:[e.jsx(t.BlockMath,{math:"\\text{RMS}(x) = \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n} x_i^2}"}),e.jsx(t.BlockMath,{math:"y_i = \\frac{x_i}{\\text{RMS}(x) + \\epsilon} \\cdot \\gamma_i"}),e.jsx("p",{className:"mt-2",children:"No mean subtraction, no learned bias — just scale by the root mean square. This reduces computation by ~7-10% compared to LayerNorm with negligible quality loss."})]}),e.jsxs(W,{title:"Why Removing the Mean Works",id:"rmsnorm-theory",children:[e.jsxs("p",{children:["The re-centering in LayerNorm provides invariance to shifts in activation distributions. In practice, the learned parameters ",e.jsx(t.InlineMath,{math:"\\gamma"})," and the subsequent linear layers can compensate for this. Empirically, the scaling (RMS) component does most of the heavy lifting:"]}),e.jsx(t.BlockMath,{math:"\\text{LayerNorm}(x) \\approx \\text{RMSNorm}(x) \\text{ when } \\mu_x \\approx 0"})]}),e.jsxs(v,{title:"Group Normalization",children:[e.jsx(t.BlockMath,{math:"\\mu_g = \\frac{1}{|S_g|}\\sum_{i \\in S_g} x_i, \\quad \\sigma_g^2 = \\frac{1}{|S_g|}\\sum_{i \\in S_g}(x_i - \\mu_g)^2"}),e.jsxs("p",{className:"mt-2",children:["Channels are divided into ",e.jsx(t.InlineMath,{math:"G"})," groups, each normalized independently. When ",e.jsx(t.InlineMath,{math:"G = 1"}),", it is LayerNorm; when ",e.jsx(t.InlineMath,{math:"G = C"}),", it is InstanceNorm."]})]}),e.jsx(ie,{}),e.jsx(L,{title:"RMSNorm in Modern LLMs",children:e.jsx("p",{children:"LLaMA, Mistral, Gemma, and most recent LLMs use RMSNorm instead of LayerNorm. The savings compound across billions of tokens and hundreds of layers: for a 70B parameter model, RMSNorm saves significant compute per forward pass."})}),e.jsx(S,{title:"RMSNorm & GroupNorm Implementation",code:`import torch
import torch.nn as nn

# RMSNorm (not built into PyTorch, easy to implement)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

# Usage comparison
d_model = 512
rmsnorm = RMSNorm(d_model)
layernorm = nn.LayerNorm(d_model)
groupnorm = nn.GroupNorm(num_groups=32, num_channels=d_model)

x = torch.randn(8, 10, d_model)
print(f"RMSNorm:   {rmsnorm(x).shape}")
print(f"LayerNorm: {layernorm(x).shape}")

# GroupNorm for conv features (B, C, H, W)
x_conv = torch.randn(8, d_model, 16, 16)
print(f"GroupNorm: {groupnorm(x_conv).shape}")

# GroupNorm works with any batch size
x_single = torch.randn(1, d_model, 16, 16)
print(f"GroupNorm batch=1: {groupnorm(x_single).shape}")`}),e.jsx(z,{type:"note",title:"Choosing the Right Norm",children:e.jsxs("p",{children:[e.jsx("strong",{children:"RMSNorm"}),": LLMs and Transformers (fast, effective).",e.jsx("strong",{children:" GroupNorm"}),": CNNs with small batch sizes (detection, segmentation).",e.jsx("strong",{children:" LayerNorm"}),": General Transformers. ",e.jsx("strong",{children:"BatchNorm"}),": CNNs with large batches (classification). The trend is clearly toward simpler norms."]})})]})}const Se=Object.freeze(Object.defineProperty({__proto__:null,default:oe},Symbol.toStringTag,{value:"Module"}));function le(){const[a,j]=_.useState(256),[l,h]=_.useState("xavier"),i=380,s=180,p=10,m=[1];for(let d=1;d<=p;d++){const r=m[d-1];let n;l==="xavier"?n=2/(a+a):l==="he"?n=2/a:n=1/a;const g=l==="he"?.5:1;m.push(r*a*n*g)}const x=Math.max(...m)*1.2,o=i/(p+1),c=o*.7;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Activation Variance Through Layers"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3 flex-wrap",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["fan_in: ",a,e.jsx("input",{type:"range",min:32,max:1024,step:32,value:a,onChange:d=>j(parseInt(d.target.value)),className:"w-28 accent-violet-500"})]}),e.jsx("div",{className:"flex gap-2",children:["xavier","he","naive"].map(d=>e.jsx("button",{onClick:()=>h(d),className:`px-3 py-1 rounded text-xs font-medium ${l===d?"bg-violet-500 text-white":"bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"}`,children:d==="naive"?"1/n":d},d))})]}),e.jsxs("svg",{width:i,height:s,className:"mx-auto block",children:[e.jsx("line",{x1:0,y1:s-20,x2:i,y2:s-20,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("line",{x1:0,y1:s-20-1/x*(s-40),x2:i,y2:s-20-1/x*(s-40),stroke:"#f97316",strokeWidth:.8,strokeDasharray:"3,3"}),m.map((d,r)=>{const n=Math.min(d/x*(s-40),s-30);return e.jsxs("g",{children:[e.jsx("rect",{x:r*o+(o-c)/2,y:s-20-n,width:c,height:n,fill:"#8b5cf6",rx:3,opacity:.7}),e.jsxs("text",{x:r*o+o/2,y:s-7,textAnchor:"middle",fill:"#6b7280",fontSize:8,children:["L",r]})]},r)})]}),e.jsx("div",{className:"mt-1 text-center text-xs text-gray-500",children:"Orange dashed = ideal variance (1.0)"})]})}function ce(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Proper weight initialization ensures that activations and gradients maintain stable variance through the network. Xavier init targets linear/tanh activations, while He init accounts for the ReLU non-linearity."}),e.jsxs(v,{title:"Xavier (Glorot) Initialization",children:[e.jsx(t.BlockMath,{math:"W \\sim \\mathcal{N}\\!\\left(0, \\frac{2}{n_{\\text{in}} + n_{\\text{out}}}\\right) \\quad \\text{or} \\quad W \\sim U\\!\\left[-\\sqrt{\\frac{6}{n_{\\text{in}} + n_{\\text{out}}}}, \\sqrt{\\frac{6}{n_{\\text{in}} + n_{\\text{out}}}}\\right]"}),e.jsx("p",{className:"mt-2",children:"Designed for linear or tanh activations. Preserves variance in both forward and backward passes."})]}),e.jsxs(v,{title:"He (Kaiming) Initialization",children:[e.jsx(t.BlockMath,{math:"W \\sim \\mathcal{N}\\!\\left(0, \\frac{2}{n_{\\text{in}}}\\right)"}),e.jsxs("p",{className:"mt-2",children:["Accounts for ReLU zeroing out half the activations. The factor of 2 compensates for the ",e.jsx(t.InlineMath,{math:"1/2"})," reduction in variance from ReLU."]})]}),e.jsx(le,{}),e.jsxs(A,{title:"Derivation Sketch (He Init)",children:[e.jsxs("p",{children:["For a layer ",e.jsx(t.InlineMath,{math:"y = Wx"})," followed by ReLU:"]}),e.jsx(t.BlockMath,{math:"\\text{Var}(y_j) = n_{\\text{in}} \\cdot \\text{Var}(w) \\cdot \\text{Var}(x)"}),e.jsxs("p",{children:["ReLU zeroes negative half, so ",e.jsx(t.InlineMath,{math:"\\text{Var}(\\text{ReLU}(y)) = \\frac{1}{2}\\text{Var}(y)"}),"."]}),e.jsxs("p",{children:["Setting ",e.jsx(t.InlineMath,{math:"\\text{Var}(w) = 2/n_{\\text{in}}"})," gives ",e.jsx(t.InlineMath,{math:"\\text{Var}(\\text{ReLU}(y)) = \\text{Var}(x)"}),"."]})]}),e.jsx(L,{title:"When to Use Each",children:e.jsxs("p",{children:[e.jsx("strong",{children:"Xavier"}),": sigmoid, tanh, linear layers, SELU.",e.jsx("strong",{children:" He"}),": ReLU, Leaky ReLU, ELU, GELU. Using He init with tanh will cause exploding activations; using Xavier with ReLU will cause dying neurons."]})}),e.jsx(S,{title:"Xavier & He Init in PyTorch",code:`import torch
import torch.nn as nn

layer = nn.Linear(512, 256)

# Xavier (Glorot) — for tanh/sigmoid
nn.init.xavier_uniform_(layer.weight)    # uniform variant
nn.init.xavier_normal_(layer.weight)     # normal variant

# He (Kaiming) — for ReLU
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# Verify variance
w = layer.weight.data
print(f"Weight std: {w.std().item():.4f}")
print(f"Expected (He): {(2/512)**0.5:.4f}")

# Initialize full model
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
model.apply(init_weights)
print("Model initialized with He init")`}),e.jsx(z,{type:"note",title:"Modern Frameworks Handle This",children:e.jsx("p",{children:"PyTorch uses Kaiming uniform by default for linear and conv layers. You rarely need to manually initialize unless using custom architectures. However, understanding the theory helps diagnose training instabilities in deep or unusual networks."})})]})}const ze=Object.freeze(Object.defineProperty({__proto__:null,default:ce},Symbol.toStringTag,{value:"Module"}));function de(){const[a,j]=_.useState("orthogonal"),l=380,h=160,i=20,s=Array.from({length:i},(n,g)=>{const u=(g+1)/i;return 1+.6*Math.exp(-3*u)-.3*u+.15*Math.sin(g)}).sort((n,g)=>g-n),p=Array.from({length:i},()=>1),m=Array.from({length:i},(n,g)=>1+.05*Math.sin(g*2)-.02*Math.cos(g)).sort((n,g)=>g-n),x=a==="orthogonal"?p:a==="lsuv"?m:s,o=Math.max(...s)*1.2,c=l/(i+1),d=(h-30)/o,r=c*.7;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Singular Value Distribution"}),e.jsx("div",{className:"flex gap-2 mb-3",children:["gaussian","orthogonal","lsuv"].map(n=>e.jsx("button",{onClick:()=>j(n),className:`px-3 py-1 rounded text-xs font-medium ${a===n?"bg-violet-500 text-white":"bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"}`,children:n==="gaussian"?"Gaussian":n==="orthogonal"?"Orthogonal":"LSUV"},n))}),e.jsxs("svg",{width:l,height:h,className:"mx-auto block",children:[e.jsx("line",{x1:0,y1:h-20,x2:l,y2:h-20,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("line",{x1:0,y1:h-20-1*d,x2:l,y2:h-20-1*d,stroke:"#f97316",strokeWidth:.8,strokeDasharray:"3,3"}),x.map((n,g)=>e.jsx("rect",{x:g*c+(c-r)/2+c/2,y:h-20-n*d,width:r,height:n*d,fill:"#8b5cf6",rx:2,opacity:.7},g))]}),e.jsx("div",{className:"mt-1 text-center text-xs text-gray-500",children:"Orange = ideal σ=1 | Bars = singular values (sorted)"})]})}function he(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Orthogonal initialization sets weight matrices to (scaled) orthogonal matrices, ensuring all singular values are equal. LSUV extends this with a data-dependent calibration pass."}),e.jsxs(v,{title:"Orthogonal Initialization",children:[e.jsx(t.BlockMath,{math:"W = Q \\quad \\text{where } Q^TQ = I \\text{ (from QR decomposition of random matrix)}"}),e.jsxs("p",{className:"mt-2",children:["All singular values of ",e.jsx(t.InlineMath,{math:"W"})," are exactly 1, so",e.jsx(t.InlineMath,{math:"\\|Wx\\| = \\|x\\|"})," — the transform is norm-preserving. Optionally scale by a gain factor ",e.jsx(t.InlineMath,{math:"g"})," for specific activations."]})]}),e.jsx(de,{}),e.jsxs(W,{title:"Dynamical Isometry",id:"dynamical-isometry",children:[e.jsx("p",{children:"A network satisfies dynamical isometry when the singular values of the input-output Jacobian are concentrated near 1:"}),e.jsx(t.BlockMath,{math:"\\sigma_i\\!\\left(\\frac{\\partial f(x)}{\\partial x}\\right) \\approx 1, \\quad \\forall i"}),e.jsx("p",{children:"Orthogonal initialization achieves this for linear networks. For non-linear networks, careful combination with activation choice is needed."})]}),e.jsxs(v,{title:"LSUV (Layer-Sequential Unit-Variance)",children:[e.jsx("p",{children:"A data-dependent initialization procedure:"}),e.jsx(t.BlockMath,{math:"\\text{1. Initialize } W_l \\text{ orthogonally}"}),e.jsx(t.BlockMath,{math:"\\text{2. Forward pass a mini-batch through layers 1..}l"}),e.jsx(t.BlockMath,{math:"\\text{3. Scale } W_l \\leftarrow W_l / \\text{std}(\\text{output}_l) \\text{ until Var} \\approx 1"}),e.jsx("p",{className:"mt-2",children:"Repeat for each layer sequentially. This accounts for actual non-linearities."})]}),e.jsx(L,{title:"When Orthogonal Init Helps",children:e.jsx("p",{children:"Orthogonal init is especially beneficial for RNNs and very deep networks (50+ layers) where repeated matrix multiplication causes exponential growth or decay with non-isometric weights. For standard Transformers and CNNs with normalization, the benefit is smaller."})}),e.jsx(S,{title:"Orthogonal & LSUV in PyTorch",code:`import torch
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
print("LSUV initialization complete")`}),e.jsx(I,{title:"Orthogonal Init Requires Square-ish Matrices",children:e.jsxs("p",{children:["For non-square weight matrices, the orthogonal initialization produces a semi-orthogonal matrix. When ",e.jsx(t.InlineMath,{math:"n_{\\text{out}} \\gg n_{\\text{in}}"})," or vice versa, the singular value guarantee weakens. In such cases, He init may be more robust."]})}),e.jsx(z,{type:"note",title:"RNN-Specific Benefit",children:e.jsx("p",{children:"For RNN hidden-to-hidden weights, orthogonal init is nearly essential. It prevents the vanishing/exploding gradient problem inherent in repeated matrix multiplication across time steps. LSTMs and GRUs partially address this architecturally."})})]})}const Re=Object.freeze(Object.defineProperty({__proto__:null,default:he},Symbol.toStringTag,{value:"Module"}));function me(){const[a,j]=_.useState(20),[l,h]=_.useState("rezero"),i=380,s=180,p=[];let m=1;for(let r=0;r<=a;r++)p.push(m),l==="naive"?m+=1:l==="fixup"?m+=Math.pow(a,-.5):m+=0;const x=Math.max(...p)*1.1,o=i/(a+2),c=(s-30)/x,d=p.map((r,n)=>`${n===0?"M":"L"}${(n+.5)*o},${s-20-r*c}`).join(" ");return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Signal Variance Through Residual Blocks"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3 flex-wrap",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Depth: ",a,e.jsx("input",{type:"range",min:5,max:100,step:5,value:a,onChange:r=>j(parseInt(r.target.value)),className:"w-28 accent-violet-500"})]}),e.jsx("div",{className:"flex gap-2",children:["naive","fixup","rezero"].map(r=>e.jsx("button",{onClick:()=>h(r),className:`px-3 py-1 rounded text-xs font-medium ${l===r?"bg-violet-500 text-white":"bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"}`,children:r},r))})]}),e.jsxs("svg",{width:i,height:s,className:"mx-auto block",children:[e.jsx("line",{x1:0,y1:s-20,x2:i,y2:s-20,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("path",{d,fill:"none",stroke:"#8b5cf6",strokeWidth:2}),e.jsx("line",{x1:0,y1:s-20-1*c,x2:i,y2:s-20-1*c,stroke:"#f97316",strokeWidth:.8,strokeDasharray:"3,3"})]}),e.jsxs("div",{className:"mt-1 text-center text-xs text-gray-500",children:["Orange = input variance | ",l==="naive"?"Variance grows linearly with depth!":l==="fixup"?"Growth controlled by L^(-0.5) scaling":"α=0 at init: no growth"]})]})}function xe(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Fixup and ReZero enable training very deep residual networks without normalization layers. They address the signal explosion problem in deep ResNets through careful initialization of residual branches."}),e.jsxs(v,{title:"The Residual Variance Problem",children:[e.jsxs("p",{children:["In a standard ResNet with ",e.jsx(t.InlineMath,{math:"L"})," blocks:"]}),e.jsx(t.BlockMath,{math:"x_{l+1} = x_l + F_l(x_l)"}),e.jsx("p",{children:"If each branch adds unit-variance signal, the output variance grows as:"}),e.jsx(t.BlockMath,{math:"\\text{Var}(x_L) = \\text{Var}(x_0) + L \\cdot \\text{Var}(F)"}),e.jsxs("p",{className:"mt-2",children:["For ",e.jsx(t.InlineMath,{math:"L = 100"}),", the signal is 100x larger than the input."]})]}),e.jsx(me,{}),e.jsxs(v,{title:"Fixup Initialization",children:[e.jsx(t.BlockMath,{math:"W_l^{(1)} \\sim \\mathcal{N}(0, \\text{He variance}) \\cdot L^{-1/(2m)}"}),e.jsx(t.BlockMath,{math:"W_l^{(m)} = 0 \\quad \\text{(last layer in each residual branch)}"}),e.jsxs("p",{className:"mt-2",children:["Scale down early layers in each block by ",e.jsx(t.InlineMath,{math:"L^{-1/(2m)}"})," where",e.jsx(t.InlineMath,{math:"m"})," is the number of layers per block. Zero-initialize the last layer so each block starts as an identity function."]})]}),e.jsxs(v,{title:"ReZero",children:[e.jsx(t.BlockMath,{math:"x_{l+1} = x_l + \\alpha_l \\cdot F_l(x_l), \\quad \\alpha_l = 0 \\text{ at init}"}),e.jsxs("p",{className:"mt-2",children:["Simply multiply each residual branch by a learnable scalar ",e.jsx(t.InlineMath,{math:"\\alpha_l"})," initialized to zero. The network starts as the identity and gradually learns to incorporate residual contributions."]})]}),e.jsxs(W,{title:"Training Signal Preservation",id:"fixup-theory",children:[e.jsx("p",{children:"With Fixup scaling, the output variance satisfies:"}),e.jsx(t.BlockMath,{math:"\\text{Var}(x_L) \\leq \\text{Var}(x_0) + O(\\sqrt{L})"}),e.jsxs("p",{children:["With ReZero at initialization: ",e.jsx(t.InlineMath,{math:"\\text{Var}(x_L) = \\text{Var}(x_0)"})," exactly, since all ",e.jsx(t.InlineMath,{math:"\\alpha_l = 0"}),"."]})]}),e.jsx(L,{title:"1000-Layer ResNets",children:e.jsxs("p",{children:["ReZero successfully trains ResNets with 1000+ layers without any normalization, converging faster than BatchNorm-equipped counterparts in the early stages. The learned ",e.jsx(t.InlineMath,{math:"\\alpha_l"})," values reveal which residual blocks the network considers most important."]})}),e.jsx(S,{title:"Fixup & ReZero Residual Blocks",code:`import torch
import torch.nn as nn

class ReZeroBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))  # init to 0!
        self.fn = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.alpha * self.fn(x)

class FixupBlock(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.ones(1))
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)
        # Fixup scaling
        nn.init.kaiming_normal_(self.linear1.weight)
        self.linear1.weight.data *= num_layers ** (-0.25)
        nn.init.zeros_(self.linear2.weight)  # zero last layer

    def forward(self, x):
        out = torch.relu(self.linear1(x) + self.bias1)
        out = self.linear2(out) * self.scale + self.bias2
        return x + out

# Build deep ReZero network
depth = 100
model = nn.Sequential(
    nn.Linear(128, 256),
    *[ReZeroBlock(256) for _ in range(depth)],
    nn.Linear(256, 10)
)
x = torch.randn(8, 128)
out = model(x)
print(f"Output shape: {out.shape}")
print(f"Output std: {out.std().item():.4f}")
alphas = [m.alpha.item() for m in model if isinstance(m, ReZeroBlock)]
print(f"All alphas zero at init: {all(a == 0 for a in alphas)}")`}),e.jsx(I,{title:"ReZero and Generalization",children:e.jsx("p",{children:"While ReZero speeds up early training convergence, some studies show it may slightly underperform BatchNorm-equipped networks in final accuracy. Consider combining ReZero with normalization for the best of both worlds."})}),e.jsx(z,{type:"note",title:"Modern Impact",children:e.jsxs("p",{children:["The zero-init residual idea from ReZero appears in many modern architectures. GPT-2 uses a ",e.jsx(t.InlineMath,{math:"1/\\sqrt{N}"})," scaling on residual paths, and many Transformer implementations zero-initialize the output projection of attention layers. These are spiritual successors to Fixup and ReZero."]})})]})}const Te=Object.freeze(Object.defineProperty({__proto__:null,default:xe},Symbol.toStringTag,{value:"Module"}));export{fe as a,je as b,be as c,ve as d,_e as e,ke as f,Ne as g,we as h,Me as i,Le as j,Se as k,ze as l,Re as m,Te as n,ye as s};

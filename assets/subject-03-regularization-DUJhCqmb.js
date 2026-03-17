import{j as e,r as b}from"./vendor-DpISuAX6.js";import{r as a}from"./vendor-katex-CbWCYdth.js";import{T as M,D as j,E as v,P as _,N as k,W as z}from"./subject-01-foundations-D0A1VJsr.js";function A(){const[n,f]=b.useState(3),i=400,l=250,c=Array.from({length:20},(o,t)=>{const s=-2+t*.22,u=.5*s*s-.3*s+.1;return{x:s,y:u,noisy:u+Math.sin(t*7.3)*.4}}),p=Math.max(.05,1.2/n),r=Math.min(1.5,n*.15),d=p*p+r,m=(o,t)=>{const s=(o+2.5)*(i/5),u=l-(t+.5)*(l/4);return{cx:s,cy:Math.max(5,Math.min(l-5,u))}};return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Bias-Variance Tradeoff Demo"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Polynomial Degree: ",n,e.jsx("input",{type:"range",min:1,max:15,step:1,value:n,onChange:o=>f(parseInt(o.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("svg",{width:i,height:l,className:"mx-auto block",children:[c.map((o,t)=>{const{cx:s,cy:u}=m(o.x,o.noisy);return e.jsx("circle",{cx:s,cy:u,r:3,fill:"#8b5cf6",opacity:.6},t)}),c.map((o,t)=>{const{cx:s,cy:u}=m(o.x,o.y);return e.jsx("circle",{cx:s,cy:u,r:2,fill:"#f97316"},`t${t}`)})]}),e.jsxs("div",{className:"mt-3 flex justify-center gap-6 text-xs text-gray-600 dark:text-gray-400",children:[e.jsxs("span",{children:["Bias",e.jsx("sup",{children:"2"}),": ",(p*p).toFixed(3)]}),e.jsxs("span",{children:["Variance: ",r.toFixed(3)]}),e.jsxs("span",{className:"font-semibold text-violet-600",children:["Total Error: ",d.toFixed(3)]})]})]})}function T(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Every supervised learning model's expected error can be decomposed into three components: bias, variance, and irreducible noise. Understanding this decomposition is key to diagnosing and fixing generalization problems."}),e.jsxs(M,{title:"Bias-Variance Decomposition",id:"bias-variance-decomposition",children:[e.jsxs("p",{children:["For a model ",e.jsx(a.InlineMath,{math:"\\hat{f}"})," trained on dataset ",e.jsx(a.InlineMath,{math:"D"}),", the expected squared error at a point ",e.jsx(a.InlineMath,{math:"x"})," is:"]}),e.jsx(a.BlockMath,{math:"\\mathbb{E}_D\\left[(y - \\hat{f}(x))^2\\right] = \\text{Bias}[\\hat{f}(x)]^2 + \\text{Var}[\\hat{f}(x)] + \\sigma^2"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(a.InlineMath,{math:"\\sigma^2"})," is the irreducible noise."]})]}),e.jsxs(j,{title:"Bias",children:[e.jsx(a.BlockMath,{math:"\\text{Bias}[\\hat{f}(x)] = \\mathbb{E}_D[\\hat{f}(x)] - f(x)"}),e.jsx("p",{className:"mt-2",children:"Bias measures how far the average prediction is from the true function. High bias implies the model is too simple (underfitting)."})]}),e.jsxs(j,{title:"Variance",children:[e.jsx(a.BlockMath,{math:"\\text{Var}[\\hat{f}(x)] = \\mathbb{E}_D\\left[(\\hat{f}(x) - \\mathbb{E}_D[\\hat{f}(x)])^2\\right]"}),e.jsx("p",{className:"mt-2",children:"Variance measures how much predictions fluctuate across different training sets. High variance implies overfitting."})]}),e.jsx(A,{}),e.jsx(v,{title:"Polynomial Regression Intuition",children:e.jsxs("p",{children:["A degree-1 polynomial (linear fit) has ",e.jsx("strong",{children:"high bias"})," but ",e.jsx("strong",{children:"low variance"}),". A degree-15 polynomial has ",e.jsx("strong",{children:"low bias"})," but ",e.jsx("strong",{children:"high variance"})," since it fits training noise exactly. The sweet spot minimizes total error."]})}),e.jsx(_,{title:"Computing Bias-Variance in PyTorch",code:`import torch
import torch.nn as nn

# Simulate bias-variance with multiple training runs
n_runs, n_test = 50, 100
predictions = torch.zeros(n_runs, n_test)

for i in range(n_runs):
    x_train = torch.randn(200, 1)
    y_train = x_train ** 2 + 0.3 * torch.randn(200, 1)
    model = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(100):
        loss = nn.MSELoss()(model(x_train), y_train)
        opt.zero_grad(); loss.backward(); opt.step()
    x_test = torch.linspace(-2, 2, n_test).unsqueeze(1)
    predictions[i] = model(x_test).squeeze().detach()

y_true = torch.linspace(-2, 2, n_test) ** 2
bias_sq = (predictions.mean(0) - y_true) ** 2
variance = predictions.var(0)
print(f"Avg Bias^2: {bias_sq.mean():.4f}")
print(f"Avg Variance: {variance.mean():.4f}")`}),e.jsx(k,{type:"note",title:"Deep Learning and the Bias-Variance Tradeoff",children:e.jsx("p",{children:"Modern deep networks challenge the classical tradeoff. Overparameterized models can achieve both low bias and low variance through implicit regularization from SGD, architecture choices, and explicit regularization techniques covered in this subject."})})]})}const ce=Object.freeze(Object.defineProperty({__proto__:null,default:T},Symbol.toStringTag,{value:"Module"}));function D(){const[n,f]=b.useState(60),i=400,l=220,c=s=>1.8*Math.exp(-.04*s)+.05,p=s=>1.8*Math.exp(-.03*s)+.15+.003*Math.max(0,s-30),r=Array.from({length:100},(s,u)=>u+1),d=(s,u)=>`${20+s/100*(i-40)},${l-20-u*(l-40)/2}`,m=r.map((s,u)=>`${u===0?"M":"L"}${d(s,c(s))}`).join(" "),o=r.map((s,u)=>`${u===0?"M":"L"}${d(s,p(s))}`).join(" "),t=p(n)-c(n);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Training vs Validation Loss"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Epoch: ",n,e.jsx("input",{type:"range",min:1,max:100,step:1,value:n,onChange:s=>f(parseInt(s.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("svg",{width:i,height:l,className:"mx-auto block",children:[e.jsx("line",{x1:20,y1:l-20,x2:i-20,y2:l-20,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("line",{x1:20,y1:0,x2:20,y2:l-20,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("path",{d:m,fill:"none",stroke:"#8b5cf6",strokeWidth:2.5}),e.jsx("path",{d:o,fill:"none",stroke:"#f97316",strokeWidth:2.5}),e.jsx("line",{x1:20+n/100*(i-40),y1:0,x2:20+n/100*(i-40),y2:l-20,stroke:"#9ca3af",strokeWidth:.8,strokeDasharray:"3,3"})]}),e.jsxs("div",{className:"mt-2 flex justify-center gap-6 text-xs",children:[e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-violet-500"})," Train: ",c(n).toFixed(3)]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-orange-500"})," Val: ",p(n).toFixed(3)]}),e.jsxs("span",{className:"font-semibold text-violet-600",children:["Gap: ",t.toFixed(3)]})]})]})}function I(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Detecting overfitting early is crucial for training effective models. The primary diagnostic tool is monitoring training and validation loss curves throughout training."}),e.jsxs(j,{title:"Generalization Gap",children:[e.jsx(a.BlockMath,{math:"\\text{Gap} = \\mathcal{L}_{\\text{val}} - \\mathcal{L}_{\\text{train}}"}),e.jsx("p",{className:"mt-2",children:"A growing gap between validation and training loss signals overfitting. The model is memorizing training data instead of learning general patterns."})]}),e.jsx(D,{}),e.jsx(v,{title:"Diagnostic Checklist",children:e.jsxs("ul",{className:"list-disc ml-4 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Both losses high"}),": underfitting (increase capacity or train longer)"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Train low, val high"}),": overfitting (add regularization)"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Both losses low and close"}),": good fit"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Val loss oscillates"}),": learning rate may be too high"]})]})}),e.jsx(z,{title:"Common Pitfall: Data Leakage",children:e.jsxs("p",{children:["If validation loss is ",e.jsx("em",{children:"lower"})," than training loss, suspect data leakage or incorrect data splitting. Ensure no overlap between train and validation sets, and that preprocessing is fit only on training data."]})}),e.jsx(_,{title:"Tracking Overfitting in PyTorch",code:`import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(10, 128), nn.ReLU(), nn.Linear(128, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Simulated data
x_train, y_train = torch.randn(500, 10), torch.randn(500, 1)
x_val, y_val = torch.randn(100, 10), torch.randn(100, 1)

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(200):
    model.train()
    train_loss = criterion(model(x_train), y_train)
    optimizer.zero_grad(); train_loss.backward(); optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(x_val), y_val)

    gap = val_loss.item() - train_loss.item()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} gap={gap:.4f}")`}),e.jsx(k,{type:"note",title:"Beyond Loss Curves",children:e.jsx("p",{children:"Also monitor task-specific metrics (accuracy, F1, BLEU) on validation data. Weight norms, gradient magnitudes, and activation distributions provide additional insight into model health during training."})})]})}const he=Object.freeze(Object.defineProperty({__proto__:null,default:I},Symbol.toStringTag,{value:"Module"}));function W(){const[n,f]=b.useState(50),i=420,l=220,c=t=>t<40?1.2-.015*t:t<60?.6+.04*(t-40):1.4*Math.exp(-.02*(t-60))+.2,p=t=>t<50?Math.max(.01,.8-.016*t):.01,r=Array.from({length:100},(t,s)=>s+1),d=(t,s)=>`${25+t/100*(i-50)},${l-25-s*(l-45)/2}`,m=r.map((t,s)=>`${s===0?"M":"L"}${d(t,c(t))}`).join(" "),o=r.map((t,s)=>`${s===0?"M":"L"}${d(t,p(t))}`).join(" ");return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Double Descent Curve"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Model Parameters: ",n,"x",e.jsx("input",{type:"range",min:1,max:100,step:1,value:n,onChange:t=>f(parseInt(t.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("svg",{width:i,height:l,className:"mx-auto block",children:[e.jsx("line",{x1:25,y1:l-25,x2:i-25,y2:l-25,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("line",{x1:25,y1:5,x2:25,y2:l-25,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("line",{x1:25+50/100*(i-50),y1:5,x2:25+50/100*(i-50),y2:l-25,stroke:"#ef4444",strokeWidth:.8,strokeDasharray:"4,4",opacity:.5}),e.jsx("text",{x:25+50/100*(i-50),y:15,textAnchor:"middle",fontSize:9,fill:"#ef4444",children:"interpolation"}),e.jsx("path",{d:o,fill:"none",stroke:"#8b5cf6",strokeWidth:2,strokeDasharray:"4,4"}),e.jsx("path",{d:m,fill:"none",stroke:"#f97316",strokeWidth:2.5}),e.jsx("line",{x1:25+n/100*(i-50),y1:5,x2:25+n/100*(i-50),y2:l-25,stroke:"#9ca3af",strokeWidth:.8,strokeDasharray:"3,3"}),e.jsx("circle",{cx:25+n/100*(i-50),cy:parseFloat(d(n,c(n)).split(",")[1]),r:4,fill:"#f97316"})]}),e.jsxs("div",{className:"mt-2 flex justify-center gap-6 text-xs",children:[e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-orange-500"})," Test Risk: ",c(n).toFixed(3)]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-violet-500 opacity-60",style:{borderTop:"1px dashed"}})," Train Risk"]})]})]})}function C(){return e.jsxs("div",{className:"space-y-6",children:[e.jsxs("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:["The double descent phenomenon challenges the classical U-shaped bias-variance curve. As model complexity increases past the interpolation threshold, test error can actually",e.jsx("em",{children:" decrease again"}),", contradicting traditional wisdom."]}),e.jsx(j,{title:"Interpolation Threshold",children:e.jsxs("p",{children:["The interpolation threshold is the point where the model has just enough parameters to perfectly fit (",e.jsx(a.InlineMath,{math:"\\hat{f}(x_i) = y_i"})," for all training points). At this threshold, the model is maximally sensitive to noise, causing a peak in test error."]})}),e.jsx(W,{}),e.jsxs(M,{title:"Double Descent Regions",id:"double-descent-regions",children:[e.jsx("p",{children:"The test risk curve has three distinct regions:"}),e.jsx(a.BlockMath,{math:"\\text{Risk}(p) = \\begin{cases} \\text{decreasing (classical)} & p \\ll n \\\\ \\text{peak at interpolation} & p \\approx n \\\\ \\text{decreasing (modern)} & p \\gg n \\end{cases}"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(a.InlineMath,{math:"p"})," is the number of parameters and ",e.jsx(a.InlineMath,{math:"n"})," is the number of training samples."]})]}),e.jsx(v,{title:"Why Overparameterization Helps",children:e.jsxs("p",{children:["With ",e.jsx(a.InlineMath,{math:"p \\gg n"}),", there are many solutions that interpolate the training data. SGD and implicit regularization select the smoothest among these, which generalizes well. This is why modern networks with billions of parameters can still generalize."]})}),e.jsx(_,{title:"Observing Double Descent with Varying Width",code:`import torch
import torch.nn as nn

n_train, n_test, d = 50, 200, 20
x_train = torch.randn(n_train, d)
y_train = (x_train[:, 0] > 0).float().unsqueeze(1)
x_test = torch.randn(n_test, d)
y_test = (x_test[:, 0] > 0).float().unsqueeze(1)

for width in [10, 50, 100, 500, 2000]:
    model = nn.Sequential(nn.Linear(d, width), nn.ReLU(), nn.Linear(width, 1), nn.Sigmoid())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(2000):
        loss = nn.BCELoss()(model(x_train), y_train)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        test_loss = nn.BCELoss()(model(x_test), y_test)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Width={width:4d} Params={n_params:6d} TestLoss={test_loss:.4f}")`}),e.jsx(k,{type:"note",title:"Epoch-Wise Double Descent",children:e.jsxs("p",{children:["Double descent also occurs along the training time axis: test error can first decrease, then increase (classical overfitting), then decrease again with longer training. This is called ",e.jsx("strong",{children:"epoch-wise double descent"})," and is especially pronounced with label noise."]})})]})}const me=Object.freeze(Object.defineProperty({__proto__:null,default:C},Symbol.toStringTag,{value:"Module"}));function R(){const[n,f]=b.useState(1),[i,l]=b.useState("L2"),c=300,p=300,r=c/2,d=p/2,m=60,o=Array.from({length:201},(h,x)=>{const g=x/200*2*Math.PI,y=n*m,N=Math.abs(Math.cos(g)),S=Math.abs(Math.sin(g)),w=y/(N+S||1);return{x:r+Math.cos(g)*w,y:d-Math.sin(g)*w}}),t=Array.from({length:201},(h,x)=>{const g=x/200*2*Math.PI,y=n*m;return{x:r+Math.cos(g)*y,y:d-Math.sin(g)*y}}),u=(i==="L1"?o:t).map((h,x)=>`${x===0?"M":"L"}${h.x},${h.y}`).join(" ")+"Z";return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Constraint Region Geometry"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3",children:[e.jsx("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:e.jsxs("select",{value:i,onChange:h=>l(h.target.value),className:"rounded border px-2 py-1 text-sm dark:bg-gray-800 dark:border-gray-600",children:[e.jsx("option",{value:"L1",children:"L1 (Lasso)"}),e.jsx("option",{value:"L2",children:"L2 (Ridge)"})]})}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:[e.jsx(a.InlineMath,{math:"\\lambda"})," = ",n.toFixed(1),e.jsx("input",{type:"range",min:.2,max:2,step:.1,value:n,onChange:h=>f(parseFloat(h.target.value)),className:"w-32 accent-violet-500"})]})]}),e.jsxs("svg",{width:c,height:p,className:"mx-auto block",children:[e.jsx("line",{x1:0,y1:d,x2:c,y2:d,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("line",{x1:r,y1:0,x2:r,y2:p,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("path",{d:u,fill:"rgba(139,92,246,0.1)",stroke:"#8b5cf6",strokeWidth:2}),e.jsx("ellipse",{cx:r+60,cy:d-40,rx:80,ry:50,fill:"none",stroke:"#f97316",strokeWidth:1.5,strokeDasharray:"4,4",transform:`rotate(-30 ${r+60} ${d-40})`})]}),e.jsx("p",{className:"text-xs text-center text-gray-500 mt-2",children:"Violet: constraint region. Orange: loss contours (ellipses)."})]})}function P(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Weight regularization adds a penalty on model parameters to the loss function, discouraging overly complex models and improving generalization."}),e.jsxs(j,{title:"L2 Regularization (Ridge / Weight Decay)",children:[e.jsx(a.BlockMath,{math:"\\mathcal{L}_{\\text{reg}} = \\mathcal{L}_{\\text{data}} + \\frac{\\lambda}{2} \\|\\mathbf{w}\\|_2^2 = \\mathcal{L}_{\\text{data}} + \\frac{\\lambda}{2} \\sum_i w_i^2"}),e.jsx("p",{className:"mt-2",children:"Penalizes large weights, encouraging small distributed values. Bayesian interpretation: Gaussian prior on weights."})]}),e.jsxs(j,{title:"L1 Regularization (Lasso)",children:[e.jsx(a.BlockMath,{math:"\\mathcal{L}_{\\text{reg}} = \\mathcal{L}_{\\text{data}} + \\lambda \\|\\mathbf{w}\\|_1 = \\mathcal{L}_{\\text{data}} + \\lambda \\sum_i |w_i|"}),e.jsx("p",{className:"mt-2",children:"Encourages sparsity (many weights become exactly zero). Bayesian interpretation: Laplace prior on weights."})]}),e.jsx(R,{}),e.jsxs(M,{title:"Why L1 Produces Sparsity",id:"l1-sparsity",children:[e.jsx("p",{children:"The L1 constraint region is a diamond whose corners lie on the axes. Loss contour ellipses are more likely to touch the diamond at a corner where one coordinate is zero, producing sparse solutions:"}),e.jsx(a.BlockMath,{math:"\\text{argmin}_{\\|\\mathbf{w}\\|_1 \\leq t} \\mathcal{L}(\\mathbf{w}) \\text{ is more likely to have } w_i = 0"})]}),e.jsx(_,{title:"L1 & L2 Regularization in PyTorch",code:`import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
lam_l2, lam_l1 = 1e-4, 1e-5

x, y = torch.randn(200, 20), torch.randn(200, 1)

for epoch in range(100):
    pred = model(x)
    loss = criterion(pred, y)

    # L2 penalty
    l2_reg = sum(p.pow(2).sum() for p in model.parameters())
    # L1 penalty
    l1_reg = sum(p.abs().sum() for p in model.parameters())

    total_loss = loss + lam_l2 * l2_reg + lam_l1 * l1_reg
    optimizer.zero_grad(); total_loss.backward(); optimizer.step()

# Count near-zero weights (sparsity from L1)
n_sparse = sum((p.abs() < 1e-3).sum().item() for p in model.parameters())
n_total = sum(p.numel() for p in model.parameters())
print(f"Near-zero weights: {n_sparse}/{n_total} ({100*n_sparse/n_total:.1f}%)")`}),e.jsx(k,{type:"note",title:"Elastic Net",children:e.jsxs("p",{children:["Elastic Net combines both: ",e.jsx(a.InlineMath,{math:"\\lambda_1 \\|\\mathbf{w}\\|_1 + \\lambda_2 \\|\\mathbf{w}\\|_2^2"}),". This provides sparsity from L1 while maintaining the grouping effect of L2 for correlated features."]})})]})}const pe=Object.freeze(Object.defineProperty({__proto__:null,default:P},Symbol.toStringTag,{value:"Module"}));function B(){const[n,f]=b.useState(0),i=400,l=200,c=t=>2*Math.exp(-.03*t)*(1+.15*Math.sin(t*.3)),p=t=>2*Math.exp(-.04*t)*(1+.05*Math.sin(t*.2)),r=Array.from({length:100},(t,s)=>s),d=(t,s)=>`${25+t/100*(i-50)},${l-20-s/2.2*(l-40)}`,m=r.map((t,s)=>`${s===0?"M":"L"}${d(t,c(t))}`).join(" "),o=r.map((t,s)=>`${s===0?"M":"L"}${d(t,p(t))}`).join(" ");return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Adam + L2 vs AdamW Weight Norms"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Training Step: ",n,e.jsx("input",{type:"range",min:0,max:99,step:1,value:n,onChange:t=>f(parseInt(t.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("svg",{width:i,height:l,className:"mx-auto block",children:[e.jsx("line",{x1:25,y1:l-20,x2:i-25,y2:l-20,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("path",{d:m,fill:"none",stroke:"#f97316",strokeWidth:2.5}),e.jsx("path",{d:o,fill:"none",stroke:"#8b5cf6",strokeWidth:2.5}),e.jsx("line",{x1:25+n/100*(i-50),y1:5,x2:25+n/100*(i-50),y2:l-20,stroke:"#9ca3af",strokeWidth:.8,strokeDasharray:"3,3"})]}),e.jsxs("div",{className:"mt-2 flex justify-center gap-6 text-xs",children:[e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-orange-500"})," Adam+L2: ",c(n).toFixed(3)]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-violet-500"})," AdamW: ",p(n).toFixed(3)]})]})]})}function q(){return e.jsxs("div",{className:"space-y-6",children:[e.jsxs("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:["For SGD, L2 regularization and weight decay are mathematically equivalent. However, for adaptive optimizers like Adam, they diverge — leading to the important distinction of ",e.jsx("strong",{children:"decoupled weight decay"}),"."]}),e.jsxs(j,{title:"L2 Regularization Update",children:[e.jsx(a.BlockMath,{math:"\\nabla_w \\mathcal{L}_{\\text{reg}} = \\nabla_w \\mathcal{L} + \\lambda w"}),e.jsxs("p",{className:"mt-2",children:["The gradient of the L2 penalty is added to the loss gradient ",e.jsx("em",{children:"before"})," the optimizer processes it."]})]}),e.jsxs(j,{title:"Decoupled Weight Decay (AdamW)",children:[e.jsx(a.BlockMath,{math:"w_{t+1} = w_t - \\eta \\left( \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon} + \\lambda w_t \\right)"}),e.jsxs("p",{className:"mt-2",children:["Weight decay is applied ",e.jsx("em",{children:"directly"})," to the weights, bypassing Adam's adaptive scaling of gradients. This preserves the intended regularization strength."]})]}),e.jsxs(M,{title:"Why L2 Fails with Adam",id:"l2-adam-failure",children:[e.jsxs("p",{children:["In Adam, gradients are divided by ",e.jsx(a.InlineMath,{math:"\\sqrt{\\hat{v}_t}"}),", which scales each parameter's update by the inverse of its historical gradient magnitude. When L2 gradient ",e.jsx(a.InlineMath,{math:"\\lambda w"})," is added before this scaling:"]}),e.jsx(a.BlockMath,{math:"\\text{Effective decay} = \\frac{\\lambda w}{\\sqrt{\\hat{v}_t} + \\epsilon} \\neq \\lambda w"}),e.jsxs("p",{className:"mt-2",children:["Parameters with large gradients get ",e.jsx("em",{children:"less"})," regularization, defeating the purpose."]})]}),e.jsx(B,{}),e.jsx(v,{title:"Practical Impact",children:e.jsxs("p",{children:["The AdamW paper showed that decoupled weight decay leads to better generalization and more stable training. This is why AdamW has become the default optimizer for training transformers, with typical ",e.jsx(a.InlineMath,{math:"\\lambda"})," values of 0.01 to 0.1."]})}),e.jsx(_,{title:"AdamW vs Adam+L2 in PyTorch",code:`import torch
import torch.nn as nn

model_adamw = nn.Linear(100, 10)
model_adam_l2 = nn.Linear(100, 10)
model_adam_l2.load_state_dict(model_adamw.state_dict())

# AdamW: decoupled weight decay
opt_adamw = torch.optim.AdamW(model_adamw.parameters(), lr=1e-3, weight_decay=0.01)

# Adam + L2: weight decay applied to gradients (NOT decoupled)
opt_adam = torch.optim.Adam(model_adam_l2.parameters(), lr=1e-3, weight_decay=0.01)

x = torch.randn(32, 100)
for step in range(200):
    loss1 = model_adamw(x).pow(2).mean()
    opt_adamw.zero_grad(); loss1.backward(); opt_adamw.step()

    loss2 = model_adam_l2(x).pow(2).mean()
    opt_adam.zero_grad(); loss2.backward(); opt_adam.step()

w_adamw = model_adamw.weight.norm().item()
w_adam = model_adam_l2.weight.norm().item()
print(f"AdamW weight norm: {w_adamw:.4f}")
print(f"Adam+L2 weight norm: {w_adam:.4f}")
print(f"AdamW produces smaller weights: {w_adamw < w_adam}")`}),e.jsx(z,{title:"Always Use AdamW for Adaptive Optimizers",children:e.jsxs("p",{children:["When using Adam, AdaGrad, or RMSProp, always use the decoupled weight decay variant (e.g., ",e.jsx("code",{children:"torch.optim.AdamW"}),"). Using ",e.jsx("code",{children:"weight_decay"})," in standard",e.jsx("code",{children:" torch.optim.Adam"})," applies L2 regularization, not true weight decay."]})})]})}const xe=Object.freeze(Object.defineProperty({__proto__:null,default:q},Symbol.toStringTag,{value:"Module"}));function E(){const[n,f]=b.useState(1),i=[[2,1],[1,3]],l=d=>{let m=[1/Math.sqrt(2),1/Math.sqrt(2)],o=0;for(let t=0;t<d;t++){const s=[i[0][0]*m[0]+i[0][1]*m[1],i[1][0]*m[0]+i[1][1]*m[1]];o=Math.sqrt(s[0]*s[0]+s[1]*s[1]),m=[s[0]/o,s[1]/o]}return{sigma:o,u:m}},{sigma:c,u:p}=l(n),r=3.618;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Power Iteration for Spectral Norm"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Iterations: ",n,e.jsx("input",{type:"range",min:1,max:20,step:1,value:n,onChange:d=>f(parseInt(d.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("div",{className:"grid grid-cols-2 gap-4 text-sm text-gray-700 dark:text-gray-300",children:[e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3",children:[e.jsxs("p",{className:"font-semibold text-violet-700 dark:text-violet-300",children:["Estimated ",e.jsx(a.InlineMath,{math:"\\sigma_1"})]}),e.jsx("p",{className:"text-2xl font-mono",children:c.toFixed(4)}),e.jsxs("p",{className:"text-xs text-gray-500",children:["True value: ",r.toFixed(3)]})]}),e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3",children:[e.jsx("p",{className:"font-semibold text-violet-700 dark:text-violet-300",children:"Top Singular Vector"}),e.jsxs("p",{className:"text-lg font-mono",children:["[",p[0].toFixed(4),", ",p[1].toFixed(4),"]"]}),e.jsxs("p",{className:"text-xs text-gray-500",children:["Error: ",Math.abs(c-r).toFixed(6)]})]})]})]})}function F(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Spectral normalization constrains the Lipschitz constant of each layer by normalizing weight matrices by their largest singular value. This is critical for training stable GANs and controlling function smoothness."}),e.jsxs(j,{title:"Spectral Norm",children:[e.jsx(a.BlockMath,{math:"\\sigma(W) = \\max_{\\|h\\|_2 \\leq 1} \\|Wh\\|_2 = \\sigma_1(W)"}),e.jsx("p",{className:"mt-2",children:"The spectral norm of a matrix is its largest singular value, which equals the maximum factor by which the matrix can stretch a vector."})]}),e.jsxs(M,{title:"Lipschitz Constraint via Spectral Normalization",id:"spectral-lipschitz",children:[e.jsxs("p",{children:["Spectral normalization replaces ",e.jsx(a.InlineMath,{math:"W"})," with ",e.jsx(a.InlineMath,{math:"\\bar{W}"}),":"]}),e.jsx(a.BlockMath,{math:"\\bar{W} = \\frac{W}{\\sigma(W)}"}),e.jsxs("p",{className:"mt-2",children:["For a network ",e.jsx(a.InlineMath,{math:"f = f_L \\circ \\cdots \\circ f_1"})," with each layer spectrally normalized, the global Lipschitz constant is bounded:"]}),e.jsx(a.BlockMath,{math:"\\|f(x) - f(y)\\|_2 \\leq \\prod_{l=1}^L \\sigma(\\bar{W}_l) = 1"})]}),e.jsx(E,{}),e.jsx(v,{title:"Why One Step Suffices",children:e.jsx("p",{children:"In practice, a single power iteration step per training step is sufficient because the weight matrix changes slowly between updates. The singular vector estimate from the previous step is a warm start, converging quickly to the true value."})}),e.jsx(_,{title:"Spectral Normalization in PyTorch",code:`import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# Apply spectral normalization to a layer
layer = nn.Linear(64, 32)
sn_layer = spectral_norm(layer)

# Check that spectral norm is approximately 1
x = torch.randn(16, 64)
W = sn_layer.weight
U, S, V = torch.linalg.svd(W)
print(f"Largest singular value: {S[0].item():.4f}")  # ~1.0

# GAN discriminator with spectral normalization
class SNDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(784, 256)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 1)),
        )
    def forward(self, x):
        return self.net(x)

disc = SNDiscriminator()
print(f"Discriminator params: {sum(p.numel() for p in disc.parameters())}")`}),e.jsx(k,{type:"note",title:"Beyond GANs",children:e.jsx("p",{children:"Spectral normalization is also used in diffusion models, contrastive learning, and any setting where controlling the Lipschitz constant of a network is desirable. It can be combined with other regularization techniques like dropout and weight decay."})})]})}const ge=Object.freeze(Object.defineProperty({__proto__:null,default:F},Symbol.toStringTag,{value:"Module"}));function V(){const[n,f]=b.useState(.5),[i,l]=b.useState(0),c=[4,6,6,3],p=360,r=220,d=c.map((t,s)=>50+s*((p-100)/(c.length-1))),m=(t,s)=>Math.sin(i*1e3+t*137+s*73)*.5+.5,o=(t,s)=>t>0&&t<c.length-1&&m(t,s)<n;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Dropout Visualization"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["p = ",n.toFixed(1),e.jsx("input",{type:"range",min:0,max:.9,step:.1,value:n,onChange:t=>f(parseFloat(t.target.value)),className:"w-32 accent-violet-500"})]}),e.jsx("button",{onClick:()=>l(t=>t+1),className:"rounded bg-violet-500 px-3 py-1 text-xs text-white hover:bg-violet-600",children:"Resample"})]}),e.jsxs("svg",{width:p,height:r,className:"mx-auto block",children:[c.map((t,s)=>Array.from({length:t},(h,x)=>r/2-(t-1)*25/2+x*25).map((h,x)=>{const g=o(s,x);if(s<c.length-1){const y=c[s+1];return Array.from({length:y},(S,w)=>r/2-(y-1)*25/2+w*25).map((S,w)=>{const L=o(s+1,w);return g||L?null:e.jsx("line",{x1:d[s],y1:h,x2:d[s+1],y2:S,stroke:"#d1d5db",strokeWidth:.5},`e${s}${x}${w}`)})}return null})),c.map((t,s)=>Array.from({length:t},(h,x)=>r/2-(t-1)*25/2+x*25).map((h,x)=>{const g=o(s,x);return e.jsx("circle",{cx:d[s],cy:h,r:8,fill:g?"#e5e7eb":"#8b5cf6",stroke:g?"#9ca3af":"#7c3aed",strokeWidth:1.5,opacity:g?.4:1},`n${s}${x}`)}))]}),e.jsxs("p",{className:"text-xs text-center text-gray-500 mt-2",children:["Active neurons: ",c.slice(1,-1).reduce((t,s,u)=>t+Array.from({length:s},(h,x)=>!o(u+1,x)).filter(Boolean).length,0)," / ",c.slice(1,-1).reduce((t,s)=>t+s,0)," hidden"]})]})}function $(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Dropout is one of the most effective and widely used regularization techniques. During training, it randomly deactivates neurons, preventing co-adaptation and acting as an implicit ensemble of subnetworks."}),e.jsxs(j,{title:"Dropout",children:[e.jsxs("p",{children:["During training, each hidden unit is independently set to zero with probability ",e.jsx(a.InlineMath,{math:"p"}),":"]}),e.jsx(a.BlockMath,{math:"\\tilde{h}_i = \\begin{cases} 0 & \\text{with probability } p \\\\ \\frac{h_i}{1-p} & \\text{with probability } 1-p \\end{cases}"}),e.jsxs("p",{className:"mt-2",children:["The scaling by ",e.jsx(a.InlineMath,{math:"1/(1-p)"})," is called ",e.jsx("strong",{children:"inverted dropout"})," and ensures expected values match at test time."]})]}),e.jsx(V,{}),e.jsxs(M,{title:"Ensemble Interpretation",id:"dropout-ensemble",children:[e.jsxs("p",{children:["A network with ",e.jsx(a.InlineMath,{math:"n"})," droppable units implicitly trains ",e.jsx(a.InlineMath,{math:"2^n"})," subnetworks that share weights. At test time, using all units with scaled weights approximates the geometric mean of all subnetwork predictions:"]}),e.jsx(a.BlockMath,{math:"f_{\\text{test}}(x) \\approx \\left(\\prod_{m=1}^{2^n} f_m(x)^{1/2^n}\\right)"})]}),e.jsx(v,{title:"Typical Dropout Rates",children:e.jsxs("ul",{className:"list-disc ml-4 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Input layer"}),": ",e.jsx(a.InlineMath,{math:"p = 0.2"})," (drop 20% of inputs)"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Hidden layers"}),": ",e.jsx(a.InlineMath,{math:"p = 0.5"})," (the original default)"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Transformers"}),": ",e.jsx(a.InlineMath,{math:"p = 0.1"})," (attention and FFN sublayers)"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Convolutional layers"}),": often not used (use Spatial Dropout instead)"]})]})}),e.jsx(_,{title:"Dropout in PyTorch",code:`import torch
import torch.nn as nn

class RegularizedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=drop_p),  # inverted dropout by default
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=drop_p),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

model = RegularizedMLP(784, 256, 10)
x = torch.randn(32, 784)

model.train()   # dropout active
out_train = model(x)

model.eval()    # dropout disabled, weights scaled
out_eval = model(x)

print(f"Train output variance: {out_train.var():.4f}")
print(f"Eval output variance: {out_eval.var():.4f}")`}),e.jsx(k,{type:"note",title:"Dropout and Batch Normalization",children:e.jsxs("p",{children:["Combining dropout with batch normalization requires care. The variance shift from dropout at train time vs test time can conflict with batch norm statistics. In practice, many modern architectures use batch norm ",e.jsx("em",{children:"without"})," dropout, or apply dropout only after the final batch norm layer."]})})]})}const ue=Object.freeze(Object.defineProperty({__proto__:null,default:$},Symbol.toStringTag,{value:"Module"}));function O(){const[n,f]=b.useState("dropout"),[i,l]=b.useState(0),c=360,p=180,r=4,d=4,m=Array.from({length:r},(g,y)=>30+y*35),o=Array.from({length:d},(g,y)=>30+y*35),t=80,s=280,u=(g,y)=>Math.sin(i*997+g*131+y*67)*.5+.5,h=g=>u(g,999)<.5,x=(g,y)=>u(g,y)<.5;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Dropout vs DropConnect"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3",children:[e.jsx("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:e.jsxs("select",{value:n,onChange:g=>f(g.target.value),className:"rounded border px-2 py-1 text-sm dark:bg-gray-800 dark:border-gray-600",children:[e.jsx("option",{value:"dropout",children:"Dropout (nodes)"}),e.jsx("option",{value:"dropconnect",children:"DropConnect (edges)"})]})}),e.jsx("button",{onClick:()=>l(g=>g+1),className:"rounded bg-violet-500 px-3 py-1 text-xs text-white hover:bg-violet-600",children:"Resample"})]}),e.jsxs("svg",{width:c,height:p,className:"mx-auto block",children:[m.map((g,y)=>o.map((N,S)=>{const w=n==="dropout"?h(y):x(y,S);return e.jsx("line",{x1:t,y1:g,x2:s,y2:N,stroke:w?"#e5e7eb":"#8b5cf6",strokeWidth:w?.5:1.5,opacity:w?.3:.8},`e${y}${S}`)})),m.map((g,y)=>{const N=n==="dropout"&&h(y);return e.jsx("circle",{cx:t,cy:g,r:10,fill:N?"#e5e7eb":"#8b5cf6",stroke:N?"#9ca3af":"#7c3aed",strokeWidth:1.5,opacity:N?.4:1},`s${y}`)}),o.map((g,y)=>e.jsx("circle",{cx:s,cy:g,r:10,fill:"#8b5cf6",stroke:"#7c3aed",strokeWidth:1.5},`d${y}`)),e.jsx("text",{x:t,y:p-5,textAnchor:"middle",fontSize:10,fill:"#6b7280",children:"Source"}),e.jsx("text",{x:s,y:p-5,textAnchor:"middle",fontSize:10,fill:"#6b7280",children:"Dest"})]})]})}function H(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"DropConnect and DropPath extend the dropout idea by operating at the connection or path level rather than the neuron level, offering finer-grained regularization."}),e.jsxs(j,{title:"DropConnect",children:[e.jsx("p",{children:"Instead of zeroing activations, DropConnect zeros individual weights:"}),e.jsx(a.BlockMath,{math:"\\tilde{W}_{ij} = \\begin{cases} 0 & \\text{with probability } p \\\\ W_{ij} & \\text{with probability } 1-p \\end{cases}"}),e.jsxs("p",{className:"mt-2",children:["Each connection is independently dropped, giving ",e.jsx(a.InlineMath,{math:"2^{n \\times m}"})," possible subnetworks for an ",e.jsx(a.InlineMath,{math:"n \\times m"})," weight matrix."]})]}),e.jsx(O,{}),e.jsxs(j,{title:"DropPath (Stochastic Depth)",children:[e.jsx("p",{children:"In residual networks, entire layers (paths) are randomly skipped during training:"}),e.jsx(a.BlockMath,{math:"x_{l+1} = x_l + b_l \\cdot f_l(x_l), \\quad b_l \\sim \\text{Bernoulli}(1 - p_l)"}),e.jsxs("p",{className:"mt-2",children:["Typically ",e.jsx(a.InlineMath,{math:"p_l"})," increases linearly with depth: earlier layers are dropped less often since they learn fundamental features."]})]}),e.jsxs(M,{title:"Linear Survival Schedule",id:"linear-survival",children:[e.jsxs("p",{children:["For a network with ",e.jsx(a.InlineMath,{math:"L"})," residual blocks, the survival probability of layer ",e.jsx(a.InlineMath,{math:"l"})," is:"]}),e.jsx(a.BlockMath,{math:"p_{\\text{survive}}(l) = 1 - \\frac{l}{L}(1 - p_L)"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(a.InlineMath,{math:"p_L"})," is the survival probability of the last layer (typically 0.8)."]})]}),e.jsx(v,{title:"Stochastic Depth in Practice",children:e.jsx("p",{children:"Stochastic depth is essential in modern architectures like Vision Transformers (ViT) and ConvNeXt. For ViT-Large with 24 blocks, a typical drop path rate of 0.1-0.3 significantly improves generalization while also reducing training time by ~25%."})}),e.jsx(_,{title:"DropPath in PyTorch",code:`import torch
import torch.nn as nn

class DropPath(nn.Module):
    """Stochastic Depth: drop entire residual branches."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        # Shape: (batch, 1, 1, ...) for broadcasting
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
        return x * mask / keep_prob

class ResidualBlock(nn.Module):
    def __init__(self, dim, drop_path_rate=0.1):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        return x + self.drop_path(self.fc(x))

# Example: 12-block network with linear drop path schedule
depth = 12
dpr = [0.1 * i / (depth - 1) for i in range(depth)]
blocks = nn.Sequential(*[ResidualBlock(128, dp) for dp in dpr])
print(f"Drop rates: {[f'{r:.3f}' for r in dpr]}")`}),e.jsx(k,{type:"note",title:"Choosing Between Variants",children:e.jsxs("p",{children:[e.jsx("strong",{children:"Dropout"}),": general purpose, works well in MLPs and attention layers.",e.jsx("strong",{children:"DropConnect"}),": finer-grained but more expensive. ",e.jsx("strong",{children:"DropPath"}),": specifically designed for residual networks and now standard in transformers."]})})]})}const fe=Object.freeze(Object.defineProperty({__proto__:null,default:H},Symbol.toStringTag,{value:"Module"}));function G(){const[n,f]=b.useState(.1),i=5,l=380,c=160,p=50,r=15,d=[0,0,1,0,0],m=d.map(t=>t*(1-n)+n/i),o=(l-i*(p+r))/2;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Label Smoothing Visualization"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:[e.jsx(a.InlineMath,{math:"\\epsilon"})," = ",n.toFixed(2),e.jsx("input",{type:"range",min:0,max:.5,step:.01,value:n,onChange:t=>f(parseFloat(t.target.value)),className:"w-40 accent-violet-500"})]}),e.jsx("svg",{width:l,height:c,className:"mx-auto block",children:m.map((t,s)=>{const u=o+s*(p+r),h=t*100;return e.jsxs("g",{children:[e.jsx("rect",{x:u,y:c-30-h,width:p,height:h,fill:"#8b5cf6",rx:3,opacity:.8}),e.jsx("rect",{x:u,y:c-30-d[s]*100,width:p,height:d[s]*100,fill:"none",stroke:"#f97316",strokeWidth:1.5,strokeDasharray:"3,3",rx:3}),e.jsxs("text",{x:u+p/2,y:c-12,textAnchor:"middle",fontSize:10,fill:"#6b7280",children:["Class ",s]}),e.jsx("text",{x:u+p/2,y:c-34-h,textAnchor:"middle",fontSize:9,fill:"#8b5cf6",children:t.toFixed(3)})]},s)})}),e.jsxs("div",{className:"mt-1 flex justify-center gap-4 text-xs text-gray-500",children:[e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-2 bg-violet-500 rounded-sm"})," Smoothed"]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-2 border border-orange-500 rounded-sm"})," Hard"]})]})]})}function U(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Label smoothing and mixup are regularization techniques that soften the training signal, preventing the model from becoming overconfident and improving calibration."}),e.jsxs(j,{title:"Label Smoothing",children:[e.jsxs("p",{children:["Replace hard one-hot targets ",e.jsx(a.InlineMath,{math:"y"})," with smoothed targets:"]}),e.jsx(a.BlockMath,{math:"y_{\\text{smooth}} = (1 - \\epsilon) \\cdot y + \\frac{\\epsilon}{K}"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(a.InlineMath,{math:"\\epsilon"})," is the smoothing parameter (typically 0.1) and",e.jsx(a.InlineMath,{math:"K"})," is the number of classes."]})]}),e.jsx(G,{}),e.jsxs(j,{title:"Mixup",children:[e.jsx("p",{children:"Create virtual training examples by interpolating pairs:"}),e.jsx(a.BlockMath,{math:"\\tilde{x} = \\lambda x_i + (1 - \\lambda) x_j, \\quad \\tilde{y} = \\lambda y_i + (1 - \\lambda) y_j"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(a.InlineMath,{math:"\\lambda \\sim \\text{Beta}(\\alpha, \\alpha)"})," and ",e.jsx(a.InlineMath,{math:"\\alpha"})," controls interpolation strength (typically 0.2-0.4)."]})]}),e.jsxs(M,{title:"CutMix",id:"cutmix",children:[e.jsx("p",{children:"CutMix replaces a rectangular region of one image with a patch from another:"}),e.jsx(a.BlockMath,{math:"\\tilde{x} = \\mathbf{M} \\odot x_i + (1 - \\mathbf{M}) \\odot x_j"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(a.InlineMath,{math:"\\mathbf{M}"})," is a binary mask. The label is mixed proportionally to the area ratio: ",e.jsx(a.InlineMath,{math:"\\tilde{y} = \\lambda y_i + (1-\\lambda) y_j"})," where",e.jsx(a.InlineMath,{math:"\\lambda"})," is the fraction of the unmasked area."]})]}),e.jsx(v,{title:"Benefits of Soft Targets",children:e.jsxs("ul",{className:"list-disc ml-4 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Better calibration"}),": model probabilities reflect true uncertainty"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Reduced overconfidence"}),": logits don't grow unboundedly"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Knowledge distillation"}),": dark knowledge in soft targets carries inter-class similarities"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Label noise robustness"}),": smoothing reduces impact of mislabeled examples"]})]})}),e.jsx(_,{title:"Label Smoothing & Mixup in PyTorch",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

# Label Smoothing Cross-Entropy (built-in since PyTorch 1.10)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

logits = torch.randn(8, 10)  # batch=8, classes=10
targets = torch.randint(0, 10, (8,))
loss = criterion(logits, targets)
print(f"Label smoothed loss: {loss:.4f}")

# Mixup implementation
def mixup_data(x, y, alpha=0.2):
    lam = torch.distributions.Beta(alpha, alpha).sample()
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def mixup_criterion(pred, y_a, y_b, lam):
    return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)

# Usage in training loop
x = torch.randn(32, 3, 32, 32)
y = torch.randint(0, 10, (32,))
mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
print(f"Mixup lambda: {lam:.4f}")`}),e.jsx(k,{type:"note",title:"Combining Techniques",children:e.jsxs("p",{children:["Label smoothing and mixup are complementary but using both together requires care. CutMix generally outperforms vanilla mixup for image classification. Modern training recipes (e.g., for ViT) often combine label smoothing (",e.jsx(a.InlineMath,{math:"\\epsilon = 0.1"}),") with mixup (",e.jsx(a.InlineMath,{math:"\\alpha = 0.8"}),") and CutMix (",e.jsx(a.InlineMath,{math:"\\alpha = 1.0"}),")."]})})]})}const ye=Object.freeze(Object.defineProperty({__proto__:null,default:U},Symbol.toStringTag,{value:"Module"}));function K(){const[n,f]=b.useState(10),i=400,l=220,c=h=>1.5*Math.exp(-.035*h)+.05,p=h=>1.5*Math.exp(-.025*h)+.2+.002*Math.max(0,h-25),r=Array.from({length:100},(h,x)=>x+1),d=(h,x)=>`${25+h/100*(i-50)},${l-25-x*(l-45)/1.8}`;let m=1,o=1/0;for(let h=1;h<=100;h++){const x=p(h);x<o&&(o=x,m=h)}const t=Math.min(100,m+n),s=r.filter(h=>h<=t).map((h,x)=>`${x===0?"M":"L"}${d(h,c(h))}`).join(" "),u=r.filter(h=>h<=t).map((h,x)=>`${x===0?"M":"L"}${d(h,p(h))}`).join(" ");return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Early Stopping Visualization"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Patience: ",n,e.jsx("input",{type:"range",min:1,max:40,step:1,value:n,onChange:h=>f(parseInt(h.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("svg",{width:i,height:l,className:"mx-auto block",children:[e.jsx("line",{x1:25,y1:l-25,x2:i-25,y2:l-25,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("path",{d:s,fill:"none",stroke:"#8b5cf6",strokeWidth:2.5}),e.jsx("path",{d:u,fill:"none",stroke:"#f97316",strokeWidth:2.5}),e.jsx("line",{x1:25+m/100*(i-50),y1:5,x2:25+m/100*(i-50),y2:l-25,stroke:"#22c55e",strokeWidth:1.5,strokeDasharray:"4,4"}),e.jsx("line",{x1:25+t/100*(i-50),y1:5,x2:25+t/100*(i-50),y2:l-25,stroke:"#ef4444",strokeWidth:1.5,strokeDasharray:"4,4"}),e.jsx("text",{x:25+m/100*(i-50),y:15,textAnchor:"middle",fontSize:9,fill:"#22c55e",children:"best"}),e.jsx("text",{x:25+t/100*(i-50),y:15,textAnchor:"middle",fontSize:9,fill:"#ef4444",children:"stop"})]}),e.jsxs("div",{className:"mt-2 flex justify-center gap-6 text-xs",children:[e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-violet-500"})," Train"]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-orange-500"})," Val"]}),e.jsxs("span",{children:["Best epoch: ",m]}),e.jsxs("span",{children:["Stop epoch: ",t]})]})]})}function X(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Early stopping halts training when validation performance stops improving, preventing the model from overfitting. It acts as a form of implicit regularization by limiting the effective complexity of the learned function."}),e.jsx(j,{title:"Early Stopping",children:e.jsxs("p",{children:["Monitor a validation metric after each epoch. If the metric has not improved for ",e.jsx(a.InlineMath,{math:"P"})," consecutive epochs (the patience), stop training and restore the model weights from the best epoch."]})}),e.jsx(K,{}),e.jsxs(M,{title:"Early Stopping as Regularization",id:"early-stopping-reg",children:[e.jsxs("p",{children:["For linear models with gradient descent, early stopping after ",e.jsx(a.InlineMath,{math:"t"})," steps with learning rate ",e.jsx(a.InlineMath,{math:"\\eta"})," is equivalent to L2 regularization with:"]}),e.jsx(a.BlockMath,{math:"\\lambda_{\\text{eff}} \\approx \\frac{1}{\\eta t}"}),e.jsx("p",{className:"mt-2",children:"Fewer training steps correspond to stronger regularization, limiting how far weights can move from initialization."})]}),e.jsx(v,{title:"Checkpointing Strategy",children:e.jsxs("ul",{className:"list-disc ml-4 space-y-1",children:[e.jsx("li",{children:"Save model weights whenever validation loss reaches a new minimum"}),e.jsx("li",{children:"Track the best metric value and the epoch it occurred"}),e.jsx("li",{children:"After stopping, load the best checkpoint (not the final weights)"}),e.jsx("li",{children:"Typical patience values: 5-20 epochs depending on dataset size"})]})}),e.jsx(_,{title:"Early Stopping with Checkpointing",code:`import torch
import torch.nn as nn

class EarlyStoppingTrainer:
    def __init__(self, model, patience=10, min_delta=1e-4):
        self.model = model
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None

    def check(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            return False  # continue training
        self.counter += 1
        return self.counter >= self.patience  # stop if True

    def restore_best(self):
        if self.best_state:
            self.model.load_state_dict(self.best_state)

# Usage
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
trainer = EarlyStoppingTrainer(model, patience=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(500):
    train_loss = nn.MSELoss()(model(torch.randn(32, 10)), torch.randn(32, 1))
    optimizer.zero_grad(); train_loss.backward(); optimizer.step()
    with torch.no_grad():
        val_loss = nn.MSELoss()(model(torch.randn(32, 10)), torch.randn(32, 1))
    if trainer.check(val_loss.item()):
        print(f"Early stop at epoch {epoch}, best loss: {trainer.best_loss:.4f}")
        trainer.restore_best()
        break`}),e.jsx(k,{type:"note",title:"Patience Selection",children:e.jsx("p",{children:"Too small a patience may stop too early (missing further improvements after a plateau). Too large wastes compute. A good heuristic: set patience to 10-20% of expected total training epochs. Also consider using a learning rate scheduler before early stopping to give the optimizer a chance to escape local minima."})})]})}const je=Object.freeze(Object.defineProperty({__proto__:null,default:X},Symbol.toStringTag,{value:"Module"}));function Y(){const[n,f]=b.useState(5),[i,l]=b.useState(0),c=400,p=140,r=(c-40)/n;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"K-Fold Cross-Validation"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["K = ",n,e.jsx("input",{type:"range",min:2,max:10,step:1,value:n,onChange:d=>{f(parseInt(d.target.value)),l(0)},className:"w-28 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Fold: ",i+1,e.jsx("input",{type:"range",min:0,max:n-1,step:1,value:i,onChange:d=>l(parseInt(d.target.value)),className:"w-28 accent-violet-500"})]})]}),e.jsxs("svg",{width:c,height:p,className:"mx-auto block",children:[Array.from({length:n},(d,m)=>{const o=20+m*r,t=m===i;return e.jsxs("g",{children:[e.jsx("rect",{x:o+1,y:20,width:r-2,height:60,rx:4,fill:t?"#f97316":"#8b5cf6",opacity:.7}),e.jsx("text",{x:o+r/2,y:55,textAnchor:"middle",fontSize:11,fill:"white",fontWeight:"bold",children:t?"Val":"Train"}),e.jsxs("text",{x:o+r/2,y:100,textAnchor:"middle",fontSize:9,fill:"#6b7280",children:["Fold ",m+1]})]},m)}),e.jsxs("text",{x:c/2,y:125,textAnchor:"middle",fontSize:10,fill:"#6b7280",children:["Train: ",((n-1)/n*100).toFixed(0),"% | Val: ",(1/n*100).toFixed(0),"%"]})]})]})}function J(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Cross-validation provides a more robust estimate of model performance by using every data point for both training and validation. However, it poses unique challenges for deep learning due to high computational cost."}),e.jsxs(j,{title:"K-Fold Cross-Validation",children:[e.jsxs("p",{children:["Partition data into ",e.jsx(a.InlineMath,{math:"K"})," equal folds. For each fold ",e.jsx(a.InlineMath,{math:"k"}),", train on ",e.jsx(a.InlineMath,{math:"K-1"})," folds and validate on fold ",e.jsx(a.InlineMath,{math:"k"}),". The final performance estimate is the average:"]}),e.jsx(a.BlockMath,{math:"\\hat{\\mathcal{L}} = \\frac{1}{K} \\sum_{k=1}^K \\mathcal{L}_k"})]}),e.jsx(Y,{}),e.jsxs(M,{title:"Variance of CV Estimate",id:"cv-variance",children:[e.jsx("p",{children:"The variance of the K-fold CV estimator is approximately:"}),e.jsx(a.BlockMath,{math:"\\text{Var}(\\hat{\\mathcal{L}}) \\approx \\frac{\\sigma^2}{K} + \\frac{K-1}{K}\\rho\\sigma^2"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(a.InlineMath,{math:"\\rho"})," is the correlation between fold estimates and ",e.jsx(a.InlineMath,{math:"\\sigma^2"}),"is the per-fold variance. Larger ",e.jsx(a.InlineMath,{math:"K"})," increases the correlation term, so more folds is not always better."]})]}),e.jsx(z,{title:"Challenges for Deep Learning",children:e.jsxs("p",{children:["K-fold CV requires training ",e.jsx(a.InlineMath,{math:"K"})," separate models, each for the full training schedule. For large models (GPT, ViT), this is computationally prohibitive. Alternatives include: single train/val split, bootstrap estimation, or training once and evaluating with multiple random seeds."]})}),e.jsx(v,{title:"When to Use CV for Deep Learning",children:e.jsxs("ul",{className:"list-disc ml-4 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Small datasets"})," (medical imaging, specialized NLP): CV is essential"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Hyperparameter selection"}),": use CV to select, then retrain on full data"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Large-scale pretraining"}),": single split is standard practice"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Competition settings"}),": stratified K-fold is common (K=5 or K=10)"]})]})}),e.jsx(_,{title:"K-Fold Cross-Validation for Deep Learning",code:`import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, TensorDataset
import numpy as np

def kfold_cv(dataset, k=5, epochs=50):
    n = len(dataset)
    indices = np.random.permutation(n)
    fold_size = n // k
    scores = []

    for fold in range(k):
        val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64)

        model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(epochs):
            for xb, yb in train_loader:
                loss = nn.MSELoss()(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        val_loss = np.mean([nn.MSELoss()(model(xb), yb).item() for xb, yb in val_loader])
        scores.append(val_loss)
        print(f"Fold {fold+1}: val_loss = {val_loss:.4f}")

    print(f"Mean: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

# Example usage
X = torch.randn(500, 20); y = torch.randn(500, 1)
dataset = TensorDataset(X, y)
kfold_cv(dataset, k=5, epochs=30)`}),e.jsx(k,{type:"note",title:"Stratified K-Fold",children:e.jsx("p",{children:"For classification, always use stratified K-fold to preserve the class distribution in each fold. This is especially important with imbalanced datasets where random splits may leave some classes underrepresented in certain folds."})})]})}const be=Object.freeze(Object.defineProperty({__proto__:null,default:J},Symbol.toStringTag,{value:"Module"}));function Z(){const[n,f]=b.useState("grid"),i=300,l=300,c=[];for(let o=0;o<5;o++)for(let t=0;t<5;t++)c.push({x:30+o*60,y:30+t*60});const p=[.12,.87,.34,.56,.91,.23,.67,.45,.78,.09,.55,.38,.72,.15,.83,.41,.62,.29,.94,.51,.17,.76,.44,.88,.33],r=p.map((o,t)=>({x:15+o*(i-30),y:15+p[(t+7)%25]*(l-30)})),m=n==="grid"?c:n==="random"?r:[{x:150,y:150},{x:90,y:200},{x:200,y:100},{x:170,y:80},{x:185,y:65},{x:195,y:55},{x:200,y:50},{x:205,y:48},{x:202,y:45},{x:198,y:42},{x:200,y:40},{x:201,y:38}];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Search Strategy Comparison"}),e.jsx("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:e.jsxs("select",{value:n,onChange:o=>f(o.target.value),className:"rounded border px-2 py-1 text-sm dark:bg-gray-800 dark:border-gray-600",children:[e.jsx("option",{value:"grid",children:"Grid Search"}),e.jsx("option",{value:"random",children:"Random Search"}),e.jsx("option",{value:"bayesian",children:"Bayesian Optimization"})]})}),e.jsxs("svg",{width:i,height:l,className:"mx-auto block bg-gray-50 dark:bg-gray-800 rounded",children:[e.jsx("text",{x:i/2,y:l-5,textAnchor:"middle",fontSize:10,fill:"#6b7280",children:"Learning Rate"}),e.jsx("text",{x:10,y:l/2,textAnchor:"middle",fontSize:10,fill:"#6b7280",transform:`rotate(-90 10 ${l/2})`,children:"Weight Decay"}),m.map((o,t)=>e.jsx("circle",{cx:o.x,cy:o.y,r:5,fill:"#8b5cf6",opacity:n==="bayesian"?.3+.7*(t/m.length):.7},t)),n==="bayesian"&&m.length>1&&e.jsx("polyline",{fill:"none",stroke:"#8b5cf6",strokeWidth:1,strokeDasharray:"3,3",opacity:.4,points:m.map(o=>`${o.x},${o.y}`).join(" ")})]}),e.jsx("p",{className:"text-xs text-center text-gray-500 mt-2",children:n==="grid"?"Evenly spaced grid — wastes budget on unimportant dimensions":n==="random"?"Random samples — better coverage of important dimensions":"Bayesian — converges toward optimum guided by surrogate model"})]})}function Q(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Hyperparameter tuning systematically searches for the best configuration of non-learnable parameters (learning rate, weight decay, dropout rate, architecture choices) to maximize validation performance."}),e.jsx(j,{title:"Grid Search",children:e.jsxs("p",{children:["Evaluate all combinations of discrete hyperparameter values. For ",e.jsx(a.InlineMath,{math:"d"})," hyperparameters with ",e.jsx(a.InlineMath,{math:"n"})," values each, this requires ",e.jsx(a.InlineMath,{math:"n^d"})," trials — exponential in dimensionality."]})}),e.jsx(Z,{}),e.jsxs(M,{title:"Random Search Superiority",id:"random-search",children:[e.jsx("p",{children:"Bergstra & Bengio (2012) showed that for hyperparameter spaces where only a few dimensions matter, random search finds good configurations in fewer trials:"}),e.jsx(a.BlockMath,{math:"P(\\text{miss all good values}) = \\left(1 - \\frac{g}{G}\\right)^T"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(a.InlineMath,{math:"g/G"})," is the fraction of the space with good values and",e.jsx(a.InlineMath,{math:"T"})," is the number of trials. Random search explores each dimension more densely than grid search for the same budget."]})]}),e.jsxs(j,{title:"Bayesian Optimization",children:[e.jsx("p",{children:"Fit a surrogate model (typically a Gaussian Process) to the observed (hyperparameter, performance) pairs, then use an acquisition function to choose the next configuration to evaluate:"}),e.jsx(a.BlockMath,{math:"\\mathbf{x}_{\\text{next}} = \\arg\\max_{\\mathbf{x}} \\alpha(\\mathbf{x} \\mid \\mathcal{D}_{1:t})"}),e.jsx("p",{className:"mt-2",children:"Common acquisition functions: Expected Improvement (EI), Upper Confidence Bound (UCB)."})]}),e.jsx(v,{title:"Practical Hyperparameter Ranges",children:e.jsxs("ul",{className:"list-disc ml-4 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Learning rate"}),": log-uniform in ",e.jsx(a.InlineMath,{math:"[10^{-5}, 10^{-1}]"})]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Weight decay"}),": log-uniform in ",e.jsx(a.InlineMath,{math:"[10^{-6}, 10^{-1}]"})]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Dropout"}),": uniform in ",e.jsx(a.InlineMath,{math:"[0.0, 0.5]"})]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Batch size"}),": powers of 2 from 16 to 512"]})]})}),e.jsx(_,{title:"Hyperparameter Search with Optuna",code:`import torch
import torch.nn as nn
# pip install optuna
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    hidden = trial.suggest_categorical('hidden_dim', [64, 128, 256])

    model = nn.Sequential(
        nn.Linear(20, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    x_train, y_train = torch.randn(400, 20), torch.randn(400, 1)
    x_val, y_val = torch.randn(100, 20), torch.randn(100, 1)

    for _ in range(50):
        loss = nn.MSELoss()(model(x_train), y_train)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        val_loss = nn.MSELoss()(model(x_val), y_val).item()
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
print(f"Best params: {study.best_params}")
print(f"Best val loss: {study.best_value:.4f}")`}),e.jsx(k,{type:"note",title:"Successive Halving & Hyperband",children:e.jsx("p",{children:"For expensive deep learning runs, early stopping-based methods like Hyperband allocate more budget to promising configurations. Start many trials with small budgets, then progressively increase the budget for the best performers. This is 10-50x more efficient than standard random search."})})]})}const _e=Object.freeze(Object.defineProperty({__proto__:null,default:Q},Symbol.toStringTag,{value:"Module"}));function ee(){const[n,f]=b.useState("none"),i=200,l=200,c=Array.from({length:8},(r,d)=>Array.from({length:8},(m,o)=>{const t=Math.sqrt((d-3.5)**2+(o-3.5)**2);return t<2.5?"#8b5cf6":t<3.5?"#c4b5fd":"#ede9fe"})),p=(r,d,m)=>{const o=i/8;let t=d*o,s=r*o,u=o,h=o,x=m,g=1;if(n==="hflip"&&(t=i-(d+1)*o),n==="vflip"&&(s=l-(r+1)*o),n==="crop"&&(r<1||r>6||d<1||d>6)&&(g=.15),n==="jitter"){const y=Math.sin(r*3+d*7)*30,N=parseInt(m.slice(1),16),S=Math.min(255,Math.max(0,(N>>16&255)+y)),w=Math.min(255,Math.max(0,(N>>8&255)+y)),L=Math.min(255,Math.max(0,(N&255)+y));x=`rgb(${Math.round(S)},${Math.round(w)},${Math.round(L)})`}return n==="erase"&&r>=2&&r<=4&&d>=3&&d<=5&&(x="#9ca3af"),{x:t,y:s,w:u,h,fill:x,opacity:g}};return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Image Augmentation Demo"}),e.jsx("div",{className:"flex flex-wrap gap-2 mb-3",children:["none","hflip","vflip","crop","jitter","erase"].map(r=>e.jsx("button",{onClick:()=>f(r),className:`rounded px-3 py-1 text-xs ${n===r?"bg-violet-500 text-white":"bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300"}`,children:r==="none"?"Original":r==="hflip"?"H-Flip":r==="vflip"?"V-Flip":r==="crop"?"Center Crop":r==="jitter"?"Color Jitter":"Random Erase"},r))}),e.jsx("svg",{width:i,height:l,className:"mx-auto block border border-gray-200 rounded dark:border-gray-700",children:c.map((r,d)=>r.map((m,o)=>{const{x:t,y:s,w:u,h,fill:x,opacity:g}=p(d,o,m);return e.jsx("rect",{x:t,y:s,width:u,height:h,fill:x,opacity:g},`${d}${o}`)}))})]})}function te(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Data augmentation creates new training examples by applying label-preserving transformations, effectively expanding the dataset and encoding known invariances into the training process."}),e.jsxs(j,{title:"Data Augmentation as Regularization",children:[e.jsxs("p",{children:["Augmentation regularizes by encouraging invariance. If transformation",e.jsx(a.InlineMath,{math:"T"})," preserves labels, training on ",e.jsx(a.InlineMath,{math:"T(x)"})," encourages ",e.jsx(a.InlineMath,{math:"f(T(x)) = f(x)"}),", equivalent to minimizing:"]}),e.jsx(a.BlockMath,{math:"\\mathcal{L}_{\\text{aug}} = \\mathbb{E}_{T \\sim \\mathcal{T}} \\left[\\mathcal{L}(f(T(x)), y)\\right]"})]}),e.jsx(ee,{}),e.jsx(v,{title:"Standard Image Augmentations",children:e.jsxs("ul",{className:"list-disc ml-4 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Horizontal flip"}),": invariant for most objects, not text or asymmetric objects"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Random crop"}),": encourages translation invariance"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Color jitter"}),": robustness to lighting (brightness, contrast, saturation, hue)"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Random rotation"}),": ",e.jsx(a.InlineMath,{math:"\\pm 15°"})," typically safe"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Random erasing"}),": occlusion robustness, similar to cutout"]})]})}),e.jsx(_,{title:"Image Augmentation with torchvision",code:`import torch
from torchvision import transforms

# Standard training augmentation pipeline
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.33)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Validation: no augmentation, only resize and normalize
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Test-time augmentation (TTA)
def predict_tta(model, image, n_aug=5):
    model.eval()
    preds = []
    for _ in range(n_aug):
        aug_img = train_transform(image).unsqueeze(0)
        with torch.no_grad():
            preds.append(model(aug_img))
    return torch.stack(preds).mean(0)  # average predictions

print("Train augmentations:", len(train_transform.transforms))`}),e.jsx(z,{title:"Augmentation Pitfalls",children:e.jsx("p",{children:'Never apply training augmentations to validation or test data (except for TTA). Be careful with augmentations that can change semantics: vertical flips make "6" look like "9", aggressive color jitter can break color-dependent tasks, and random erasing can remove the entire object in small-object detection.'})}),e.jsx(k,{type:"note",title:"Augmentation Strength Schedule",children:e.jsx("p",{children:"Some modern recipes increase augmentation strength over training (progressive augmentation). Light augmentation early helps the model learn basic features, while strong augmentation later prevents overfitting as the model memorizes easy patterns."})})]})}const ve=Object.freeze(Object.defineProperty({__proto__:null,default:te},Symbol.toStringTag,{value:"Module"}));function ae(){const[n,f]=b.useState(2),[i,l]=b.useState(9),c=["Rotate","ShearX","TranslateY","AutoContrast","Equalize","Posterize","Solarize","Color","Brightness","Sharpness"],p=Array.from({length:n},(r,d)=>{const m=(d*3+i)%c.length;return{name:c[m],mag:Math.min(i+d,10)}});return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"RandAugment Policy Sampler"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["N (ops): ",n,e.jsx("input",{type:"range",min:1,max:4,step:1,value:n,onChange:r=>f(parseInt(r.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["M (magnitude): ",i,e.jsx("input",{type:"range",min:1,max:10,step:1,value:i,onChange:r=>l(parseInt(r.target.value)),className:"w-28 accent-violet-500"})]})]}),e.jsx("div",{className:"flex gap-3 flex-wrap",children:p.map((r,d)=>e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 px-4 py-3 text-center",children:[e.jsx("p",{className:"text-sm font-semibold text-violet-700 dark:text-violet-300",children:r.name}),e.jsx("div",{className:"mt-1 h-2 w-20 rounded bg-gray-200 dark:bg-gray-700",children:e.jsx("div",{className:"h-full rounded bg-violet-500",style:{width:`${r.mag*10}%`}})}),e.jsxs("p",{className:"text-xs text-gray-500 mt-1",children:["mag: ",r.mag]})]},d))}),e.jsxs("p",{className:"text-xs text-gray-500 mt-3",children:["RandAugment randomly selects ",n," operations with uniform magnitude ",i,"/10 for each image."]})]})}function se(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Manual augmentation design requires domain expertise and extensive tuning. AutoAugment and RandAugment automate this process by learning or simplifying augmentation policy selection."}),e.jsxs(j,{title:"AutoAugment",children:[e.jsxs("p",{children:["Uses reinforcement learning to search for optimal augmentation policies. A policy consists of ",e.jsx(a.InlineMath,{math:"N"})," sub-policies, each with two operations specified by:"]}),e.jsx(a.BlockMath,{math:"\\text{Sub-policy} = \\{(op_1, p_1, m_1), (op_2, p_2, m_2)\\}"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(a.InlineMath,{math:"op"})," is the operation, ",e.jsx(a.InlineMath,{math:"p"})," is the probability of applying it, and ",e.jsx(a.InlineMath,{math:"m"})," is the magnitude. The search space has",e.jsx(a.InlineMath,{math:"\\sim 10^{32}"})," possible policies."]})]}),e.jsxs(j,{title:"RandAugment",children:[e.jsx("p",{children:"Drastically simplifies the search space to just two parameters:"}),e.jsx(a.BlockMath,{math:"\\text{RandAugment}(N, M): \\text{apply } N \\text{ random ops, each with magnitude } M"}),e.jsxs("p",{className:"mt-2",children:[e.jsx(a.InlineMath,{math:"N"})," and ",e.jsx(a.InlineMath,{math:"M"})," can be tuned with simple grid search. Despite its simplicity, RandAugment matches or exceeds AutoAugment performance."]})]}),e.jsx(ae,{}),e.jsxs(M,{title:"Why RandAugment Works",id:"randaugment-theory",children:[e.jsxs("p",{children:["The key insight is that optimal augmentation magnitude tends to scale with model and dataset size. A single shared magnitude parameter ",e.jsx(a.InlineMath,{math:"M"})," captures this relationship, eliminating the need for per-operation magnitude tuning:"]}),e.jsx(a.BlockMath,{math:"M^* \\propto \\log(\\text{model size} \\times \\text{dataset size})"})]}),e.jsx(v,{title:"TrivialAugment",children:e.jsxs("p",{children:["TrivialAugment (2021) further simplifies: apply exactly ",e.jsx("strong",{children:"one"})," random operation with a random magnitude per image. No hyperparameters to tune at all. Surprisingly, this matches RandAugment on ImageNet while being even simpler."]})}),e.jsx(_,{title:"RandAugment and AutoAugment in PyTorch",code:`import torch
from torchvision import transforms

# RandAugment: N operations, magnitude M (0-31 scale in torchvision)
train_transform_rand = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),  # N=2, M=9
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# AutoAugment with ImageNet policy
train_transform_auto = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# TrivialAugment
train_transform_trivial = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Tuning RandAugment: simple grid search over N and M
for N in [1, 2, 3]:
    for M in [5, 9, 14]:
        aug = transforms.RandAugment(num_ops=N, magnitude=M)
        print(f"RandAugment(N={N}, M={M})")`}),e.jsx(k,{type:"note",title:"Choosing an Augmentation Strategy",children:e.jsxs("p",{children:["Start with ",e.jsx("strong",{children:"TrivialAugment"})," for zero-config baseline. Use ",e.jsx("strong",{children:"RandAugment"})," if you can afford to tune N and M (typically N=2, M=9 for ImageNet-scale). ",e.jsx("strong",{children:"AutoAugment"})," is mainly historical — the search cost rarely justifies the marginal gain over RandAugment."]})})]})}const ke=Object.freeze(Object.defineProperty({__proto__:null,default:se},Symbol.toStringTag,{value:"Module"}));function ne(){const[n,f]=b.useState("synonym"),i="The quick brown fox jumps over the lazy dog",l={synonym:"The fast brown fox leaps over the idle dog",deletion:"The quick fox jumps over lazy dog",swap:"The quick brown jumps fox over the lazy dog",insertion:"The very quick brown fox jumps swiftly over the lazy dog",backtranslation:"The swift brown fox leaps over the indolent dog"},c=i.split(" "),p=l[n].split(" ");return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Text Augmentation Demo"}),e.jsx("div",{className:"flex flex-wrap gap-2 mb-3",children:Object.keys(l).map(r=>e.jsx("button",{onClick:()=>f(r),className:`rounded px-3 py-1 text-xs ${n===r?"bg-violet-500 text-white":"bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300"}`,children:r==="backtranslation"?"Back-Translation":r.charAt(0).toUpperCase()+r.slice(1)},r))}),e.jsxs("div",{className:"space-y-2 text-sm",children:[e.jsxs("div",{className:"rounded bg-gray-50 dark:bg-gray-800 p-3",children:[e.jsx("span",{className:"text-xs text-gray-500 block mb-1",children:"Original:"}),e.jsx("span",{className:"text-gray-700 dark:text-gray-300",children:i})]}),e.jsxs("div",{className:"rounded bg-violet-50 dark:bg-violet-900/20 p-3",children:[e.jsxs("span",{className:"text-xs text-violet-500 block mb-1",children:["Augmented (",n,"):"]}),e.jsx("span",{className:"text-gray-700 dark:text-gray-300",children:p.map((r,d)=>{const m=!c.includes(r)||c[d]!==r&&n!=="insertion";return e.jsxs("span",{className:m?"font-bold text-violet-600 dark:text-violet-400":"",children:[r," "]},d)})})]})]})]})}function re(){const[n,f]=b.useState("both"),i=320,l=160,c=16,p=32,r=i/p,d=l/c,m={start:5,end:9},o={start:12,end:20};return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"SpecAugment Visualization"}),e.jsx("div",{className:"flex gap-2 mb-3",children:["freq","time","both"].map(t=>e.jsx("button",{onClick:()=>f(t),className:`rounded px-3 py-1 text-xs ${n===t?"bg-violet-500 text-white":"bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300"}`,children:t==="freq"?"Freq Mask":t==="time"?"Time Mask":"Both"},t))}),e.jsx("svg",{width:i,height:l,className:"mx-auto block",children:Array.from({length:c},(t,s)=>Array.from({length:p},(u,h)=>{const x=Math.sin(s*.5+h*.2)*.3+.5+Math.sin(h*.4)*.2,g=n!=="time"&&s>=m.start&&s<=m.end||n!=="freq"&&h>=o.start&&h<=o.end;return e.jsx("rect",{x:h*r,y:s*d,width:r,height:d,fill:g?"#1f2937":`hsl(263,${Math.round(x*70+20)}%,${Math.round(x*40+30)}%)`},`${s}${h}`)}))})]})}function ie(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Text and audio domains require specialized augmentation approaches that respect the discrete nature of language and the spectral structure of audio."}),e.jsx(j,{title:"Text Augmentation Techniques",children:e.jsxs("ul",{className:"list-disc ml-4 space-y-1 mt-2",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Synonym replacement"}),": replace words with synonyms from WordNet"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Random deletion"}),": remove words with probability ",e.jsx(a.InlineMath,{math:"p"})]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Random swap"}),": swap positions of two random words"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Back-translation"}),": translate to another language and back"]})]})}),e.jsx(ne,{}),e.jsxs(j,{title:"SpecAugment",children:[e.jsx("p",{children:"SpecAugment applies augmentation directly to log-mel spectrograms with two operations:"}),e.jsx(a.BlockMath,{math:"\\text{FreqMask}: X[f_0 : f_0 + f, :] = 0, \\quad \\text{TimeMask}: X[:, t_0 : t_0 + t] = 0"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(a.InlineMath,{math:"f"})," and ",e.jsx(a.InlineMath,{math:"t"})," are randomly chosen mask widths. This is simple, effective, and requires no external data."]})]}),e.jsx(re,{}),e.jsx(v,{title:"SpecAugment Results",children:e.jsx("p",{children:"SpecAugment reduced word error rate on LibriSpeech from 3.9% to 2.8% without additional data. Frequency masking provides speaker robustness; time masking handles temporal distortions."})}),e.jsx(_,{title:"Text & Audio Augmentation in PyTorch",code:`import torch
import random

# --- Text Augmentation (EDA: Easy Data Augmentation) ---
def synonym_replace(words, n=1):
    syns = {'quick': 'fast', 'jumps': 'leaps', 'lazy': 'idle'}
    new = words.copy()
    for _ in range(n):
        i = random.randint(0, len(new) - 1)
        if new[i] in syns: new[i] = syns[new[i]]
    return new

def random_deletion(words, p=0.1):
    return [w for w in words if random.random() > p] or [words[0]]

text = "The quick brown fox jumps over the lazy dog"
print("Synonym:", ' '.join(synonym_replace(text.split())))
print("Deletion:", ' '.join(random_deletion(text.split(), p=0.2)))

# --- SpecAugment for Audio ---
def spec_augment(spec, freq_mask=15, time_mask=20):
    cloned = spec.clone()
    F, T = cloned.shape
    f = random.randint(0, min(freq_mask, F))
    f0 = random.randint(0, F - f)
    cloned[f0:f0+f, :] = 0
    t = random.randint(0, min(time_mask, T))
    t0 = random.randint(0, T - t)
    cloned[:, t0:t0+t] = 0
    return cloned

spec = torch.randn(80, 200)  # 80 mel bins, 200 time steps
augmented = spec_augment(spec)
print(f"Zeroed: {(augmented == 0).sum().item()} / {spec.numel()}")`}),e.jsx(z,{title:"Text Augmentation Caveats",children:e.jsx("p",{children:"Text augmentation can change semantics easily. Synonym replacement may alter meaning in context. Back-translation quality depends on the translation model. For large language models, augmentation is less critical since pretraining provides regularization."})}),e.jsx(k,{type:"note",title:"Modality-Specific Considerations",children:e.jsx("p",{children:"Each modality has unique invariances. The best augmentation strategy encodes the invariances specific to your task: flips for images, paraphrase for text, speed changes and noise for audio."})})]})}const we=Object.freeze(Object.defineProperty({__proto__:null,default:ie},Symbol.toStringTag,{value:"Module"}));export{he as a,me as b,pe as c,xe as d,ge as e,ue as f,fe as g,ye as h,je as i,be as j,_e as k,ve as l,ke as m,we as n,ce as s};

import{j as e,r as p}from"./vendor-DpISuAX6.js";import{r as t}from"./vendor-katex-CbWCYdth.js";import{D as h,E as g,P as u,N as f,T as _,W as y}from"./subject-01-foundations-D0A1VJsr.js";function b(){const[o,c]=p.useState(!1),r=[{id:0,x:80,y:50,label:"v0"},{id:1,x:200,y:30,label:"v1"},{id:2,x:280,y:100,label:"v2"},{id:3,x:180,y:140,label:"v3"},{id:4,x:60,y:130,label:"v4"}],d=[[0,1],[1,2],[2,3],[3,4],[4,0],[0,3]],i=Array.from({length:5},()=>Array(5).fill(0));return d.forEach(([a,s])=>{i[a][s]=1,i[s][a]=1}),e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Graph & Adjacency Matrix"}),e.jsx("div",{className:"flex items-center gap-4 mb-3",children:e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:[e.jsx("input",{type:"checkbox",checked:o,onChange:a=>c(a.target.checked),className:"accent-violet-500"}),"Show adjacency matrix"]})}),e.jsxs("div",{className:"flex items-start gap-6 justify-center",children:[e.jsxs("svg",{width:340,height:170,className:"block",children:[d.map(([a,s],n)=>e.jsx("line",{x1:r[a].x,y1:r[a].y,x2:r[s].x,y2:r[s].y,stroke:"#7c3aed",strokeWidth:1.5,opacity:.4},n)),r.map(a=>e.jsxs("g",{children:[e.jsx("circle",{cx:a.x,cy:a.y,r:18,fill:"#7c3aed",opacity:.15,stroke:"#7c3aed",strokeWidth:2}),e.jsx("text",{x:a.x,y:a.y+4,textAnchor:"middle",fill:"#7c3aed",fontSize:12,fontWeight:"bold",children:a.label})]},a.id))]}),o&&e.jsxs("table",{className:"text-xs border-collapse font-mono",children:[e.jsx("thead",{children:e.jsxs("tr",{children:[e.jsx("th",{className:"px-2 py-1"}),r.map(a=>e.jsx("th",{className:"px-2 py-1 text-violet-600",children:a.label},a.id))]})}),e.jsx("tbody",{children:i.map((a,s)=>e.jsxs("tr",{children:[e.jsx("td",{className:"px-2 py-1 text-violet-600 font-bold",children:r[s].label}),a.map((n,l)=>e.jsx("td",{className:`px-2 py-1 text-center ${n?"bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-400 font-bold":"text-gray-400"}`,children:n},l))]},s))})]})]})]})}function v(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Graphs are the natural data structure for relational data. Before applying neural networks to graphs, we need efficient representations of graph topology and node/edge features."}),e.jsxs(h,{title:"Graph Definition",children:[e.jsxs("p",{children:["A graph ",e.jsx(t.InlineMath,{math:"\\mathcal{G} = (\\mathcal{V}, \\mathcal{E})"})," consists of:"]}),e.jsx(t.BlockMath,{math:"\\mathcal{V} = \\{v_1, \\ldots, v_N\\} \\text{ (nodes)}, \\quad \\mathcal{E} \\subseteq \\mathcal{V} \\times \\mathcal{V} \\text{ (edges)}"}),e.jsxs("p",{className:"mt-2",children:["Each node ",e.jsx(t.InlineMath,{math:"v_i"})," has a feature vector ",e.jsx(t.InlineMath,{math:"\\mathbf{x}_i \\in \\mathbb{R}^d"}),". The full feature matrix is ",e.jsx(t.InlineMath,{math:"\\mathbf{X} \\in \\mathbb{R}^{N \\times d}"}),"."]})]}),e.jsxs(h,{title:"Adjacency Matrix",children:[e.jsx(t.BlockMath,{math:"A_{ij} = \\begin{cases} 1 & \\text{if } (v_i, v_j) \\in \\mathcal{E} \\\\ 0 & \\text{otherwise} \\end{cases}"}),e.jsxs("p",{className:"mt-2",children:["The degree matrix: ",e.jsx(t.InlineMath,{math:"D_{ii} = \\sum_j A_{ij}"}),". For undirected graphs, ",e.jsx(t.InlineMath,{math:"A = A^\\top"}),"."]})]}),e.jsx(b,{}),e.jsxs(g,{title:"Common Graph Representations",children:[e.jsxs("p",{children:[e.jsx("strong",{children:"Adjacency matrix"}),": Dense ",e.jsx(t.InlineMath,{math:"O(N^2)"})," storage. Good for small, dense graphs."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Edge list"}),": Store pairs ",e.jsx(t.InlineMath,{math:"(i, j)"}),". ",e.jsx(t.InlineMath,{math:"O(|\\mathcal{E}|)"})," space."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"COO format"}),": Two arrays of source and destination indices. Used by PyG and DGL."]}),e.jsxs("p",{children:["Most real-world graphs are sparse (",e.jsx(t.InlineMath,{math:"|\\mathcal{E}| \\ll N^2"}),"), making sparse formats preferred."]})]}),e.jsx(u,{title:"Graph Representations with PyTorch Geometric",code:`import torch
from torch_geometric.data import Data

# Define a small graph: 5 nodes, 6 edges
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 3],  # source
    [1, 0, 2, 1, 3, 2, 4, 3, 0, 4, 3, 0],  # target
], dtype=torch.long)

# Node features: 5 nodes, 3 features each
x = torch.randn(5, 3)

# Create a PyG Data object
data = Data(x=x, edge_index=edge_index)
print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
print(f"Node features shape: {data.x.shape}")
print(f"Is undirected: {data.is_undirected()}")
print(f"Average degree: {data.num_edges / data.num_nodes:.1f}")

# Convert to adjacency matrix (dense) for inspection
adj = torch.zeros(5, 5)
adj[edge_index[0], edge_index[1]] = 1
print(f"Adjacency matrix:\\n{adj}")`}),e.jsx(f,{type:"note",title:"Heterogeneous Graphs",children:e.jsx("p",{children:'Many real-world graphs have multiple node and edge types (e.g., users and products connected by "purchased" and "reviewed" edges). These require typed adjacency matrices or separate edge stores per relation type.'})})]})}const ee=Object.freeze(Object.defineProperty({__proto__:null,default:v},Symbol.toStringTag,{value:"Module"}));function N(){const[o,c]=p.useState(!1),r=[[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]],d=r.map(n=>n.reduce((l,m)=>l+m,0)),i=r.map((n,l)=>n.map((m,x)=>(l===x?d[l]:0)-m)),a=r.map((n,l)=>n.map((m,x)=>l===x&&d[l]>0?1:m===1?(-1/Math.sqrt(d[l]*d[x])).toFixed(2):0)),s=o?a:i;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Graph Laplacian"}),e.jsx("div",{className:"flex items-center gap-4 mb-3",children:e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:[e.jsx("input",{type:"checkbox",checked:o,onChange:n=>c(n.target.checked),className:"accent-violet-500"}),"Normalized Laplacian"]})}),e.jsxs("div",{className:"flex items-center gap-8 justify-center",children:[e.jsxs("div",{children:[e.jsx("p",{className:"text-xs text-gray-500 mb-1 text-center",children:"A (adjacency)"}),e.jsx("table",{className:"text-xs font-mono border-collapse",children:e.jsx("tbody",{children:r.map((n,l)=>e.jsx("tr",{children:n.map((m,x)=>e.jsx("td",{className:`px-2 py-1 text-center ${m?"text-violet-600 font-bold":"text-gray-400"}`,children:m},x))},l))})})]}),e.jsx("span",{className:"text-gray-400 text-lg",children:"→"}),e.jsxs("div",{children:[e.jsx("p",{className:"text-xs text-gray-500 mb-1 text-center",children:o?"L_norm":"L = D - A"}),e.jsx("table",{className:"text-xs font-mono border-collapse",children:e.jsx("tbody",{children:s.map((n,l)=>e.jsx("tr",{children:n.map((m,x)=>e.jsx("td",{className:`px-2 py-1 text-center ${l===x?"text-violet-700 dark:text-violet-400 font-bold bg-violet-50 dark:bg-violet-900/20":parseFloat(m)<0?"text-orange-600":"text-gray-400"}`,children:m},x))},l))})})]})]})]})}function k(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Spectral graph theory studies graphs through the eigenvalues and eigenvectors of matrices associated with them, particularly the graph Laplacian. This forms the mathematical foundation for spectral graph convolutions."}),e.jsxs(h,{title:"Graph Laplacian",children:[e.jsx("p",{children:"The combinatorial Laplacian:"}),e.jsx(t.BlockMath,{math:"L = D - A"}),e.jsx("p",{className:"mt-2",children:"The symmetric normalized Laplacian:"}),e.jsx(t.BlockMath,{math:"L_\\text{sym} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}"}),e.jsxs("p",{className:"mt-2",children:[e.jsx(t.InlineMath,{math:"L"})," is positive semi-definite with eigenvalues ",e.jsx(t.InlineMath,{math:"0 = \\lambda_1 \\le \\lambda_2 \\le \\cdots \\le \\lambda_N"}),"."]})]}),e.jsx(N,{}),e.jsxs(_,{title:"Spectral Decomposition",id:"spectral-decomp",children:[e.jsx("p",{children:"The Laplacian admits an eigendecomposition:"}),e.jsx(t.BlockMath,{math:"L = U \\Lambda U^\\top, \\quad \\Lambda = \\text{diag}(\\lambda_1, \\ldots, \\lambda_N)"}),e.jsxs("p",{className:"mt-2",children:["The eigenvectors ",e.jsx(t.InlineMath,{math:"U"})," form an orthonormal basis for signals on the graph. The ",e.jsx("strong",{children:"Graph Fourier Transform"})," of a signal ",e.jsx(t.InlineMath,{math:"\\mathbf{x}"})," is:"]}),e.jsx(t.BlockMath,{math:"\\hat{\\mathbf{x}} = U^\\top \\mathbf{x}"})]}),e.jsxs(g,{title:"Eigenvalue Interpretation",children:[e.jsxs("p",{children:[e.jsx(t.InlineMath,{math:"\\lambda_1 = 0"}),": constant eigenvector. Number of zero eigenvalues = number of connected components."]}),e.jsxs("p",{children:[e.jsx(t.InlineMath,{math:"\\lambda_2"})," (Fiedler value): measures graph connectivity. Used for spectral clustering."]}),e.jsx("p",{children:"Higher eigenvalues correspond to higher-frequency variations across the graph."})]}),e.jsx(u,{title:"Spectral Analysis with NumPy",code:`import numpy as np

# Define adjacency matrix
A = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 1, 1],
    [0, 1, 1, 0, 1],
    [0, 0, 1, 1, 0]
])

D = np.diag(A.sum(axis=1))
L = D - A  # Combinatorial Laplacian

# Normalized Laplacian
D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1)))
L_norm = np.eye(5) - D_inv_sqrt @ A @ D_inv_sqrt

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
print("Eigenvalues:", eigenvalues.round(4))
print("Fiedler value (algebraic connectivity):", eigenvalues[1].round(4))
print("Fiedler vector:", eigenvectors[:, 1].round(3))

# Graph Fourier Transform of a signal
x = np.array([1.0, 0.5, -0.5, -1.0, 0.0])
x_hat = eigenvectors.T @ x  # spectral coefficients
print("Spectral coefficients:", x_hat.round(3))`}),e.jsx(f,{type:"note",title:"From Spectral to Spatial",children:e.jsxs("p",{children:["Spectral convolution (",e.jsx(t.InlineMath,{math:"g_\\theta \\star x = U g_\\theta(\\Lambda) U^\\top x"}),") requires computing the full eigendecomposition (",e.jsx(t.InlineMath,{math:"O(N^3)"}),"). ChebNet approximates the filter with Chebyshev polynomials, and GCN simplifies further to a first-order approximation, leading to the efficient spatial message-passing framework."]})})]})}const te=Object.freeze(Object.defineProperty({__proto__:null,default:k},Symbol.toStringTag,{value:"Module"}));function w(){const[o,c]=p.useState("node"),r=[{x:60,y:50},{x:160,y:30},{x:260,y:60},{x:100,y:130},{x:210,y:140}],d=[[0,1],[1,2],[0,3],[3,4],[1,4],[2,4]],i=o==="node"?["#7c3aed","#f97316","#7c3aed","#f97316","#7c3aed"]:Array(5).fill("#7c3aed"),a=o==="edge"?[!1,!1,!1,!0,!1,!1]:Array(6).fill(!1);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Graph Learning Tasks"}),e.jsx("div",{className:"flex items-center gap-2 mb-3",children:["node","edge","graph"].map(s=>e.jsxs("button",{onClick:()=>c(s),className:`px-3 py-1 rounded text-sm ${s===o?"bg-violet-500 text-white":"bg-gray-100 dark:bg-gray-800 text-gray-600"}`,children:[s,"-level"]},s))}),e.jsxs("svg",{width:320,height:170,className:"mx-auto block",children:[o==="graph"&&e.jsx("rect",{x:20,y:5,width:280,height:160,rx:12,fill:"none",stroke:"#7c3aed",strokeWidth:2,strokeDasharray:"5,3"}),d.map(([s,n],l)=>e.jsx("line",{x1:r[s].x,y1:r[s].y,x2:r[n].x,y2:r[n].y,stroke:a[l]?"#f97316":"#d1d5db",strokeWidth:a[l]?3:1.5},l)),r.map((s,n)=>e.jsx("circle",{cx:s.x,cy:s.y,r:16,fill:i[n],opacity:.8},n)),o==="graph"&&e.jsx("text",{x:160,y:168,textAnchor:"middle",fill:"#7c3aed",fontSize:10,children:"graph classification"}),o==="edge"&&e.jsx("text",{x:155,y:155,textAnchor:"middle",fill:"#f97316",fontSize:10,children:"link prediction"}),o==="node"&&e.jsx("text",{x:160,y:168,textAnchor:"middle",fill:"#7c3aed",fontSize:10,children:"node classification (2 classes)"})]})]})}function G(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Graph learning tasks can be categorized by the level at which predictions are made: individual nodes, pairs of nodes (edges), or entire graphs."}),e.jsxs(h,{title:"Node-Level Tasks",children:[e.jsx("p",{children:"Predict a label or property for each node:"}),e.jsx(t.BlockMath,{math:"\\hat{y}_i = f(\\mathbf{h}_i), \\quad \\mathbf{h}_i = \\text{GNN}(\\mathcal{G}, \\mathbf{X})_i"}),e.jsx("p",{className:"mt-2",children:"Examples: citation network classification, fraud detection, protein function prediction."})]}),e.jsxs(h,{title:"Edge-Level Tasks",children:[e.jsx("p",{children:"Predict the existence or properties of edges (link prediction):"}),e.jsx(t.BlockMath,{math:"\\hat{y}_{ij} = g(\\mathbf{h}_i, \\mathbf{h}_j), \\quad \\text{e.g., } g = \\sigma(\\mathbf{h}_i^\\top \\mathbf{h}_j)"}),e.jsx("p",{className:"mt-2",children:"Examples: social network friend recommendation, knowledge graph completion."})]}),e.jsxs(h,{title:"Graph-Level Tasks",children:[e.jsx("p",{children:"Predict a property of the entire graph using a readout function:"}),e.jsx(t.BlockMath,{math:"\\hat{y} = \\text{READOUT}(\\{\\mathbf{h}_i : v_i \\in \\mathcal{V}\\})"}),e.jsx("p",{className:"mt-2",children:"Common readouts: mean/sum/max pooling, or learned hierarchical pooling. Examples: molecular property prediction, program analysis."})]}),e.jsx(w,{}),e.jsxs(g,{title:"Benchmark Datasets",children:[e.jsxs("p",{children:[e.jsx("strong",{children:"Node"}),": Cora (2.7K papers, 7 classes), ogbn-arxiv (170K papers)."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Edge"}),": ogbl-collab (author collaboration), ogbl-citation2."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Graph"}),": ogbg-molhiv (41K molecules), ZINC (12K molecules)."]})]}),e.jsx(u,{title:"Three Task Types in PyG",code:`import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class MultiTaskGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, node_classes, graph_classes):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.node_head = nn.Linear(hidden_dim, node_classes)
        self.graph_head = nn.Linear(hidden_dim, graph_classes)

    def encode(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index)
        return h

    def node_classification(self, x, edge_index):
        h = self.encode(x, edge_index)
        return self.node_head(h)  # (N, node_classes)

    def link_prediction(self, x, edge_index, src, dst):
        h = self.encode(x, edge_index)
        return (h[src] * h[dst]).sum(dim=-1)  # dot product score

    def graph_classification(self, x, edge_index, batch):
        h = self.encode(x, edge_index)
        h_graph = global_mean_pool(h, batch)  # (B, hidden_dim)
        return self.graph_head(h_graph)

model = MultiTaskGNN(in_dim=16, hidden_dim=64, node_classes=7, graph_classes=2)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")`}),e.jsx(f,{type:"note",title:"Inductive vs Transductive",children:e.jsxs("p",{children:[e.jsx("strong",{children:"Transductive"}),": test nodes are visible during training (without labels). Common in node classification on a single graph.",e.jsx("strong",{children:"Inductive"}),": the model must generalize to entirely unseen graphs. Graph classification is inherently inductive; GraphSAGE enables inductive node classification."]})})]})}const ae=Object.freeze(Object.defineProperty({__proto__:null,default:G},Symbol.toStringTag,{value:"Module"}));function M(){const[o,c]=p.useState(0),r=[{x:160,y:40},{x:60,y:110},{x:260,y:110},{x:100,y:190},{x:220,y:190}],d=[[0,1],[0,2],[1,3],[1,4],[2,4]],i=[[[0]],[[0,1,2]],[[0,1,2,3,4]]],a=new Set(i[Math.min(o,2)][0]||[]),s=["#7c3aed","#f97316","#10b981"];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Message Passing Layers (node v0)"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Layers: ",o,e.jsx("input",{type:"range",min:0,max:2,step:1,value:o,onChange:n=>c(parseInt(n.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("span",{className:"text-xs text-violet-600 dark:text-violet-400",children:["Receptive field: ",a.size," nodes"]})]}),e.jsxs("svg",{width:320,height:220,className:"mx-auto block",children:[d.map(([n,l],m)=>e.jsx("line",{x1:r[n].x,y1:r[n].y,x2:r[l].x,y2:r[l].y,stroke:a.has(n)&&a.has(l)?"#7c3aed":"#e5e7eb",strokeWidth:a.has(n)&&a.has(l)?2:1},m)),r.map((n,l)=>e.jsxs("g",{children:[e.jsx("circle",{cx:n.x,cy:n.y,r:18,fill:a.has(l)?s[Math.min(o,2)]:"#e5e7eb",opacity:a.has(l)?.8:.4}),e.jsxs("text",{x:n.x,y:n.y+4,textAnchor:"middle",fill:a.has(l)?"white":"#9ca3af",fontSize:11,fontWeight:"bold",children:["v",l]})]},l))]}),e.jsx("p",{className:"text-center text-xs text-gray-500 mt-1",children:"Each layer expands the receptive field by one hop"})]})}function S(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"The Message Passing Neural Network (MPNN) framework unifies most GNN architectures under a common aggregate-update paradigm. Each layer aggregates information from neighbors and updates node representations."}),e.jsxs(h,{title:"Message Passing Framework",children:[e.jsxs("p",{children:["Each layer ",e.jsx(t.InlineMath,{math:"k"})," performs two operations:"]}),e.jsx(t.BlockMath,{math:"\\mathbf{m}_i^{(k)} = \\bigoplus_{j \\in \\mathcal{N}(i)} \\phi\\!\\left(\\mathbf{h}_i^{(k-1)}, \\mathbf{h}_j^{(k-1)}, \\mathbf{e}_{ij}\\right)"}),e.jsx(t.BlockMath,{math:"\\mathbf{h}_i^{(k)} = \\psi\\!\\left(\\mathbf{h}_i^{(k-1)}, \\mathbf{m}_i^{(k)}\\right)"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"\\phi"})," is the message function, ",e.jsx(t.InlineMath,{math:"\\bigoplus"})," is a permutation-invariant aggregation (sum, mean, max), and ",e.jsx(t.InlineMath,{math:"\\psi"})," is the update function."]})]}),e.jsx(M,{}),e.jsxs(_,{title:"Expressiveness and WL Test",id:"wl-test",children:[e.jsx("p",{children:"The Weisfeiler-Lehman (WL) graph isomorphism test provides an upper bound on GNN expressiveness:"}),e.jsx(t.BlockMath,{math:"\\text{MPNN distinguishes } \\mathcal{G}_1, \\mathcal{G}_2 \\implies \\text{1-WL distinguishes } \\mathcal{G}_1, \\mathcal{G}_2"}),e.jsx("p",{className:"mt-2",children:"GIN (Graph Isomorphism Network) achieves this upper bound with sum aggregation and injective update functions."})]}),e.jsxs(g,{title:"Common Instantiations",children:[e.jsxs("p",{children:[e.jsx("strong",{children:"GCN"}),": ",e.jsx(t.InlineMath,{math:"\\phi(h_j) = h_j / \\sqrt{d_i d_j}"}),", mean-like aggregation."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"GraphSAGE"}),": ",e.jsx(t.InlineMath,{math:"\\phi(h_j) = h_j"}),", sample and aggregate (mean/LSTM/pool)."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"GAT"}),": ",e.jsx(t.InlineMath,{math:"\\phi(h_i, h_j) = \\alpha_{ij} W h_j"}),", learned attention weights."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"GIN"}),": ",e.jsx(t.InlineMath,{math:"\\psi(h_i, m_i) = \\text{MLP}((1+\\varepsilon)h_i + m_i)"}),", sum aggregation."]})]}),e.jsx(u,{title:"Custom Message Passing Layer in PyG",code:`import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class SimpleMP(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # sum aggregation
        self.lin = nn.Linear(in_channels, out_channels)
        self.update_mlp = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.ReLU(),
        )

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x_transformed = self.lin(x)
        # Triggers message() -> aggregate() -> update()
        return self.propagate(edge_index, x=x_transformed)

    def message(self, x_j):
        return x_j  # messages are neighbor features

    def update(self, aggr_out, x):
        # Combine aggregated messages with self features
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))

layer = SimpleMP(16, 32)
x = torch.randn(5, 16)
edge_index = torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]])
out = layer(x, edge_index)
print(f"Input: {x.shape} -> Output: {out.shape}")`}),e.jsx(f,{type:"note",title:"Over-Smoothing",children:e.jsxs("p",{children:["As the number of message passing layers increases, node representations converge to the same vector (over-smoothing). After ",e.jsx(t.InlineMath,{math:"K"})," layers, each node's representation is influenced by its ",e.jsx(t.InlineMath,{math:"K"}),"-hop neighborhood. Most GNNs use 2-4 layers. Techniques like residual connections, jumping knowledge, and DropEdge help mitigate this."]})})]})}const se=Object.freeze(Object.defineProperty({__proto__:null,default:S},Symbol.toStringTag,{value:"Module"}));function A(){const[o,c]=p.useState("sym"),r=[2,3,1,4,2],d=["v0","v1","v2","v3","v4"],i=(a,s)=>o==="none"?1:o==="row"?(1/a).toFixed(2):(1/Math.sqrt(a*s)).toFixed(3);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Aggregation Normalization"}),e.jsx("div",{className:"flex items-center gap-2 mb-3",children:[["none","A"],["row","D^{-1}A"],["sym","D^{-1/2}AD^{-1/2}"]].map(([a,s])=>e.jsx("button",{onClick:()=>c(a),className:`px-3 py-1 rounded text-xs ${a===o?"bg-violet-500 text-white":"bg-gray-100 dark:bg-gray-800 text-gray-600"}`,children:s},a))}),e.jsx("div",{className:"flex gap-3 justify-center flex-wrap",children:d.map((a,s)=>e.jsxs("div",{className:"text-center px-3 py-2 rounded-lg bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700",children:[e.jsxs("div",{className:"text-xs text-gray-500",children:[a," (deg=",r[s],")"]}),e.jsxs("div",{className:"text-sm font-mono text-violet-600 dark:text-violet-400",children:["w = ",i(r[s],r[Math.min(s+1,4)])]})]},s))}),e.jsx("p",{className:"text-center text-xs text-gray-500 mt-2",children:"Symmetric normalization prevents high-degree nodes from dominating"})]})}function T(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Graph Convolutional Networks (GCN) by Kipf & Welling (2017) simplified spectral graph convolutions to a first-order approximation, creating the most influential GNN architecture."}),e.jsxs(h,{title:"GCN Layer",children:[e.jsx("p",{children:"The GCN propagation rule:"}),e.jsx(t.BlockMath,{math:"H^{(l+1)} = \\sigma\\!\\left(\\tilde{D}^{-1/2} \\tilde{A} \\tilde{D}^{-1/2} H^{(l)} W^{(l)}\\right)"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"\\tilde{A} = A + I"})," (self-loops added),",e.jsx(t.InlineMath,{math:"\\tilde{D}_{ii} = \\sum_j \\tilde{A}_{ij}"}),", and ",e.jsx(t.InlineMath,{math:"W^{(l)}"})," is the trainable weight matrix."]})]}),e.jsxs(_,{title:"Spectral Motivation",id:"gcn-spectral",children:[e.jsx("p",{children:"GCN derives from a first-order Chebyshev approximation of spectral filters:"}),e.jsx(t.BlockMath,{math:"g_\\theta \\star x \\approx \\theta_0 x + \\theta_1 (L - I) x = \\theta_0 x - \\theta_1 D^{-1/2} A D^{-1/2} x"}),e.jsxs("p",{className:"mt-2",children:["Setting ",e.jsx(t.InlineMath,{math:"\\theta_0 = -\\theta_1 = \\theta"})," and adding self-loops yields the GCN formula. This connects spectral theory to a simple, efficient spatial computation."]})]}),e.jsx(A,{}),e.jsxs(g,{title:"GCN on Cora",children:[e.jsx("p",{children:"The Cora citation network: 2708 papers, 5429 edges, 7 classes, 1433-dim bag-of-words features."}),e.jsx("p",{children:"A 2-layer GCN (16 hidden units) achieves ~81% test accuracy with just 140 labeled nodes (20 per class). This demonstrated the power of semi-supervised learning on graphs."})]}),e.jsx(u,{title:"GCN Implementation from Scratch",code:`import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# Load Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(dataset.num_features, 16, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training loop
for epoch in range(200):
    model.train()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data.x, data.edge_index).argmax(dim=1)
acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
print(f"Test accuracy: {acc:.4f}")`}),e.jsx(f,{type:"note",title:"GCN Limitations",children:e.jsx("p",{children:"GCN uses fixed, symmetric normalization weights, treating all neighbors equally (up to degree). It cannot distinguish structurally different neighborhoods with the same degree. GAT addresses this by learning edge-specific attention weights."})})]})}const ne=Object.freeze(Object.defineProperty({__proto__:null,default:T},Symbol.toStringTag,{value:"Module"}));function P(){const[o,c]=p.useState(2),r=[0,1,2,3,4,5,6,7],[d,i]=p.useState(0),a=new Set;let s=d;for(;a.size<Math.min(o,r.length);)s=s*1103515245+12345&2147483647,a.add(r[s%r.length]);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Neighbor Sampling"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Sample size: ",o,e.jsx("input",{type:"range",min:1,max:8,step:1,value:o,onChange:n=>c(parseInt(n.target.value)),className:"w-24 accent-violet-500"})]}),e.jsx("button",{onClick:()=>i(d+1),className:"px-3 py-1 rounded bg-violet-500 text-white text-sm hover:bg-violet-600",children:"Resample"})]}),e.jsxs("div",{className:"flex items-center gap-6 justify-center",children:[e.jsx("div",{className:"w-14 h-14 rounded-full bg-violet-500 flex items-center justify-center text-white font-bold text-sm",children:"target"}),e.jsx("div",{className:"flex flex-wrap gap-2 max-w-[200px]",children:r.map(n=>e.jsxs("div",{className:`w-10 h-10 rounded-full flex items-center justify-center text-xs font-bold ${a.has(n)?"bg-violet-200 dark:bg-violet-800 text-violet-700 dark:text-violet-300 border-2 border-violet-500":"bg-gray-100 dark:bg-gray-800 text-gray-400 border border-gray-300 dark:border-gray-600"}`,children:["n",n]},n))})]}),e.jsxs("p",{className:"text-center text-xs text-gray-500 mt-2",children:["Sample ",o," of 8 neighbors per layer (reduces computation)"]})]})}function C(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"GraphSAGE (SAmple and aggreGatE) enables inductive learning on graphs by sampling and aggregating features from a fixed-size neighborhood, making it scalable to large, evolving graphs."}),e.jsxs(h,{title:"GraphSAGE Algorithm",children:[e.jsxs("p",{children:["For each layer ",e.jsx(t.InlineMath,{math:"k"})," and node ",e.jsx(t.InlineMath,{math:"v"}),":"]}),e.jsx(t.BlockMath,{math:"\\mathbf{h}_{\\mathcal{N}(v)}^{(k)} = \\text{AGGREGATE}_k\\!\\left(\\left\\{\\mathbf{h}_u^{(k-1)} : u \\in \\mathcal{S}(v)\\right\\}\\right)"}),e.jsx(t.BlockMath,{math:"\\mathbf{h}_v^{(k)} = \\sigma\\!\\left(W^{(k)} \\cdot \\text{CONCAT}\\!\\left(\\mathbf{h}_v^{(k-1)}, \\mathbf{h}_{\\mathcal{N}(v)}^{(k)}\\right)\\right)"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"\\mathcal{S}(v)"})," is a fixed-size sample of neighbors."]})]}),e.jsx(P,{}),e.jsxs(g,{title:"Aggregator Choices",children:[e.jsxs("p",{children:[e.jsx("strong",{children:"Mean"}),": ",e.jsx(t.InlineMath,{math:"\\text{AGG} = \\text{mean}(\\{h_u\\})"}),". Simple, equivalent to GCN."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Pool"}),": ",e.jsx(t.InlineMath,{math:"\\text{AGG} = \\max(\\{\\sigma(W_\\text{pool} h_u + b)\\})"}),". Non-linear transform before pooling."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"LSTM"}),": Apply LSTM on a random permutation of neighbors. More expressive but sensitive to ordering."]}),e.jsx("p",{children:"Mean aggregator is most commonly used due to simplicity and competitive performance."})]}),e.jsxs(h,{title:"Mini-Batch Training",children:[e.jsx("p",{children:"GraphSAGE enables mini-batch training on large graphs through neighbor sampling:"}),e.jsx(t.BlockMath,{math:"\\text{Layer } K: \\text{sample } S_K \\text{ neighbors} \\to \\text{Layer } K-1: \\text{sample } S_{K-1} \\text{ per node} \\to \\cdots"}),e.jsxs("p",{className:"mt-2",children:["Total computation per target node: ",e.jsx(t.InlineMath,{math:"O(\\prod_{k=1}^K S_k)"}),". With ",e.jsx(t.InlineMath,{math:"K=2, S_1=25, S_2=10"}),", each node aggregates from up to 250 second-hop neighbors."]})]}),e.jsx(u,{title:"GraphSAGE with Neighbor Sampling in PyG",code:`import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader

class GraphSAGEModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

# Mini-batch training with neighbor sampling
# loader = NeighborLoader(
#     data,
#     num_neighbors=[25, 10],  # sample sizes per layer
#     batch_size=256,
#     input_nodes=data.train_mask,
# )
# for batch in loader:
#     out = model(batch.x, batch.edge_index)
#     loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])

# Key advantage: scales to millions of nodes!
model = GraphSAGEModel(16, 64, 7)
x = torch.randn(100, 16)
edge_index = torch.randint(0, 100, (2, 500))
out = model(x, edge_index)
print(f"Output shape: {out.shape}")`}),e.jsx(y,{title:"Sampling Variance",children:e.jsx("p",{children:"Neighbor sampling introduces variance into gradient estimates. Larger sample sizes reduce variance but increase computation. Layer-wise sampling (as in GraphSAGE) can lead to exponential neighborhood expansion. Alternatives like ClusterGCN and GraphSAINT sample subgraphs instead to reduce this issue."})}),e.jsx(f,{type:"note",title:"Inductive Capability",children:e.jsx("p",{children:"Unlike GCN (which uses a fixed adjacency matrix), GraphSAGE learns aggregation functions that generalize to unseen nodes and graphs. This makes it ideal for production systems where the graph evolves over time (e.g., new users joining a social network)."})})]})}const re=Object.freeze(Object.defineProperty({__proto__:null,default:C},Symbol.toStringTag,{value:"Module"}));function L(){const[o,c]=p.useState(0),r={0:[{id:1,w:.35},{id:2,w:.45},{id:3,w:.2}],1:[{id:0,w:.6},{id:2,w:.25},{id:4,w:.15}],2:[{id:0,w:.3},{id:1,w:.3},{id:3,w:.4}]},d={0:[80,80],1:[200,40],2:[200,120],3:[320,80],4:[320,30]},i=r[o]||[];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"GAT Attention Weights"}),e.jsx("div",{className:"flex items-center gap-2 mb-3",children:[0,1,2].map(a=>e.jsxs("button",{onClick:()=>c(a),className:`px-3 py-1 rounded text-sm ${a===o?"bg-violet-500 text-white":"bg-gray-100 dark:bg-gray-800 text-gray-600"}`,children:["v",a]},a))}),e.jsxs("svg",{width:380,height:150,className:"mx-auto block",children:[i.map((a,s)=>{const[n,l]=d[o],[m,x]=d[a.id];return e.jsx("line",{x1:n,y1:l,x2:m,y2:x,stroke:"#7c3aed",strokeWidth:a.w*8,opacity:.6},s)}),Object.entries(d).map(([a,[s,n]])=>e.jsxs("g",{children:[e.jsx("circle",{cx:s,cy:n,r:18,fill:parseInt(a)===o?"#7c3aed":"#e5e7eb",stroke:"#7c3aed",strokeWidth:1.5}),e.jsxs("text",{x:s,y:n+4,textAnchor:"middle",fill:parseInt(a)===o?"white":"#374151",fontSize:11,fontWeight:"bold",children:["v",a]})]},a)),i.map((a,s)=>{const[n,l]=d[o],[m,x]=d[a.id];return e.jsx("text",{x:(n+m)/2,y:(l+x)/2-6,textAnchor:"middle",fill:"#7c3aed",fontSize:10,fontWeight:"bold",children:a.w.toFixed(2)},s)})]}),e.jsx("p",{className:"text-center text-xs text-gray-500 mt-1",children:"Edge thickness proportional to learned attention weight"})]})}function I(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Graph Attention Networks (GAT) apply the attention mechanism to graphs, learning to weight different neighbors differently. This allows the model to focus on the most relevant neighbors for each node."}),e.jsxs(h,{title:"GAT Attention Mechanism",children:[e.jsxs("p",{children:["Attention coefficients between node ",e.jsx(t.InlineMath,{math:"i"})," and neighbor ",e.jsx(t.InlineMath,{math:"j"}),":"]}),e.jsx(t.BlockMath,{math:"e_{ij} = \\text{LeakyReLU}\\!\\left(\\mathbf{a}^\\top [\\mathbf{W}\\mathbf{h}_i \\| \\mathbf{W}\\mathbf{h}_j]\\right)"}),e.jsx(t.BlockMath,{math:"\\alpha_{ij} = \\text{softmax}_j(e_{ij}) = \\frac{\\exp(e_{ij})}{\\sum_{k \\in \\mathcal{N}(i)} \\exp(e_{ik})}"}),e.jsx("p",{className:"mt-2",children:"The output:"}),e.jsx(t.BlockMath,{math:"\\mathbf{h}_i' = \\sigma\\!\\left(\\sum_{j \\in \\mathcal{N}(i)} \\alpha_{ij} \\mathbf{W} \\mathbf{h}_j\\right)"})]}),e.jsxs(h,{title:"Multi-Head Attention",children:[e.jsxs("p",{children:["Use ",e.jsx(t.InlineMath,{math:"K"})," independent attention heads and concatenate (or average in the final layer):"]}),e.jsx(t.BlockMath,{math:"\\mathbf{h}_i' = \\Big\\|_{k=1}^K \\sigma\\!\\left(\\sum_{j \\in \\mathcal{N}(i)} \\alpha_{ij}^{(k)} \\mathbf{W}^{(k)} \\mathbf{h}_j\\right)"})]}),e.jsx(L,{}),e.jsxs(g,{title:"GAT on Cora",children:[e.jsx("p",{children:"Original GAT: 2 layers, 8 attention heads in layer 1 (hidden dim 8 each = 64 total), 1 head in layer 2. Achieves ~83% accuracy on Cora, improving over GCN's ~81%."}),e.jsx("p",{children:"The attention weights are interpretable: we can see which neighbors each node considers most important."})]}),e.jsx(u,{title:"GAT in PyTorch Geometric",code:`import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=0.6)
        # Output layer: 1 head with averaging
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GAT(in_dim=1433, hidden_dim=8, out_dim=7, heads=8)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
# Much fewer params than Transformer: attention is only over neighbors!`}),e.jsx(f,{type:"note",title:"GAT vs Transformer Attention",children:e.jsxs("p",{children:["GAT attention is computed only over graph neighbors (sparse), while Transformer attention is over all tokens (dense). GAT uses a single-layer attention function",e.jsx(t.InlineMath,{math:"\\mathbf{a}^\\top [\\cdot \\| \\cdot]"}),", which is actually limited in expressiveness. GATv2 addresses this with a more powerful dynamic attention mechanism."]})})]})}const ie=Object.freeze(Object.defineProperty({__proto__:null,default:I},Symbol.toStringTag,{value:"Module"}));function E(){const[o,c]=p.useState("v1"),r=[{x:160,y:30,features:"f1"},{x:60,y:110,features:"f2"},{x:260,y:110,features:"f3"}],d=o==="v1"?[.5,.3,.2]:[.2,.6,.2];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Static vs Dynamic Attention"}),e.jsxs("div",{className:"flex items-center gap-2 mb-3",children:[e.jsx("button",{onClick:()=>c("v1"),className:`px-3 py-1 rounded text-sm ${o==="v1"?"bg-violet-500 text-white":"bg-gray-100 dark:bg-gray-800 text-gray-600"}`,children:"GATv1 (static)"}),e.jsx("button",{onClick:()=>c("v2"),className:`px-3 py-1 rounded text-sm ${o==="v2"?"bg-violet-500 text-white":"bg-gray-100 dark:bg-gray-800 text-gray-600"}`,children:"GATv2 (dynamic)"})]}),e.jsxs("svg",{width:320,height:150,className:"mx-auto block",children:[r.slice(1).map((i,a)=>e.jsx("line",{x1:r[0].x,y1:r[0].y,x2:i.x,y2:i.y,stroke:"#7c3aed",strokeWidth:d[a+1]*10,opacity:.6},a)),r.map((i,a)=>e.jsxs("g",{children:[e.jsx("circle",{cx:i.x,cy:i.y,r:20,fill:a===0?"#7c3aed":"#e5e7eb",stroke:"#7c3aed",strokeWidth:1.5}),e.jsx("text",{x:i.x,y:i.y+4,textAnchor:"middle",fill:a===0?"white":"#374151",fontSize:10,children:i.features})]},a)),r.slice(1).map((i,a)=>e.jsx("text",{x:(r[0].x+i.x)/2+(a===0?-15:15),y:(r[0].y+i.y)/2,textAnchor:"middle",fill:"#7c3aed",fontSize:10,children:d[a+1].toFixed(1)},a))]}),e.jsx("p",{className:"text-center text-xs text-gray-500 mt-1",children:o==="v1"?"GATv1: attention ranking is fixed regardless of query node":"GATv2: attention ranking can change based on query features"})]})}function z(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:'GATv2 addresses a fundamental limitation of the original GAT: its attention is "static" and cannot compute dynamic attention where the ranking of neighbors depends on the query node.'}),e.jsxs(_,{title:"Static Attention Problem",id:"static-attention",children:[e.jsx("p",{children:"In GATv1, the attention function is:"}),e.jsx(t.BlockMath,{math:"e_{ij} = \\mathbf{a}^\\top \\text{LeakyReLU}\\!\\left([\\mathbf{W}\\mathbf{h}_i \\| \\mathbf{W}\\mathbf{h}_j]\\right)"}),e.jsxs("p",{className:"mt-2",children:["Because ",e.jsx(t.InlineMath,{math:"\\mathbf{a}"})," is split as ",e.jsx(t.InlineMath,{math:"[\\mathbf{a}_L \\| \\mathbf{a}_R]"}),", this becomes ",e.jsx(t.InlineMath,{math:"\\mathbf{a}_L^\\top \\mathbf{W}\\mathbf{h}_i + \\mathbf{a}_R^\\top \\mathbf{W}\\mathbf{h}_j"}),". The contribution of key ",e.jsx(t.InlineMath,{math:"j"})," is independent of query ",e.jsx(t.InlineMath,{math:"i"}),", so the attention ranking is the same for all queries."]})]}),e.jsxs(h,{title:"GATv2: Dynamic Attention",children:[e.jsxs("p",{children:["GATv2 applies the nonlinearity ",e.jsx("strong",{children:"before"})," the dot product with ",e.jsx(t.InlineMath,{math:"\\mathbf{a}"}),":"]}),e.jsx(t.BlockMath,{math:"e_{ij} = \\mathbf{a}^\\top \\text{LeakyReLU}\\!\\left(\\mathbf{W} [\\mathbf{h}_i \\| \\mathbf{h}_j]\\right)"}),e.jsxs("p",{className:"mt-2",children:["This allows the attention function to compute any ranking of neighbors as a function of the query, making it a ",e.jsx("strong",{children:"universal approximator"})," of attention."]})]}),e.jsx(E,{}),e.jsxs(g,{title:"When Does It Matter?",children:[e.jsx("p",{children:"Consider a knowledge graph where node A is connected to nodes B and C."}),e.jsx("p",{children:"When predicting A's profession, neighbor B (employer) should get high attention."}),e.jsx("p",{children:"When predicting A's hometown, neighbor C (family) should get high attention."}),e.jsx("p",{children:"GATv1 always assigns the same attention ranking; GATv2 can adapt based on context."})]}),e.jsx(u,{title:"GATv2 Implementation",code:`import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GATv2Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8):
        super().__init__()
        # GATv2Conv applies LeakyReLU before attention dot product
        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_dim * heads, out_dim, heads=1,
                               concat=False, dropout=0.6)

    def forward(self, x, edge_index, return_attention=False):
        x = F.dropout(x, p=0.6, training=self.training)
        x, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = x.relu()
        x = F.dropout(x, p=0.6, training=self.training)
        x, attn2 = self.conv2(x, edge_index, return_attention_weights=True)
        if return_attention:
            return x, attn1, attn2
        return x

model = GATv2Model(in_dim=16, hidden_dim=8, out_dim=7)
x = torch.randn(10, 16)
edge_index = torch.randint(0, 10, (2, 30))
out, attn1, attn2 = model(x, edge_index, return_attention=True)
print(f"Output: {out.shape}")
print(f"Attention (layer 1): edge_index {attn1[0].shape}, weights {attn1[1].shape}")`}),e.jsx(f,{type:"note",title:"Computational Cost",children:e.jsxs("p",{children:["GATv2 has the same computational complexity as GATv1 (",e.jsx(t.InlineMath,{math:"O(|\\mathcal{E}| \\cdot d)"}),") but empirically takes slightly longer due to the larger weight matrix ",e.jsx(t.InlineMath,{math:"\\mathbf{W}"}),". The expressiveness gain is well worth the small overhead on tasks requiring dynamic attention."]})})]})}const oe=Object.freeze(Object.defineProperty({__proto__:null,default:z},Symbol.toStringTag,{value:"Module"}));function B(){const[o,c]=p.useState(!0),r=[{x:60,y:60,type:"user",label:"U1"},{x:160,y:30,type:"user",label:"U2"},{x:260,y:60,type:"item",label:"I1"},{x:320,y:120,type:"item",label:"I2"},{x:100,y:140,type:"tag",label:"T1"}],d=[{from:0,to:2,type:"buys"},{from:1,to:2,type:"buys"},{from:1,to:3,type:"buys"},{from:0,to:1,type:"follows"},{from:2,to:4,type:"tagged"},{from:3,to:4,type:"tagged"}],i={user:"#7c3aed",item:"#f97316",tag:"#10b981"},a={buys:"#7c3aed",follows:"#f43f5e",tagged:"#10b981"};return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Heterogeneous Graph"}),e.jsx("div",{className:"flex items-center gap-4 mb-3",children:e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:[e.jsx("input",{type:"checkbox",checked:o,onChange:s=>c(s.target.checked),className:"accent-violet-500"}),"Color by type"]})}),e.jsxs("svg",{width:380,height:170,className:"mx-auto block",children:[d.map((s,n)=>e.jsx("line",{x1:r[s.from].x,y1:r[s.from].y,x2:r[s.to].x,y2:r[s.to].y,stroke:o?a[s.type]:"#d1d5db",strokeWidth:1.5,opacity:.6},n)),r.map((s,n)=>e.jsxs("g",{children:[e.jsx("circle",{cx:s.x,cy:s.y,r:18,fill:o?i[s.type]:"#7c3aed",opacity:.8}),e.jsx("text",{x:s.x,y:s.y+4,textAnchor:"middle",fill:"white",fontSize:10,fontWeight:"bold",children:s.label})]},n))]}),e.jsxs("div",{className:"flex justify-center gap-4 text-xs mt-2",children:[e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"w-3 h-3 rounded-full bg-violet-500 inline-block"})," User"]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"w-3 h-3 rounded-full bg-orange-500 inline-block"})," Item"]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"w-3 h-3 rounded-full bg-emerald-500 inline-block"})," Tag"]})]})]})}function W(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Real-world graphs often contain multiple types of nodes and edges. Heterogeneous graph neural networks handle this by using type-specific transformations and aggregations."}),e.jsxs(h,{title:"Heterogeneous Graph",children:[e.jsxs("p",{children:["A heterogeneous graph has node type mapping ",e.jsx(t.InlineMath,{math:"\\tau: \\mathcal{V} \\to \\mathcal{T}"})," and edge type mapping ",e.jsx(t.InlineMath,{math:"\\phi: \\mathcal{E} \\to \\mathcal{R}"}),":"]}),e.jsx(t.BlockMath,{math:"\\mathcal{G} = (\\mathcal{V}, \\mathcal{E}, \\tau, \\phi), \\quad |\\mathcal{T}| + |\\mathcal{R}| > 2"}),e.jsxs("p",{className:"mt-2",children:["Each relation ",e.jsx(t.InlineMath,{math:"r"})," connects a source type to a target type, forming a ",e.jsx("strong",{children:"metapath"})," schema."]})]}),e.jsx(B,{}),e.jsxs(h,{title:"Relational Graph Attention (R-GAT)",children:[e.jsx("p",{children:"Use relation-specific transformations:"}),e.jsx(t.BlockMath,{math:"\\mathbf{h}_i^{(l+1)} = \\sigma\\!\\left(\\sum_{r \\in \\mathcal{R}} \\sum_{j \\in \\mathcal{N}_r(i)} \\alpha_{ij}^r \\mathbf{W}_r \\mathbf{h}_j^{(l)}\\right)"}),e.jsxs("p",{className:"mt-2",children:["Each relation ",e.jsx(t.InlineMath,{math:"r"})," has its own weight matrix ",e.jsx(t.InlineMath,{math:"\\mathbf{W}_r"})," and attention parameters. Messages from different relation types are aggregated (sum, mean, or attention)."]})]}),e.jsxs(g,{title:"HAN: Hierarchical Attention",children:[e.jsx("p",{children:"Heterogeneous Attention Networks use two levels of attention:"}),e.jsxs("p",{children:[e.jsx("strong",{children:"Node-level"}),": Attention over neighbors within each metapath."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Semantic-level"}),": Attention over different metapaths to learn which relation types are most informative."]}),e.jsx("p",{children:"For an academic graph: Author-Paper-Author and Author-Paper-Venue-Paper-Author are two different metapaths."})]}),e.jsx(u,{title:"Heterogeneous GNN with PyG",code:`import torch
from torch_geometric.nn import HeteroConv, GATConv, Linear
from torch_geometric.data import HeteroData

# Create heterogeneous graph
data = HeteroData()
data['user'].x = torch.randn(100, 16)
data['item'].x = torch.randn(200, 32)
data['user', 'buys', 'item'].edge_index = torch.randint(0, 100, (2, 500))
data['user', 'follows', 'user'].edge_index = torch.randint(0, 100, (2, 300))

# Heterogeneous convolution: different conv per edge type
class HeteroGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Project all node types to same dimension
        self.lin_user = Linear(16, 64)
        self.lin_item = Linear(32, 64)
        # Different conv for each relation
        self.conv = HeteroConv({
            ('user', 'buys', 'item'): GATConv(64, 64),
            ('user', 'follows', 'user'): GATConv(64, 64),
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            'user': self.lin_user(x_dict['user']).relu(),
            'item': self.lin_item(x_dict['item']).relu(),
        }
        return self.conv(x_dict, edge_index_dict)

model = HeteroGNN()
out = model(data.x_dict, data.edge_index_dict)
print({k: v.shape for k, v in out.items()})`}),e.jsx(y,{title:"Parameter Explosion",children:e.jsxs("p",{children:["With ",e.jsx(t.InlineMath,{math:"|\\mathcal{R}|"})," relation types, each needing its own weight matrix, parameter count grows linearly with the number of relations. For knowledge graphs with hundreds of relations, use basis decomposition: ",e.jsx(t.InlineMath,{math:"\\mathbf{W}_r = \\sum_b a_{rb} \\mathbf{B}_b"}),"where ",e.jsx(t.InlineMath,{math:"\\mathbf{B}_b"})," are shared basis matrices."]})}),e.jsx(f,{type:"note",title:"Applications",children:e.jsx("p",{children:"Heterogeneous GNNs are used for knowledge graph reasoning, recommendation systems (user-item-attribute graphs), drug-target interaction prediction, and academic network analysis. They naturally encode the rich relational structure that homogeneous GNNs flatten."})})]})}const le=Object.freeze(Object.defineProperty({__proto__:null,default:W},Symbol.toStringTag,{value:"Module"}));function D(){const[o,c]=p.useState("laplacian"),r=["v0","v1","v2","v3","v4"],a=o==="laplacian"?[[.45,-.37],[.45,.37],[.45,.6],[.45,-.6],[.45,0]]:[[.33,.11],[.5,.25],[.33,.11],[.5,.25],[.25,.06]];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Graph Positional Encodings"}),e.jsxs("div",{className:"flex items-center gap-2 mb-3",children:[e.jsx("button",{onClick:()=>c("laplacian"),className:`px-3 py-1 rounded text-sm ${o==="laplacian"?"bg-violet-500 text-white":"bg-gray-100 dark:bg-gray-800 text-gray-600"}`,children:"Laplacian PE"}),e.jsx("button",{onClick:()=>c("rwse"),className:`px-3 py-1 rounded text-sm ${o==="rwse"?"bg-violet-500 text-white":"bg-gray-100 dark:bg-gray-800 text-gray-600"}`,children:"Random Walk SE"})]}),e.jsx("div",{className:"flex gap-2 justify-center flex-wrap",children:r.map((s,n)=>e.jsxs("div",{className:"px-3 py-2 rounded-lg bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-center",children:[e.jsx("div",{className:"text-xs text-gray-500 font-bold",children:s}),e.jsxs("div",{className:"text-xs font-mono text-violet-600 dark:text-violet-400",children:["[",a[n][0].toFixed(2),", ",a[n][1].toFixed(2),"]"]})]},n))}),e.jsx("p",{className:"text-center text-xs text-gray-500 mt-2",children:o==="laplacian"?"Laplacian PE: eigenvectors of graph Laplacian (sign ambiguity!)":"RWSE: diagonal of random walk matrix powers (no sign ambiguity)"})]})}function F(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Unlike sequences, graphs have no canonical ordering of nodes. Positional encodings for graphs must encode structural information (centrality, distance, substructure) without relying on a fixed node ordering."}),e.jsxs(h,{title:"Laplacian Positional Encoding",children:[e.jsxs("p",{children:["Use the first ",e.jsx(t.InlineMath,{math:"k"})," non-trivial eigenvectors of the normalized Laplacian:"]}),e.jsx(t.BlockMath,{math:"\\text{PE}(v_i) = [\\phi_2(v_i), \\phi_3(v_i), \\ldots, \\phi_{k+1}(v_i)] \\in \\mathbb{R}^k"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"L\\phi_j = \\lambda_j \\phi_j"}),". These capture graph structure at multiple scales."]})]}),e.jsxs(_,{title:"Sign Ambiguity Problem",id:"sign-ambiguity",children:[e.jsxs("p",{children:["Eigenvectors are defined up to sign: if ",e.jsx(t.InlineMath,{math:"\\phi"})," is an eigenvector, so is ",e.jsx(t.InlineMath,{math:"-\\phi"}),". This means Laplacian PE is not unique:"]}),e.jsx(t.BlockMath,{math:"\\phi_j \\text{ and } -\\phi_j \\text{ are equally valid}"}),e.jsx("p",{className:"mt-2",children:"Solutions: (1) use sign-invariant networks (SignNet), (2) use random sign augmentation during training, (3) use random walk encodings which avoid the issue entirely."})]}),e.jsxs(h,{title:"Random Walk Structural Encoding (RWSE)",children:[e.jsxs("p",{children:["The landing probability of a random walk returning to node ",e.jsx(t.InlineMath,{math:"i"})," after ",e.jsx(t.InlineMath,{math:"k"})," steps:"]}),e.jsx(t.BlockMath,{math:"\\text{RWSE}(v_i) = \\left[(\\hat{A}^1)_{ii}, (\\hat{A}^2)_{ii}, \\ldots, (\\hat{A}^K)_{ii}\\right]"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"\\hat{A} = D^{-1}A"}),". This encodes local structural information (degree at ",e.jsx(t.InlineMath,{math:"k=1"}),", triangle count at ",e.jsx(t.InlineMath,{math:"k=3"}),", etc.) without sign ambiguity."]})]}),e.jsx(D,{}),e.jsxs(g,{title:"Other Graph PEs",children:[e.jsxs("p",{children:[e.jsx("strong",{children:"Distance encoding"}),": shortest-path distances between node pairs."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Node degree"}),": simplest structural feature, often surprisingly effective."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Learnable PE"}),": Initialize random PE per node, learn via backpropagation."]}),e.jsx("p",{children:"Most graph transformers combine multiple PE types for best results."})]}),e.jsx(u,{title:"Computing Graph Positional Encodings",code:`import torch
import numpy as np
from scipy import sparse

def laplacian_pe(edge_index, num_nodes, k=8):
    """Compute Laplacian Positional Encoding."""
    # Build adjacency and Laplacian
    row, col = edge_index
    A = sparse.coo_matrix((np.ones(len(row)), (row, col)),
                          shape=(num_nodes, num_nodes))
    A = A + A.T  # symmetrize
    D = sparse.diags(np.array(A.sum(1)).flatten())
    L = D - A
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(np.array(A.sum(1)).flatten() + 1e-8))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    # Compute smallest eigenvectors (skip first = constant)
    eigenvalues, eigenvectors = sparse.linalg.eigsh(L_norm, k=k+1, which='SM')
    pe = torch.FloatTensor(eigenvectors[:, 1:k+1])  # skip first
    return pe

def random_walk_se(edge_index, num_nodes, walk_length=16):
    """Compute Random Walk Structural Encoding."""
    row, col = edge_index
    A = sparse.coo_matrix((np.ones(len(row)), (row, col)),
                          shape=(num_nodes, num_nodes)).tocsr()
    D_inv = sparse.diags(1.0 / (np.array(A.sum(1)).flatten() + 1e-8))
    RW = D_inv @ A  # random walk matrix
    pe = torch.zeros(num_nodes, walk_length)
    RW_power = sparse.eye(num_nodes)
    for k in range(walk_length):
        RW_power = RW_power @ RW
        pe[:, k] = torch.FloatTensor(RW_power.diagonal())
    return pe

# Example: 10 nodes, random edges
edge_index = np.array([np.random.randint(0, 10, 30), np.random.randint(0, 10, 30)])
lap_pe = laplacian_pe(edge_index, 10, k=4)
rw_pe = random_walk_se(edge_index, 10, walk_length=8)
print(f"Laplacian PE: {lap_pe.shape}, RWSE: {rw_pe.shape}")`}),e.jsx(f,{type:"note",title:"PE in Practice",children:e.jsx("p",{children:"Positional encodings are typically concatenated with or added to node features before being fed into the graph transformer. The PE dimension is a hyperparameter (commonly 16-64). GPS (General Powerful Scalable) uses both Laplacian PE and RWSE together."})})]})}const de=Object.freeze(Object.defineProperty({__proto__:null,default:F},Symbol.toStringTag,{value:"Module"}));function O(){const[o,c]=p.useState("sparse"),r=[{x:80,y:50},{x:180,y:30},{x:280,y:50},{x:130,y:130},{x:230,y:130}],d=[[0,1],[1,2],[0,3],[3,4],[1,4]],i=[];for(let s=0;s<5;s++)for(let n=s+1;n<5;n++)i.push([s,n]);const a=o==="sparse"?d:i;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Sparse vs Full Attention"}),e.jsxs("div",{className:"flex items-center gap-2 mb-3",children:[e.jsx("button",{onClick:()=>c("sparse"),className:`px-3 py-1 rounded text-sm ${o==="sparse"?"bg-violet-500 text-white":"bg-gray-100 dark:bg-gray-800 text-gray-600"}`,children:"GNN (neighbors only)"}),e.jsx("button",{onClick:()=>c("full"),className:`px-3 py-1 rounded text-sm ${o==="full"?"bg-violet-500 text-white":"bg-gray-100 dark:bg-gray-800 text-gray-600"}`,children:"Graph Transformer (all pairs)"})]}),e.jsxs("svg",{width:360,height:160,className:"mx-auto block",children:[a.map(([s,n],l)=>e.jsx("line",{x1:r[s].x,y1:r[s].y,x2:r[n].x,y2:r[n].y,stroke:o==="sparse"?"#7c3aed":"#f97316",strokeWidth:1.5,opacity:.4},l)),r.map((s,n)=>e.jsx("circle",{cx:s.x,cy:s.y,r:16,fill:o==="sparse"?"#7c3aed":"#f97316",opacity:.8},n))]}),e.jsxs("p",{className:"text-center text-xs text-gray-500 mt-1",children:[o==="sparse"?`${d.length} edges (O(|E|))`:`${i.length} edges (O(N^2))`," -- full attention captures long-range but costs more"]})]})}function R(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Graph Transformers apply full self-attention to graph nodes, overcoming the limited receptive field of message-passing GNNs. The key challenge is encoding graph structure into the attention mechanism."}),e.jsxs(h,{title:"Graph Transformer Layer",children:[e.jsx("p",{children:"Full self-attention with structural bias:"}),e.jsx(t.BlockMath,{math:"\\text{Attn}(Q, K, V) = \\text{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d}} + B\\right) V"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"B \\in \\mathbb{R}^{N \\times N}"})," is a structural bias encoding graph topology (e.g., shortest-path distances, edge features, or spatial encoding)."]})]}),e.jsx(O,{}),e.jsxs(_,{title:"Structural Encoding via Attention Bias",id:"structural-bias",children:[e.jsxs("p",{children:["The bias term ",e.jsx(t.InlineMath,{math:"B_{ij}"})," can encode various structural features:"]}),e.jsx(t.BlockMath,{math:"B_{ij} = \\phi_\\text{dist}(d(i,j)) + \\phi_\\text{edge}(e_{ij}) + \\phi_\\text{degree}(\\deg(i), \\deg(j))"}),e.jsx("p",{className:"mt-2",children:"Graphormer uses centrality encoding (degree), spatial encoding (shortest path), and edge encoding. This allows the Transformer to be aware of graph structure without being limited to local neighborhoods."})]}),e.jsxs(g,{title:"Graphormer (Ying et al., 2021)",children:[e.jsx("p",{children:"Graphormer won 1st place in the OGB Large-Scale Challenge using three structural encodings:"}),e.jsxs("p",{children:[e.jsx("strong",{children:"Centrality"}),": ",e.jsx(t.InlineMath,{math:"h_i^{(0)} = x_i + z^-_{\\deg^-(i)} + z^+_{\\deg^+(i)}"})]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Spatial"}),": ",e.jsx(t.InlineMath,{math:"B_{ij} = b_{\\phi(v_i, v_j)}"})," where ",e.jsx(t.InlineMath,{math:"\\phi"})," is shortest-path distance."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Edge"}),": Average edge features along the shortest path between nodes."]})]}),e.jsx(u,{title:"Graph Transformer Layer",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, attn_bias=None):
        """
        x: (B, N, d_model) node features
        attn_bias: (B*H, N, N) structural bias (optional)
        """
        # Self-attention with structural bias
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_bias)
        x = x + h

        # Feed-forward
        x = x + self.ffn(self.norm2(x))
        return x

# Example
layer = GraphTransformerLayer(d_model=64, n_heads=8)
x = torch.randn(2, 10, 64)  # batch of 2, 10 nodes, dim 64
out = layer(x)
print(f"Input: {x.shape} -> Output: {out.shape}")`}),e.jsx(f,{type:"note",title:"Scalability Challenge",children:e.jsxs("p",{children:["Full self-attention is ",e.jsx(t.InlineMath,{math:"O(N^2)"})," in the number of nodes, limiting graph transformers to small/medium graphs (thousands of nodes). For larger graphs, sparse attention patterns, neighborhood sampling, or hybrid approaches (GPS) that combine local MPNN with global attention are necessary."]})})]})}const ce=Object.freeze(Object.defineProperty({__proto__:null,default:R},Symbol.toStringTag,{value:"Module"}));function q(){const[o,c]=p.useState({mpnn:!0,attn:!0,ffn:!0}),r=i=>c(a=>({...a,[i]:!a[i]})),d=[{key:"mpnn",label:"Local MPNN",color:"#7c3aed",desc:"Neighbor aggregation"},{key:"attn",label:"Global Attn",color:"#f97316",desc:"Full self-attention"},{key:"ffn",label:"FFN",color:"#10b981",desc:"Feed-forward"}];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"GPS Layer Architecture"}),e.jsx("div",{className:"flex items-center gap-2 mb-3",children:d.map(i=>e.jsxs("label",{className:"flex items-center gap-1 text-sm text-gray-600 dark:text-gray-400",children:[e.jsx("input",{type:"checkbox",checked:o[i.key],onChange:()=>r(i.key),className:"accent-violet-500"}),i.label]},i.key))}),e.jsxs("div",{className:"flex items-center gap-2 justify-center",children:[e.jsx("div",{className:"px-3 py-2 rounded bg-gray-100 dark:bg-gray-800 text-xs text-center border",children:"Input h"}),e.jsx("span",{className:"text-gray-400",children:"→"}),d.filter(i=>o[i.key]).map((i,a)=>e.jsxs("div",{className:"flex items-center gap-2",children:[e.jsxs("div",{className:"px-3 py-2 rounded text-xs text-center text-white font-medium",style:{backgroundColor:i.color},children:[i.label,e.jsx("div",{className:"text-[10px] opacity-80",children:i.desc})]}),e.jsx("span",{className:"text-gray-400",children:a<d.filter(s=>o[s.key]).length-1?"+":"&#8594;"})]},i.key)),e.jsx("div",{className:"px-3 py-2 rounded bg-gray-100 dark:bg-gray-800 text-xs text-center border",children:"Output h'"})]}),e.jsx("p",{className:"text-center text-xs text-gray-500 mt-2",children:"GPS combines local and global processing in each layer"})]})}function U(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"GPS (General, Powerful, Scalable) is a hybrid architecture that combines local message passing with global self-attention in each layer, achieving state-of-the-art results while remaining scalable."}),e.jsxs(h,{title:"GPS Layer",children:[e.jsx("p",{children:"Each GPS layer applies MPNN and Transformer attention in parallel:"}),e.jsx(t.BlockMath,{math:"\\mathbf{h}_i' = \\mathbf{h}_i + \\underbrace{\\text{MPNN}(\\mathbf{h}_i, \\{\\mathbf{h}_j : j \\in \\mathcal{N}(i)\\})}_{\\text{local}} + \\underbrace{\\text{Attn}(\\mathbf{h}_i, \\mathbf{H})}_{\\text{global}}"}),e.jsx(t.BlockMath,{math:"\\mathbf{h}_i'' = \\mathbf{h}_i' + \\text{FFN}(\\text{Norm}(\\mathbf{h}_i'))"}),e.jsx("p",{className:"mt-2",children:"The MPNN captures local graph structure; global attention captures long-range dependencies."})]}),e.jsx(q,{}),e.jsxs(_,{title:"Why Hybrid?",id:"hybrid-motivation",children:[e.jsxs("p",{children:["Pure MPNN: limited to ",e.jsx(t.InlineMath,{math:"K"}),"-hop neighborhoods after ",e.jsx(t.InlineMath,{math:"K"})," layers. Misses long-range interactions."]}),e.jsxs("p",{children:["Pure Transformer: ",e.jsx(t.InlineMath,{math:"O(N^2)"})," cost, may ignore useful graph structure."]}),e.jsx("p",{className:"mt-2",children:"GPS combines both: the MPNN provides strong structural bias and the Transformer provides long-range information flow, with each compensating the other's weakness."})]}),e.jsxs(g,{title:"GPS Recipe",children:[e.jsx("p",{children:"The recommended GPS configuration:"}),e.jsxs("p",{children:[e.jsx("strong",{children:"PE"}),": Laplacian PE (k=16) + RWSE (k=16), processed by a small MLP."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"MPNN"}),": GatedGCN or GINE for the local component."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Attention"}),": Standard multi-head self-attention (Performer for large graphs)."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Depth"}),": 10-16 layers with pre-norm and residual connections."]}),e.jsx("p",{children:"GPS achieves state-of-the-art on ZINC, PCQM4Mv2, and Peptides benchmarks."})]}),e.jsx(u,{title:"GPS Layer Implementation",code:`import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, GPSConv

class GPSModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, pe_dim=16, n_layers=6):
        super().__init__()
        self.pe_encoder = nn.Linear(pe_dim, hidden_dim)
        self.node_encoder = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            # Local MPNN: GIN with edge features
            local_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            local_model = GINEConv(local_nn)
            # GPS wraps local MPNN + global attention
            gps_layer = GPSConv(
                channels=hidden_dim,
                conv=local_model,
                heads=4,
                dropout=0.1,
                attn_type='multihead',  # or 'performer' for O(N)
            )
            self.layers.append(gps_layer)

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, pe, edge_index, edge_attr, batch):
        h = self.node_encoder(x) + self.pe_encoder(pe)
        for layer in self.layers:
            h = layer(h, edge_index, batch, edge_attr=edge_attr)
        # Graph-level readout
        from torch_geometric.nn import global_mean_pool
        h_graph = global_mean_pool(h, batch)
        return self.output(h_graph)

model = GPSModel(in_dim=16, hidden_dim=64, out_dim=1)
print(f"GPS params: {sum(p.numel() for p in model.parameters()):,}")`}),e.jsx(f,{type:"note",title:"Scalability with Linear Attention",children:e.jsxs("p",{children:["For graphs with thousands of nodes, the ",e.jsx(t.InlineMath,{math:"O(N^2)"})," attention becomes a bottleneck. GPS supports Performer attention (",e.jsx(t.InlineMath,{math:"O(N)"}),") as a drop-in replacement. This makes GPS scalable to large molecular graphs and biological networks while retaining the benefits of global information flow."]})})]})}const he=Object.freeze(Object.defineProperty({__proto__:null,default:U},Symbol.toStringTag,{value:"Module"}));function V(){const[o,c]=p.useState(!1),r=[{x:80,y:80,elem:"C",color:"#374151"},{x:160,y:40,elem:"C",color:"#374151"},{x:240,y:80,elem:"O",color:"#dc2626"},{x:160,y:120,elem:"N",color:"#2563eb"},{x:80,y:160,elem:"C",color:"#374151"}],d=[{from:0,to:1,type:"single"},{from:1,to:2,type:"double"},{from:1,to:3,type:"single"},{from:3,to:4,type:"single"},{from:4,to:0,type:"single"}];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Molecule as a Graph"}),e.jsx("div",{className:"flex items-center gap-4 mb-3",children:e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:[e.jsx("input",{type:"checkbox",checked:o,onChange:i=>c(i.target.checked),className:"accent-violet-500"}),"Show node features"]})}),e.jsxs("svg",{width:320,height:200,className:"mx-auto block",children:[d.map((i,a)=>{const s=i.type==="double"?3:0;return e.jsxs("g",{children:[e.jsx("line",{x1:r[i.from].x,y1:r[i.from].y-s,x2:r[i.to].x,y2:r[i.to].y-s,stroke:"#9ca3af",strokeWidth:2}),i.type==="double"&&e.jsx("line",{x1:r[i.from].x,y1:r[i.from].y+s,x2:r[i.to].x,y2:r[i.to].y+s,stroke:"#9ca3af",strokeWidth:2})]},a)}),r.map((i,a)=>e.jsxs("g",{children:[e.jsx("circle",{cx:i.x,cy:i.y,r:20,fill:i.color,opacity:.85}),e.jsx("text",{x:i.x,y:i.y+5,textAnchor:"middle",fill:"white",fontSize:14,fontWeight:"bold",children:i.elem}),o&&e.jsx("text",{x:i.x,y:i.y+32,textAnchor:"middle",fill:"#7c3aed",fontSize:8,children:"[Z, deg, hyb, ...]"})]},a))]}),e.jsx("p",{className:"text-center text-xs text-gray-500 mt-1",children:"Atoms = nodes, bonds = edges, with rich chemical features"})]})}function H(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Molecules are naturally represented as graphs, with atoms as nodes and bonds as edges. GNNs have become the dominant approach for molecular property prediction, with applications in drug discovery, materials science, and chemical engineering."}),e.jsxs(h,{title:"Molecular Graph Representation",children:[e.jsx("p",{children:"Node features (per atom):"}),e.jsx(t.BlockMath,{math:"\\mathbf{x}_i = [\\text{atomic\\_num}, \\text{degree}, \\text{formal\\_charge}, \\text{hybridization}, \\text{aromaticity}, \\ldots]"}),e.jsx("p",{className:"mt-2",children:"Edge features (per bond):"}),e.jsx(t.BlockMath,{math:"\\mathbf{e}_{ij} = [\\text{bond\\_type}, \\text{is\\_conjugated}, \\text{is\\_ring}, \\text{stereo}]"})]}),e.jsx(V,{}),e.jsxs(h,{title:"Graph-Level Property Prediction",children:[e.jsx("p",{children:"For molecular properties (solubility, toxicity, binding affinity):"}),e.jsx(t.BlockMath,{math:"\\hat{y} = \\text{MLP}\\!\\left(\\bigoplus_{i \\in \\mathcal{V}} \\mathbf{h}_i^{(L)}\\right)"}),e.jsxs("p",{className:"mt-2",children:["The readout ",e.jsx(t.InlineMath,{math:"\\bigoplus"})," aggregates all atom representations into a single molecular fingerprint. Sum pooling preserves size information; mean pooling is size-invariant."]})]}),e.jsxs(g,{title:"Key Benchmarks",children:[e.jsxs("p",{children:[e.jsx("strong",{children:"ogbg-molhiv"}),": Predict HIV inhibition (41K molecules, binary classification, AUC-ROC ~80%)."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"PCQM4Mv2"}),": Predict HOMO-LUMO gap (3.8M molecules, regression, MAE ~0.085 eV)."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"QM9"}),": 12 quantum chemical properties for 134K small molecules."]}),e.jsx("p",{children:"State-of-the-art models (GPS, Graphormer) achieve chemical accuracy on several properties."})]}),e.jsx(u,{title:"Molecular GNN with PyG and RDKit",code:`import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.datasets import MoleculeNet

# Load HIV dataset
dataset = MoleculeNet(root='/tmp/hiv', name='HIV')
print(f"Molecules: {len(dataset)}, Features: {dataset.num_features}")

class MolGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1, n_layers=4):
        super().__init__()
        self.atom_encoder = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.layers.append(GINEConv(mlp, edge_dim=3))
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.atom_encoder(x.float())
        for conv in self.layers:
            h = h + conv(h, edge_index, edge_attr.float())  # residual
            h = h.relu()
        h_mol = global_add_pool(h, batch)  # sum over atoms
        return self.output(h_mol)

model = MolGNN(in_dim=9, hidden_dim=128)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")`}),e.jsx(f,{type:"note",title:"3D Molecular Graphs",children:e.jsx("p",{children:"Many molecular properties depend on 3D geometry (distances, angles, dihedral angles). Equivariant GNNs like SchNet, DimeNet, and PaiNN incorporate 3D coordinates as continuous edge features, respecting physical symmetries (rotation, translation, reflection). This is crucial for force fields and protein structure prediction."})})]})}const me=Object.freeze(Object.defineProperty({__proto__:null,default:H},Symbol.toStringTag,{value:"Module"}));function K(){const[o,c]=p.useState(.5),r=[{x:60,y:50,c:0},{x:100,y:30,c:0},{x:130,y:70,c:0},{x:230,y:40,c:1},{x:270,y:60,c:1},{x:250,y:100,c:1},{x:160,y:130,c:2},{x:120,y:150,c:2}],d=[[0,1],[1,2],[0,2],[3,4],[4,5],[3,5],[2,6],[5,6],[6,7]],i=["#7c3aed","#f97316","#10b981"],a=[.9,.85,.88,.92,.87,.9,.3,.35,.82];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Community Detection"}),e.jsx("div",{className:"flex items-center gap-4 mb-3",children:e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Similarity threshold: ",o.toFixed(2),e.jsx("input",{type:"range",min:0,max:1,step:.05,value:o,onChange:s=>c(parseFloat(s.target.value)),className:"w-28 accent-violet-500"})]})}),e.jsxs("svg",{width:340,height:180,className:"mx-auto block",children:[d.map(([s,n],l)=>e.jsx("line",{x1:r[s].x,y1:r[s].y,x2:r[n].x,y2:r[n].y,stroke:a[l]>=o?"#7c3aed":"#e5e7eb",strokeWidth:a[l]>=o?2:1,opacity:a[l]>=o?.6:.3},l)),r.map((s,n)=>e.jsx("circle",{cx:s.x,cy:s.y,r:14,fill:i[s.c],opacity:.8},n))]}),e.jsx("p",{className:"text-center text-xs text-gray-500 mt-1",children:"GNN embeddings reveal community structure through learned similarities"})]})}function $(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Social networks are among the most natural applications of GNNs. Users are nodes, relationships are edges, and tasks include community detection, link prediction, influence modeling, and content recommendation."}),e.jsxs(h,{title:"Link Prediction",children:[e.jsxs("p",{children:["Predict the likelihood of a future edge between nodes ",e.jsx(t.InlineMath,{math:"u"})," and ",e.jsx(t.InlineMath,{math:"v"}),":"]}),e.jsx(t.BlockMath,{math:"P(e_{uv}) = \\sigma\\!\\left(\\mathbf{h}_u^\\top \\mathbf{h}_v\\right) \\quad \\text{or} \\quad P(e_{uv}) = \\text{MLP}([\\mathbf{h}_u \\| \\mathbf{h}_v \\| \\mathbf{h}_u \\odot \\mathbf{h}_v])"}),e.jsxs("p",{className:"mt-2",children:["Training uses negative sampling: for each positive edge, sample ",e.jsx(t.InlineMath,{math:"K"})," random non-edges as negatives."]})]}),e.jsx(K,{}),e.jsxs(h,{title:"Influence Maximization",children:[e.jsxs("p",{children:["Find a seed set ",e.jsx(t.InlineMath,{math:"S"})," of ",e.jsx(t.InlineMath,{math:"k"})," nodes that maximizes information spread:"]}),e.jsx(t.BlockMath,{math:"S^* = \\arg\\max_{|S| = k} \\mathbb{E}[|\\text{Influenced}(S)|]"}),e.jsx("p",{className:"mt-2",children:"GNNs can learn node influence scores by encoding network structure and historical cascade data, replacing expensive Monte Carlo simulations."})]}),e.jsxs(g,{title:"Social Network Tasks",children:[e.jsxs("p",{children:[e.jsx("strong",{children:"Friend recommendation"}),": Link prediction on the social graph (Facebook, LinkedIn)."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Fake account detection"}),": Node classification using structural features (suspicious connectivity patterns)."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Content virality"}),": Predict which posts will spread based on the poster's graph neighborhood."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Community detection"}),": Unsupervised clustering of users into interest groups."]})]}),e.jsx(u,{title:"Link Prediction with GNN",code:`import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling

class LinkPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

    def encode(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        return self.conv2(h, edge_index)

    def decode(self, z, edge_index):
        """Dot product link predictor."""
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, edge_index, pos_edges, neg_edges):
        z = self.encode(x, edge_index)
        pos_score = self.decode(z, pos_edges)
        neg_score = self.decode(z, neg_edges)
        return pos_score, neg_score

def link_pred_loss(pos_score, neg_score):
    pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-8).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-8).mean()
    return pos_loss + neg_loss

model = LinkPredictor(in_dim=16, hidden_dim=64)
x = torch.randn(1000, 16)
edge_index = torch.randint(0, 1000, (2, 5000))
neg_edges = negative_sampling(edge_index, num_nodes=1000)
pos_score, neg_score = model(x, edge_index, edge_index, neg_edges)
loss = link_pred_loss(pos_score, neg_score)
print(f"Loss: {loss.item():.4f}")`}),e.jsx(y,{title:"Scalability at Web Scale",children:e.jsx("p",{children:"Social networks have billions of nodes and edges. Full-batch GNN training is impossible. Production systems use mini-batch training with neighbor sampling (PinSage at Pinterest), distributed training across machines, and quantization of embeddings. Inference often pre-computes embeddings offline and serves them via ANN indices."})}),e.jsx(f,{type:"note",title:"Dynamic Graphs",children:e.jsx("p",{children:"Social networks evolve over time: users join, friendships form and dissolve. Temporal GNNs (TGN, TGAT) incorporate timestamps into message passing, learning from the sequence of graph snapshots rather than a single static graph."})})]})}const xe=Object.freeze(Object.defineProperty({__proto__:null,default:$},Symbol.toStringTag,{value:"Module"}));function Q(){const[o,c]=p.useState(0),r=["U0","U1","U2","U3"],d=["I0","I1","I2","I3","I4"],i=[[0,0],[0,2],[0,3],[1,1],[1,2],[2,0],[2,4],[3,1],[3,3],[3,4]],a=r.map((n,l)=>({x:50,y:30+l*40})),s=d.map((n,l)=>({x:280,y:20+l*38}));return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"User-Item Bipartite Graph"}),e.jsx("div",{className:"flex items-center gap-2 mb-3",children:r.map((n,l)=>e.jsx("button",{onClick:()=>c(l),className:`px-3 py-1 rounded text-sm ${l===o?"bg-violet-500 text-white":"bg-gray-100 dark:bg-gray-800 text-gray-600"}`,children:n},l))}),e.jsxs("svg",{width:340,height:200,className:"mx-auto block",children:[i.map(([n,l],m)=>{const x=n===o;return e.jsx("line",{x1:a[n].x+20,y1:a[n].y,x2:s[l].x-20,y2:s[l].y,stroke:x?"#7c3aed":"#e5e7eb",strokeWidth:x?2:1,opacity:x?.8:.3},m)}),a.map((n,l)=>e.jsxs("g",{children:[e.jsx("circle",{cx:n.x,cy:n.y,r:16,fill:l===o?"#7c3aed":"#e5e7eb",stroke:"#7c3aed",strokeWidth:1}),e.jsx("text",{x:n.x,y:n.y+4,textAnchor:"middle",fill:l===o?"white":"#374151",fontSize:10,fontWeight:"bold",children:r[l]})]},`u${l}`)),s.map((n,l)=>{const m=i.some(([x,j])=>x===o&&j===l);return e.jsxs("g",{children:[e.jsx("rect",{x:n.x-16,y:n.y-14,width:32,height:28,rx:4,fill:m?"#f97316":"#e5e7eb",stroke:"#f97316",strokeWidth:1}),e.jsx("text",{x:n.x,y:n.y+4,textAnchor:"middle",fill:m?"white":"#374151",fontSize:10,fontWeight:"bold",children:d[l]})]},`i${l}`)})]}),e.jsx("p",{className:"text-center text-xs text-gray-500 mt-1",children:"Select a user to see their item interactions"})]})}function Z(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Recommendation systems naturally fit the graph framework: users and items form a bipartite graph connected by interactions. GNN-based methods learn embeddings that capture collaborative filtering signals through message passing."}),e.jsxs(h,{title:"GNN-Based Collaborative Filtering",children:[e.jsxs("p",{children:["Model the user-item bipartite graph. After ",e.jsx(t.InlineMath,{math:"L"})," layers of message passing:"]}),e.jsx(t.BlockMath,{math:"\\mathbf{e}_u^{(l+1)} = \\text{AGG}\\!\\left(\\left\\{\\mathbf{e}_i^{(l)} : i \\in \\mathcal{N}_u\\right\\}\\right)"}),e.jsx(t.BlockMath,{math:"\\mathbf{e}_i^{(l+1)} = \\text{AGG}\\!\\left(\\left\\{\\mathbf{e}_u^{(l)} : u \\in \\mathcal{N}_i\\right\\}\\right)"}),e.jsx("p",{className:"mt-2",children:"The predicted preference score:"}),e.jsx(t.BlockMath,{math:"\\hat{y}_{ui} = \\mathbf{e}_u^{\\top} \\mathbf{e}_i"})]}),e.jsx(Q,{}),e.jsxs(_,{title:"LightGCN Simplification",id:"lightgcn",children:[e.jsx("p",{children:"LightGCN (He et al., 2020) removes feature transformations and nonlinearities from GCN, using only neighborhood aggregation:"}),e.jsx(t.BlockMath,{math:"\\mathbf{e}_u^{(l+1)} = \\sum_{i \\in \\mathcal{N}_u} \\frac{1}{\\sqrt{|\\mathcal{N}_u|}\\sqrt{|\\mathcal{N}_i|}} \\mathbf{e}_i^{(l)}"}),e.jsx("p",{className:"mt-2",children:"Final embeddings are the weighted sum across layers:"}),e.jsx(t.BlockMath,{math:"\\mathbf{e}_u = \\sum_{l=0}^{L} \\alpha_l \\, \\mathbf{e}_u^{(l)}, \\quad \\alpha_l = \\frac{1}{L+1}"}),e.jsx("p",{children:"This simple design outperforms more complex models, showing that for recommendation, the core benefit of GNNs is multi-hop connectivity."})]}),e.jsxs(g,{title:"Real-World Deployments",children:[e.jsxs("p",{children:[e.jsx("strong",{children:"PinSage"})," (Pinterest): 3B nodes, 18B edges. Uses random walk-based sampling and efficient MapReduce training."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Uber Eats"}),": GNN for restaurant recommendation based on user-restaurant-dish graph."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Amazon"}),": Product recommendation using product co-purchase and co-view graphs."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"TikTok"}),": Video recommendation incorporating user-video-creator heterogeneous graph."]})]}),e.jsx(u,{title:"LightGCN for Recommendation",code:`import torch
import torch.nn as nn
from torch_geometric.nn import LGConv

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64, n_layers=3):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.convs = nn.ModuleList([LGConv() for _ in range(n_layers)])
        self.n_layers = n_layers
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def forward(self, edge_index):
        x = torch.cat([self.user_emb.weight, self.item_emb.weight])
        embs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            embs.append(x)
        # Layer combination (mean)
        out = torch.stack(embs, dim=0).mean(dim=0)
        return out

    def recommend(self, user_ids, edge_index, k=10):
        all_embs = self.forward(edge_index)
        n_users = self.user_emb.weight.shape[0]
        user_embs = all_embs[user_ids]
        item_embs = all_embs[n_users:]
        scores = user_embs @ item_embs.T
        _, top_k = scores.topk(k, dim=-1)
        return top_k

model = LightGCN(num_users=1000, num_items=5000, embed_dim=64)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")`}),e.jsx(f,{type:"note",title:"Beyond Collaborative Filtering",children:e.jsxs("p",{children:["Modern GNN-based recommenders incorporate side information (item descriptions, user profiles), knowledge graphs (product categories, attributes), and temporal signals (session-based sequences). The graph structure enables reasoning about",e.jsx("em",{children:"why"})," an item is recommended through path-based explanations."]})})]})}const pe=Object.freeze(Object.defineProperty({__proto__:null,default:Z},Symbol.toStringTag,{value:"Module"}));export{te as a,ae as b,se as c,ne as d,re as e,ie as f,oe as g,le as h,de as i,ce as j,he as k,me as l,xe as m,pe as n,ee as s};

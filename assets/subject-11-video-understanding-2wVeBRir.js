import{j as e,r as p}from"./vendor-DpISuAX6.js";import{r as t}from"./vendor-katex-CbWCYdth.js";import{D as x,T as _,E as f,P as u,N as g,W as v}from"./subject-01-foundations-D0A1VJsr.js";function b(){const[n,h]=p.useState(3),[i,s]=p.useState(3),[a,r]=p.useState(16),c=n*i*i*64*64,l=i*i*64*64,o=i*i*64*64+n*64*64;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"3D Convolution Parameter Explorer"}),e.jsxs("div",{className:"flex flex-wrap gap-4 mb-4",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Temporal kernel: ",n,e.jsx("input",{type:"range",min:1,max:7,step:2,value:n,onChange:d=>h(Number(d.target.value)),className:"w-24 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Spatial kernel: ",i,e.jsx("input",{type:"range",min:1,max:7,step:2,value:i,onChange:d=>s(Number(d.target.value)),className:"w-24 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Input frames: ",a,e.jsx("input",{type:"range",min:4,max:64,step:4,value:a,onChange:d=>r(Number(d.target.value)),className:"w-24 accent-violet-500"})]})]}),e.jsxs("div",{className:"grid grid-cols-3 gap-3 text-sm",children:[e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3",children:[e.jsx("p",{className:"text-xs text-violet-600 dark:text-violet-400 font-semibold",children:"Full 3D Conv"}),e.jsxs("p",{className:"text-lg font-bold text-violet-600",children:[(c/1e3).toFixed(1),"K params"]}),e.jsxs("p",{className:"text-xs text-gray-500",children:["kernel: ",n,"x",i,"x",i]})]}),e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3",children:[e.jsx("p",{className:"text-xs text-violet-600 dark:text-violet-400 font-semibold",children:"2D Conv (per frame)"}),e.jsxs("p",{className:"text-lg font-bold text-violet-600",children:[(l/1e3).toFixed(1),"K params"]}),e.jsxs("p",{className:"text-xs text-gray-500",children:["kernel: ",i,"x",i]})]}),e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3",children:[e.jsx("p",{className:"text-xs text-violet-600 dark:text-violet-400 font-semibold",children:"(2+1)D Factorized"}),e.jsxs("p",{className:"text-lg font-bold text-violet-600",children:[(o/1e3).toFixed(1),"K params"]}),e.jsx("p",{className:"text-xs text-gray-500",children:"spatial + temporal"})]})]}),e.jsxs("p",{className:"mt-2 text-xs text-gray-500",children:["Input volume: [",a,", ",64,", 224, 224] — Full 3D is ",(c/o).toFixed(1),"x more parameters than (2+1)D"]})]})}function j(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"3D convolutional networks extend image CNNs to video by learning spatiotemporal features jointly. From C3D to I3D, these architectures capture motion and temporal patterns that frame-level models miss entirely."}),e.jsxs(x,{title:"3D Convolution",children:[e.jsx("p",{children:"A 3D convolution operates over time, height, and width simultaneously:"}),e.jsx(t.BlockMath,{math:"y[t,h,w] = \\sum_{\\tau,i,j} W[\\tau,i,j] \\cdot x[t+\\tau, h+i, w+j] + b"}),e.jsxs("p",{className:"mt-2",children:["The kernel has shape ",e.jsx(t.InlineMath,{math:"(k_t, k_h, k_w)"}),", where ",e.jsx(t.InlineMath,{math:"k_t"})," is the temporal extent. This enables learning motion patterns, speed changes, and temporal textures directly from raw video clips."]})]}),e.jsx(b,{}),e.jsxs(_,{title:"Inflating 2D to 3D: I3D",id:"i3d-inflation",children:[e.jsx("p",{children:"I3D (Inflated 3D ConvNets) inflates pre-trained 2D ImageNet weights into 3D by repeating along the temporal dimension and rescaling:"}),e.jsx(t.BlockMath,{math:"W_\\text{3D}[\\tau, i, j] = \\frac{1}{k_t} W_\\text{2D}[i, j] \\quad \\forall \\tau \\in \\{1, \\ldots, k_t\\}"}),e.jsx("p",{className:"mt-1",children:"This preserves the spatial semantics learned on ImageNet while enabling temporal learning. I3D with Two-Stream (RGB + optical flow) achieved breakthrough results on Kinetics."})]}),e.jsxs(f,{title:"(2+1)D Factorization",children:[e.jsx("p",{children:"R(2+1)D decomposes 3D convolution into spatial and temporal components:"}),e.jsx(t.BlockMath,{math:"3\\text{D conv} \\approx (1 \\times k \\times k) \\text{ spatial} + (k_t \\times 1 \\times 1) \\text{ temporal}"}),e.jsx("p",{className:"mt-1",children:"This doubles the number of nonlinearities (one after each sub-convolution) and reduces parameters. The factorization makes optimization easier while maintaining the ability to model spatiotemporal features."})]}),e.jsx(u,{title:"3D CNNs for Video Classification",code:`import torch
import torch.nn as nn

# Basic 3D convolution
conv3d = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
video_clip = torch.randn(2, 3, 16, 224, 224)  # [B, C, T, H, W]
features = conv3d(video_clip)
print(f"3D conv output: {features.shape}")  # [2, 64, 16, 112, 112]

# (2+1)D factorized convolution
class R2Plus1DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_kernel=3, s_kernel=3):
        super().__init__()
        mid_ch = (in_ch * out_ch * s_kernel**2 * t_kernel) // (in_ch * s_kernel**2 + out_ch * t_kernel)
        self.spatial = nn.Conv3d(in_ch, mid_ch, (1, s_kernel, s_kernel), padding=(0, s_kernel//2, s_kernel//2))
        self.temporal = nn.Conv3d(mid_ch, out_ch, (t_kernel, 1, 1), padding=(t_kernel//2, 0, 0))
        self.bn1 = nn.BatchNorm3d(mid_ch)
        self.bn2 = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        x = torch.relu(self.bn1(self.spatial(x)))
        return torch.relu(self.bn2(self.temporal(x)))

block = R2Plus1DBlock(64, 64)
x = torch.randn(2, 64, 16, 56, 56)
out = block(x)
print(f"(2+1)D output: {out.shape}")  # [2, 64, 16, 56, 56]

# Full 3D params vs (2+1)D params
full_3d_params = 3 * 3 * 3 * 64 * 64
r21d_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
print(f"Full 3D: {full_3d_params:,} vs (2+1)D: {r21d_params:,}")`}),e.jsx(g,{type:"note",title:"Two-Stream Hypothesis",children:e.jsx("p",{children:"The two-stream architecture processes RGB frames (appearance) and optical flow (motion) through separate networks, fusing predictions at the end. While I3D showed optical flow helps significantly, modern approaches like SlowFast learn temporal patterns directly from RGB, reducing the need for expensive optical flow computation."})})]})}const U=Object.freeze(Object.defineProperty({__proto__:null,default:j},Symbol.toStringTag,{value:"Module"}));function y(){const[n,h]=p.useState(8),[i,s]=p.useState(8),a=64,r=Math.floor(a/n),c=a,l=64,o=Math.floor(l/i);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"SlowFast Dual-Pathway Design"}),e.jsxs("div",{className:"flex flex-wrap gap-4 mb-4",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Alpha (temporal stride): ",n,e.jsx("input",{type:"range",min:2,max:16,step:2,value:n,onChange:d=>h(Number(d.target.value)),className:"w-24 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Beta (channel ratio): ",i,e.jsx("input",{type:"range",min:2,max:16,step:2,value:i,onChange:d=>s(Number(d.target.value)),className:"w-24 accent-violet-500"})]})]}),e.jsxs("div",{className:"grid grid-cols-2 gap-4",children:[e.jsxs("div",{className:"rounded-lg bg-violet-100 dark:bg-violet-900/30 p-4",children:[e.jsx("p",{className:"font-bold text-violet-700 dark:text-violet-300 text-sm",children:"Slow Pathway"}),e.jsxs("p",{className:"text-xs text-gray-600 dark:text-gray-400 mt-1",children:["Frames: ",r," (every ",n,"th frame)"]}),e.jsxs("p",{className:"text-xs text-gray-600 dark:text-gray-400",children:["Channels: ",l]}),e.jsx("p",{className:"text-xs text-gray-600 dark:text-gray-400",children:"FLOPs share: ~80%"}),e.jsxs("div",{className:"flex gap-1 mt-2",children:[Array.from({length:Math.min(r,12)}).map((d,m)=>e.jsx("div",{className:"w-4 h-6 rounded bg-violet-500"},m)),r>12&&e.jsxs("span",{className:"text-xs text-violet-500",children:["+",r-12]})]})]}),e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 p-4",children:[e.jsx("p",{className:"font-bold text-violet-600 dark:text-violet-400 text-sm",children:"Fast Pathway"}),e.jsxs("p",{className:"text-xs text-gray-600 dark:text-gray-400 mt-1",children:["Frames: ",c," (all frames)"]}),e.jsxs("p",{className:"text-xs text-gray-600 dark:text-gray-400",children:["Channels: ",o]}),e.jsx("p",{className:"text-xs text-gray-600 dark:text-gray-400",children:"FLOPs share: ~20%"}),e.jsxs("div",{className:"flex gap-0.5 mt-2 flex-wrap",children:[Array.from({length:Math.min(c,24)}).map((d,m)=>e.jsx("div",{className:"w-2 h-4 rounded-sm bg-violet-400"},m)),e.jsxs("span",{className:"text-xs text-violet-400",children:["+",c-24]})]})]})]}),e.jsx("p",{className:"text-xs text-gray-500 mt-2",children:"Slow: high spatial detail, low temporal rate. Fast: fine temporal resolution, lightweight channels."})]})}function k(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"SlowFast Networks process video through two parallel pathways operating at different temporal rates, inspired by the primate visual system's distinction between sustained (parvocellular) and transient (magnocellular) processing."}),e.jsxs(x,{title:"SlowFast Architecture",children:[e.jsx("p",{children:"Two pathways process video at different temporal resolutions:"}),e.jsx(t.BlockMath,{math:"\\text{Slow}: T/\\alpha \\text{ frames}, \\quad \\text{Fast}: T \\text{ frames with } C/\\beta \\text{ channels}"}),e.jsxs("p",{className:"mt-2",children:["The Slow pathway operates at ",e.jsx(t.InlineMath,{math:"1/\\alpha"})," temporal rate (e.g., 2 fps) with full channel capacity, capturing spatial semantics. The Fast pathway runs at full frame rate with ",e.jsx(t.InlineMath,{math:"1/\\beta"})," channels (lightweight), capturing fine temporal patterns and motion."]})]}),e.jsx(y,{}),e.jsxs(_,{title:"Lateral Connections",id:"lateral-connections",children:[e.jsx("p",{children:"Information flows from Fast to Slow pathway via lateral connections at each stage. Since frame rates differ, temporal resolution must be matched:"}),e.jsx(t.BlockMath,{math:"x_\\text{slow}^{l+1} = f(x_\\text{slow}^l, \\text{Fuse}(x_\\text{fast}^l))"}),e.jsxs("p",{className:"mt-1",children:["The fusion uses either time-strided convolution (",e.jsx(t.InlineMath,{math:"5 \\times 1^2"})," with stride ",e.jsx(t.InlineMath,{math:"\\alpha"}),") or time-to-channel reshaping. This allows the Slow pathway to benefit from fine temporal information without processing all frames."]})]}),e.jsxs(f,{title:"Computational Efficiency",children:[e.jsxs("p",{children:["With typical settings (",e.jsx(t.InlineMath,{math:"\\alpha=8, \\beta=8"}),"):"]}),e.jsx(t.BlockMath,{math:"\\frac{\\text{FLOPs}_\\text{Fast}}{\\text{FLOPs}_\\text{Slow}} \\approx \\frac{\\alpha}{\\beta^2} = \\frac{8}{64} = 12.5\\%"}),e.jsx("p",{className:"mt-1",children:"The Fast pathway adds only ~20% computational overhead while processing 8x more frames. This asymmetric design is key: temporal resolution is cheap when channels are few."})]}),e.jsx(u,{title:"SlowFast Network in PyTorch",code:`import torch
import torch.nn as nn

class SlowFastBlock(nn.Module):
    def __init__(self, slow_ch=64, fast_ch=8, alpha=8):
        super().__init__()
        self.alpha = alpha
        # Slow pathway: standard 3D conv
        self.slow_conv = nn.Conv3d(slow_ch, slow_ch, (1, 3, 3), padding=(0, 1, 1))
        # Fast pathway: lightweight 3D conv
        self.fast_conv = nn.Conv3d(fast_ch, fast_ch, (3, 3, 3), padding=(1, 1, 1))
        # Lateral connection: Fast -> Slow
        self.lateral = nn.Conv3d(fast_ch, slow_ch, (alpha, 1, 1), stride=(alpha, 1, 1))
        self.slow_bn = nn.BatchNorm3d(slow_ch)
        self.fast_bn = nn.BatchNorm3d(fast_ch)

    def forward(self, x_slow, x_fast):
        # Process each pathway
        slow_out = torch.relu(self.slow_bn(self.slow_conv(x_slow)))
        fast_out = torch.relu(self.fast_bn(self.fast_conv(x_fast)))
        # Fuse fast into slow via lateral connection
        lateral = self.lateral(fast_out)
        slow_out = slow_out + lateral
        return slow_out, fast_out

# Create SlowFast inputs
alpha, beta = 8, 8
T = 64  # total frames
slow_input = torch.randn(2, 64, T // alpha, 56, 56)  # [B, C, T/alpha, H, W]
fast_input = torch.randn(2, 64 // beta, T, 56, 56)    # [B, C/beta, T, H, W]

block = SlowFastBlock(slow_ch=64, fast_ch=64 // beta, alpha=alpha)
slow_out, fast_out = block(slow_input, fast_input)
print(f"Slow output: {slow_out.shape}")  # [2, 64, 8, 56, 56]
print(f"Fast output: {fast_out.shape}")  # [2, 8, 64, 56, 56]`}),e.jsx(g,{type:"note",title:"SlowFast for Detection",children:e.jsx("p",{children:"SlowFast is widely used as the backbone for spatiotemporal action detection (e.g., AVA dataset). Features from both pathways are RoI-pooled around person bounding boxes, concatenated, and classified per-actor. This achieved the first superhuman results on several action detection benchmarks."})})]})}const $=Object.freeze(Object.defineProperty({__proto__:null,default:k},Symbol.toStringTag,{value:"Module"}));function T(){const[n,h]=p.useState(.25),i=8,s=5,a=Math.round(i*n);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Temporal Shift Visualization"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-4",children:["Shift ratio: ",(n*100).toFixed(0),"% (",a," forward + ",a," backward of ",i," channels)",e.jsx("input",{type:"range",min:.125,max:.5,step:.125,value:n,onChange:r=>h(Number(r.target.value)),className:"w-32 accent-violet-500"})]}),e.jsx("div",{className:"overflow-x-auto",children:e.jsxs("div",{className:"grid gap-1",style:{gridTemplateColumns:`60px repeat(${s}, 1fr)`},children:[e.jsx("div",{className:"text-xs text-gray-500"}),Array.from({length:s}).map((r,c)=>e.jsxs("div",{className:"text-xs text-center text-gray-500 font-semibold",children:["t=",c]},c)),Array.from({length:i}).map((r,c)=>e.jsxs(React.Fragment,{children:[e.jsxs("div",{className:"text-xs text-gray-500 flex items-center",children:["ch ",c]}),Array.from({length:s}).map((l,o)=>{let d="bg-gray-200 dark:bg-gray-700",m="";return c<a?(d="bg-violet-400 text-white",m=o>0?`t${o-1}`:"pad"):c<2*a?(d="bg-violet-600 text-white",m=o<s-1?`t${o+1}`:"pad"):(d="bg-violet-100 dark:bg-violet-900/30",m=`t${o}`),e.jsx("div",{className:`${d} rounded text-xs text-center py-1`,children:m},o)})]},c))]})}),e.jsxs("div",{className:"flex gap-4 mt-2 text-xs",children:[e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"w-3 h-3 rounded bg-violet-400"})," Forward shift"]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"w-3 h-3 rounded bg-violet-600"})," Backward shift"]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"w-3 h-3 rounded bg-violet-100 dark:bg-violet-900/30"})," No shift"]})]})]})}function N(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"The Temporal Shift Module (TSM) enables temporal reasoning with 2D CNNs by simply shifting a portion of channels along the time dimension, achieving 3D CNN-level accuracy at 2D CNN computational cost."}),e.jsxs(x,{title:"Temporal Shift Module",children:[e.jsx("p",{children:"TSM shifts a fraction of channels along the temporal axis:"}),e.jsx(t.BlockMath,{math:"X'_t[c] = \\begin{cases} X_{t-1}[c] & \\text{if } c < C/4 \\text{ (forward shift)} \\\\ X_{t+1}[c] & \\text{if } C/4 \\leq c < C/2 \\text{ (backward shift)} \\\\ X_t[c] & \\text{otherwise (no shift)} \\end{cases}"}),e.jsx("p",{className:"mt-2",children:"This zero-parameter, zero-FLOP operation enables temporal information exchange. When followed by a 2D convolution, the network effectively computes 3D features."})]}),e.jsx(T,{}),e.jsxs(_,{title:"TSM Equivalence to 3D Convolution",id:"tsm-equivalence",children:[e.jsxs("p",{children:["A temporal shift followed by a ",e.jsx(t.InlineMath,{math:"1 \\times 1"})," convolution across channels is equivalent to a ",e.jsx(t.InlineMath,{math:"3 \\times 1 \\times 1"})," depthwise-separable 3D convolution:"]}),e.jsx(t.BlockMath,{math:"\\text{Shift} + \\text{Conv2D}(1{\\times}1) \\equiv \\text{Conv3D}_\\text{depthwise}(3{\\times}1{\\times}1)"}),e.jsx("p",{className:"mt-1",children:"Combined with spatial convolutions in a ResNet block, TSM captures full spatiotemporal patterns without any 3D convolution parameters."})]}),e.jsxs(f,{title:"Residual Shift for Stability",children:[e.jsx("p",{children:"In-place shifting can harm spatial feature learning. The residual shift variant adds the shifted features instead of replacing:"}),e.jsx(t.BlockMath,{math:"X'_t = X_t + \\alpha \\cdot \\text{Shift}(X_t)"}),e.jsxs("p",{className:"mt-1",children:["With ",e.jsx(t.InlineMath,{math:"\\alpha"})," initialized small, this preserves the pre-trained 2D features while gradually learning temporal patterns. In practice, the partial shift (1/8 or 1/4 of channels) works best."]})]}),e.jsx(u,{title:"Temporal Shift Module Implementation",code:`import torch
import torch.nn as nn

class TemporalShift(nn.Module):
    def __init__(self, n_segment=8, shift_ratio=0.25):
        super().__init__()
        self.n_segment = n_segment
        self.shift_ratio = shift_ratio

    def forward(self, x):
        B, C, H, W = x.shape
        T = self.n_segment
        BT = B // T  # batch per clip
        x = x.view(BT, T, C, H, W)
        shift = int(C * self.shift_ratio)

        out = x.clone()
        # Forward shift: channels [0, shift) get frame t-1
        out[:, 1:, :shift] = x[:, :-1, :shift]
        out[:, 0, :shift] = 0  # zero-pad first frame
        # Backward shift: channels [shift, 2*shift) get frame t+1
        out[:, :-1, shift:2*shift] = x[:, 1:, shift:2*shift]
        out[:, -1, shift:2*shift] = 0  # zero-pad last frame
        # Remaining channels unchanged

        return out.view(B, C, H, W)

# Apply TSM to a ResNet backbone
class TSMResBlock(nn.Module):
    def __init__(self, channels=64, n_segment=8):
        super().__init__()
        self.tsm = TemporalShift(n_segment=n_segment)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        x = self.tsm(x)  # temporal shift (zero params!)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return torch.relu(x + identity)

T, B = 8, 2
x = torch.randn(B * T, 64, 56, 56)  # [BT, C, H, W]
block = TSMResBlock(64, n_segment=T)
out = block(x)
print(f"TSM output: {out.shape}")  # [16, 64, 56, 56]`}),e.jsx(g,{type:"note",title:"TSM for Online/Streaming Video",children:e.jsx("p",{children:"TSM's unidirectional variant (forward shift only) enables online video understanding where future frames are unavailable. This is critical for real-time applications like autonomous driving, live sports analysis, and streaming action detection. The computational overhead is essentially zero compared to frame-by-frame 2D CNN processing."})})]})}const X=Object.freeze(Object.defineProperty({__proto__:null,default:N},Symbol.toStringTag,{value:"Module"}));function w(){const[n,h]=p.useState("divided"),i={joint:{name:"Joint Space-Time",complexity:"O((T*N)^2)",memory:"Very high",quality:"Best (small scale)",desc:"Full attention over all patches across all frames simultaneously"},divided:{name:"Divided Space-Time",complexity:"O(T*N^2 + N*T^2)",memory:"Moderate",quality:"Best (scalable)",desc:"Separate spatial attention within frames, then temporal attention across frames"},sparse:{name:"Sparse (Local+Global)",complexity:"O(T*N*k)",memory:"Low",quality:"Good",desc:"Local spatial attention + sparse global temporal attention"},axial:{name:"Axial",complexity:"O(T*N*(sqrt(N)+T))",memory:"Low",quality:"Good",desc:"Factorized along height, width, and time axes independently"}},s=i[n];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Space-Time Attention Schemes"}),e.jsx("div",{className:"flex flex-wrap gap-2 mb-4",children:Object.entries(i).map(([a,r])=>e.jsx("button",{onClick:()=>h(a),className:`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${n===a?"bg-violet-600 text-white":"bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400"}`,children:r.name},a))}),e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 p-4",children:[e.jsx("p",{className:"text-sm text-gray-700 dark:text-gray-300",children:s.desc}),e.jsxs("div",{className:"grid grid-cols-3 gap-3 mt-3 text-sm",children:[e.jsxs("div",{children:[e.jsx("span",{className:"text-xs text-violet-600 dark:text-violet-400 font-semibold block",children:"Complexity"}),s.complexity]}),e.jsxs("div",{children:[e.jsx("span",{className:"text-xs text-violet-600 dark:text-violet-400 font-semibold block",children:"Memory"}),s.memory]}),e.jsxs("div",{children:[e.jsx("span",{className:"text-xs text-violet-600 dark:text-violet-400 font-semibold block",children:"Quality"}),s.quality]})]})]})]})}function M(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"TimeSformer adapts Vision Transformers to video by introducing divided space-time attention, where spatial and temporal attention are computed separately in each block. This factorization makes video Transformers scalable to long clips."}),e.jsxs(x,{title:"Divided Space-Time Attention",children:[e.jsx("p",{children:"Each TimeSformer block applies two attention operations sequentially:"}),e.jsx(t.BlockMath,{math:"z'_t = \\text{TemporalAttn}(z_t) + z_t, \\quad z''_t = \\text{SpatialAttn}(z'_t) + z'_t"}),e.jsxs("p",{className:"mt-2",children:[e.jsx("strong",{children:"Temporal attention:"})," each patch attends to the same spatial position across all ",e.jsx(t.InlineMath,{math:"T"})," frames. ",e.jsx("strong",{children:"Spatial attention:"})," patches within the same frame attend to each other. This avoids the ",e.jsx(t.InlineMath,{math:"O((TN)^2)"})," cost of joint attention."]})]}),e.jsx(w,{}),e.jsxs(_,{title:"Computational Savings",id:"timesformer-savings",children:[e.jsxs("p",{children:["For a video with ",e.jsx(t.InlineMath,{math:"T"})," frames and ",e.jsx(t.InlineMath,{math:"N"})," spatial patches per frame, divided attention reduces complexity from quadratic to:"]}),e.jsx(t.BlockMath,{math:"\\text{Joint: } O(T^2 N^2 d) \\quad \\to \\quad \\text{Divided: } O(TN^2 d + NT^2 d)"}),e.jsxs("p",{className:"mt-1",children:["With ",e.jsx(t.InlineMath,{math:"T=8, N=196"})," (ViT-B/16 on 224px), this is a",e.jsx(t.InlineMath,{math:"\\sim 6\\times"})," reduction. The savings grow linearly with clip length."]})]}),e.jsxs(f,{title:"Patch Embedding for Video",children:[e.jsx("p",{children:"TimeSformer embeds video frames independently using the same ViT patch embedding, then adds learnable temporal position embeddings:"}),e.jsx(t.BlockMath,{math:"z_{t,p} = \\text{PatchEmbed}(x_{t,p}) + e_p^\\text{spatial} + e_t^\\text{temporal}"}),e.jsxs("p",{className:"mt-1",children:["A special ",e.jsx(t.InlineMath,{math:"\\texttt{[CLS]}"})," token aggregates information across all frames for final classification."]})]}),e.jsx(u,{title:"TimeSformer Divided Attention Block",code:`import torch
import torch.nn as nn

class DividedSpaceTimeAttention(nn.Module):
    def __init__(self, dim=768, num_heads=12, num_frames=8):
        super().__init__()
        self.num_frames = num_frames
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_t = nn.LayerNorm(dim)
        self.norm_s = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, T*N, D] where T=frames, N=spatial patches
        B, TN, D = x.shape
        T = self.num_frames
        N = TN // T

        # Temporal attention: group by spatial position
        xt = x.view(B, T, N, D).permute(0, 2, 1, 3).reshape(B * N, T, D)
        xt = self.norm_t(xt)
        temporal_out, _ = self.temporal_attn(xt, xt, xt)
        temporal_out = temporal_out.reshape(B, N, T, D).permute(0, 2, 1, 3).reshape(B, TN, D)
        x = x + temporal_out

        # Spatial attention: group by frame
        xs = x.view(B, T, N, D).reshape(B * T, N, D)
        xs = self.norm_s(xs)
        spatial_out, _ = self.spatial_attn(xs, xs, xs)
        spatial_out = spatial_out.reshape(B, T, N, D).reshape(B, TN, D)
        x = x + spatial_out

        return x

# Example usage
T, N, D = 8, 196, 768  # 8 frames, 14x14 patches, ViT-Base dim
block = DividedSpaceTimeAttention(dim=D, num_heads=12, num_frames=T)
x = torch.randn(2, T * N, D)
out = block(x)
print(f"Input: {x.shape}")   # [2, 1568, 768]
print(f"Output: {out.shape}")  # [2, 1568, 768]

# Compare parameter counts
joint_params = sum(p.numel() for p in nn.MultiheadAttention(D, 12).parameters())
divided_params = sum(p.numel() for p in block.parameters())
print(f"Joint attn params: {joint_params:,}")
print(f"Divided attn params: {divided_params:,} (2x due to dual attention)")`}),e.jsx(g,{type:"note",title:"From TimeSformer to Efficient Video Transformers",children:e.jsxs("p",{children:["TimeSformer demonstrated that divided attention matches or exceeds joint attention while being far more scalable. This insight led to many efficient video Transformer designs:",e.jsx("strong",{children:"ViViT"})," (factorized encoder), ",e.jsx("strong",{children:"MViT"})," (pooling attention), and ",e.jsx("strong",{children:"VideoSwin"})," (shifted window attention in 3D). The common theme is exploiting spatiotemporal locality to avoid full attention over all tokens."]})})]})}const J=Object.freeze(Object.defineProperty({__proto__:null,default:M},Symbol.toStringTag,{value:"Module"}));function S(){const[n,h]=p.useState("factorized_enc"),i={spatio_temporal:{name:"Model 1: Spatio-temporal",tokens:"T*N",encoders:"1 Transformer",tubelets:"Yes",complexity:"O((T*N)^2)",desc:"All tokens attend to each other in a single Transformer"},factorized_enc:{name:"Model 2: Factorized Encoder",tokens:"N then T",encoders:"Spatial + Temporal",tubelets:"Yes",complexity:"O(N^2 + T^2)",desc:"Spatial encoder per frame, then temporal encoder over CLS tokens"},factorized_self:{name:"Model 3: Factorized Self-Attn",tokens:"T*N",encoders:"1 Transformer (factorized)",tubelets:"Yes",complexity:"O(T*N^2 + N*T^2)",desc:"Like TimeSformer divided attention within a single model"},factorized_dot:{name:"Model 4: Factorized Dot-Product",tokens:"T*N",encoders:"1 Transformer (factorized KV)",tubelets:"Yes",complexity:"O(T*N*(N+T))",desc:"Separate spatial/temporal heads with concatenated outputs"}},s=i[n];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"ViViT Architecture Variants"}),e.jsx("div",{className:"flex flex-wrap gap-2 mb-4",children:Object.entries(i).map(([a,r])=>e.jsx("button",{onClick:()=>h(a),className:`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${n===a?"bg-violet-600 text-white":"bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400"}`,children:r.name},a))}),e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 p-4",children:[e.jsx("p",{className:"text-sm text-gray-700 dark:text-gray-300",children:s.desc}),e.jsx("div",{className:"grid grid-cols-2 gap-3 mt-3 text-sm",children:[["Encoders",s.encoders],["Complexity",s.complexity],["Tubelet embedding",s.tubelets],["Token count",s.tokens]].map(([a,r])=>e.jsxs("div",{children:[e.jsx("span",{className:"text-xs text-violet-600 dark:text-violet-400 font-semibold block",children:a}),e.jsx("span",{className:"text-gray-600 dark:text-gray-400",children:r})]},a))})]})]})}function C(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"ViViT explores multiple strategies for applying Vision Transformers to video, introducing tubelet embeddings for spatiotemporal tokenization. VideoMAE extends masked autoencoders to video, achieving strong results with minimal labeled data."}),e.jsxs(x,{title:"Tubelet Embedding",children:[e.jsx("p",{children:"Instead of embedding 2D patches per frame, ViViT embeds 3D tubelets:"}),e.jsx(t.BlockMath,{math:"z_{t,p} = \\text{Linear}(\\text{flatten}(x[t:t{+}t_s, p_h:p_h{+}h, p_w:p_w{+}w]))"}),e.jsxs("p",{className:"mt-2",children:["A tubelet of size ",e.jsx(t.InlineMath,{math:"t_s \\times h \\times w"})," (e.g., ",e.jsx(t.InlineMath,{math:"2 \\times 16 \\times 16"}),") captures local spatiotemporal patterns in a single token, reducing the total token count by a factor of ",e.jsx(t.InlineMath,{math:"t_s"})," compared to frame-level patch embedding."]})]}),e.jsx(S,{}),e.jsxs(_,{title:"VideoMAE: Masked Video Pre-training",id:"videomae",children:[e.jsx("p",{children:"VideoMAE masks a very high ratio (90-95%) of video tokens and trains the encoder to reconstruct them:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = \\frac{1}{|\\mathcal{M}|}\\sum_{i \\in \\mathcal{M}} \\|x_i - \\hat{x}_i\\|^2"}),e.jsx("p",{className:"mt-1",children:"The extreme masking ratio works because video is highly redundant temporally. Tube masking ensures consistent masks across frames, preventing trivial solutions from temporal interpolation. After pre-training, the encoder is fine-tuned for downstream tasks."})]}),e.jsxs(f,{title:"Why 90% Masking Works for Video",children:[e.jsx("p",{children:'Adjacent video frames differ by only a few pixels. If masking is applied independently per frame, the model can "cheat" by copying from unmasked patches in nearby frames. Tube masking forces the model to reason about motion and semantics:'}),e.jsx(t.BlockMath,{math:"\\text{Tube mask: } \\mathcal{M}_t = \\mathcal{M}_0 \\quad \\forall t \\in \\{0, \\ldots, T{-}1\\}"}),e.jsx("p",{className:"mt-1",children:"The same spatial positions are masked across all frames, requiring genuine understanding to reconstruct."})]}),e.jsx(u,{title:"ViViT Factorized Encoder",code:`import torch
import torch.nn as nn

class ViViTFactorized(nn.Module):
    """ViViT Model 2: Factorized Encoder."""
    def __init__(self, num_frames=8, num_patches=196, dim=768, num_classes=400):
        super().__init__()
        self.num_frames = num_frames
        # Tubelet embedding: 2x16x16 tubelets
        self.patch_embed = nn.Conv3d(3, dim, kernel_size=(2, 16, 16), stride=(2, 16, 16))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # Spatial encoder (processes each frame independently)
        spatial_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=12, batch_first=True)
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, num_layers=6)

        # Temporal encoder (processes CLS tokens across frames)
        temporal_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=12, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=4)
        self.temporal_pos = nn.Parameter(torch.randn(1, num_frames // 2, dim))

        self.head = nn.Linear(dim, num_classes)

    def forward(self, video):  # video: [B, C, T, H, W]
        B = video.shape[0]
        # Tubelet embedding
        x = self.patch_embed(video)  # [B, D, T', H', W']
        T_out = x.shape[2]
        x = x.flatten(3).permute(0, 2, 3, 1)  # [B, T', N, D]

        # Spatial encoding per frame
        cls_tokens = []
        for t in range(T_out):
            frame_tokens = x[:, t]  # [B, N, D]
            cls = self.cls_token.expand(B, -1, -1)
            frame_tokens = torch.cat([cls, frame_tokens], dim=1) + self.pos_embed
            encoded = self.spatial_encoder(frame_tokens)
            cls_tokens.append(encoded[:, 0])  # CLS token

        # Temporal encoding over CLS tokens
        temporal_tokens = torch.stack(cls_tokens, dim=1) + self.temporal_pos
        temporal_out = self.temporal_encoder(temporal_tokens)

        # Classify from mean-pooled temporal CLS tokens
        return self.head(temporal_out.mean(dim=1))

model = ViViTFactorized(num_frames=8, num_patches=196, dim=768, num_classes=400)
video = torch.randn(2, 3, 8, 224, 224)
logits = model(video)
print(f"Classification logits: {logits.shape}")  # [2, 400]
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")`}),e.jsx(g,{type:"note",title:"Pre-training Data Efficiency",children:e.jsx("p",{children:"VideoMAE demonstrates remarkable data efficiency: pre-training on just 3,000 videos from Kinetics-400 and fine-tuning on the full set achieves competitive accuracy. This suggests that video's temporal redundancy makes self-supervised learning particularly effective, requiring far less data than image-based MAE for comparable gains."})})]})}const Y=Object.freeze(Object.defineProperty({__proto__:null,default:C},Symbol.toStringTag,{value:"Module"}));function D(){const[n,h]=p.useState("videoclip"),i={videoclip:{name:"VideoCLIP",approach:"Contrastive",vision:"TimeSformer",text:"CLIP text encoder",training:"Video-text contrastive alignment",tasks:"Retrieval, zero-shot classification"},videollava:{name:"Video-LLaVA",approach:"Generative",vision:"ViT + LanguageBind",text:"Vicuna-7B",training:"Visual instruction tuning",tasks:"Video QA, captioning, reasoning"},videochat:{name:"VideoChat",approach:"Generative",vision:"ViT-G/14",text:"StableLM / LLaMA",training:"Video instruction tuning + chat",tasks:"Conversational video understanding"},internvideo:{name:"InternVideo2",approach:"Hybrid",vision:"ViT-6B",text:"InternLM",training:"Multimodal masked + contrastive + generative",tasks:"All video tasks + generation"}},s=i[n];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Video-Language Models"}),e.jsx("div",{className:"flex flex-wrap gap-2 mb-4",children:Object.entries(i).map(([a,r])=>e.jsx("button",{onClick:()=>h(a),className:`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${n===a?"bg-violet-600 text-white":"bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400"}`,children:r.name},a))}),e.jsx("div",{className:"grid grid-cols-2 gap-3 text-sm",children:[["Approach",s.approach],["Vision encoder",s.vision],["Language model",s.text],["Training",s.training],["Tasks",s.tasks]].map(([a,r])=>e.jsxs("div",{className:`rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3 ${a==="Tasks"?"col-span-2":""}`,children:[e.jsx("p",{className:"text-xs text-violet-600 dark:text-violet-400 font-semibold",children:a}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:r})]},a))})]})}function V(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Video-language models extend multimodal LLMs to understand temporal dynamics, enabling video question answering, dense captioning, and temporal grounding through natural language interaction with video content."}),e.jsxs(x,{title:"Video-Language Alignment",children:[e.jsx("p",{children:"Contrastive video-language models learn a shared embedding space:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = -\\frac{1}{B}\\sum_{i=1}^{B}\\left[\\log \\frac{e^{\\text{sim}(v_i, t_i)/\\tau}}{\\sum_j e^{\\text{sim}(v_i, t_j)/\\tau}} + \\log \\frac{e^{\\text{sim}(t_i, v_i)/\\tau}}{\\sum_j e^{\\text{sim}(t_i, v_j)/\\tau}}\\right]"}),e.jsxs("p",{className:"mt-2",children:["Video features ",e.jsx(t.InlineMath,{math:"v_i"})," aggregate temporal information (e.g., mean pooling frame embeddings), while text features ",e.jsx(t.InlineMath,{math:"t_i"})," come from a text encoder. This enables zero-shot video retrieval and classification."]})]}),e.jsx(D,{}),e.jsxs(f,{title:"Visual Instruction Tuning for Video",children:[e.jsx("p",{children:"Video-LLaVA adapts the LLaVA framework to video by projecting frame tokens into the LLM's embedding space:"}),e.jsx(t.BlockMath,{math:"h_\\text{video} = W_\\text{proj} \\cdot \\text{Concat}[f_1, f_2, \\ldots, f_T]"}),e.jsx("p",{className:"mt-1",children:"The projected video tokens are prepended to text tokens, and the LLM generates responses autoregressively. Training uses video instruction-following datasets generated by GPT-4V or human annotators."})]}),e.jsx(u,{title:"Video-Language Model Architecture",code:`import torch
import torch.nn as nn

class SimpleVideoLLM(nn.Module):
    def __init__(self, vis_dim=768, llm_dim=4096, num_frames=8, vocab_size=32000):
        super().__init__()
        # Vision encoder (frozen ViT)
        self.vis_encoder = nn.Sequential(
            nn.Linear(3 * 224 * 224, vis_dim),  # simplified
            nn.LayerNorm(vis_dim),
        )
        # Projection: vision -> LLM space
        self.vis_proj = nn.Sequential(
            nn.Linear(vis_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
        # Temporal aggregation
        self.temporal_attn = nn.MultiheadAttention(llm_dim, 8, batch_first=True)
        # LLM decoder (simplified)
        self.llm = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=llm_dim, nhead=8, batch_first=True),
            num_layers=2
        )
        self.text_embed = nn.Embedding(vocab_size, llm_dim)
        self.output_head = nn.Linear(llm_dim, vocab_size)

    def encode_video(self, frames):
        B, T = frames.shape[:2]
        # Encode each frame
        flat = frames.view(B * T, -1)
        vis_tokens = self.vis_encoder(flat).view(B, T, -1)
        # Project to LLM space
        projected = self.vis_proj(vis_tokens)
        # Temporal aggregation
        temporal_out, _ = self.temporal_attn(projected, projected, projected)
        return temporal_out

    def forward(self, frames, text_ids):
        video_tokens = self.encode_video(frames)  # [B, T, llm_dim]
        text_tokens = self.text_embed(text_ids)    # [B, L, llm_dim]
        # Cross-attend text to video
        output = self.llm(text_tokens, video_tokens)
        return self.output_head(output)

model = SimpleVideoLLM()
frames = torch.randn(2, 8, 3, 224, 224)
question = torch.randint(0, 32000, (2, 20))
logits = model(frames, question)
print(f"Output logits: {logits.shape}")  # [2, 20, 32000]`}),e.jsx(v,{title:"Temporal Grounding Challenge",children:e.jsx("p",{children:'Current video LLMs struggle with precise temporal reasoning (e.g., "What happened at the 30-second mark?"). Uniform frame sampling loses temporal precision, and most models cannot process full-length videos. Active research areas include hierarchical video encoding, timestamp-aware tokens, and streaming video processing.'})}),e.jsx(g,{type:"note",title:"Long Video Understanding",children:e.jsxs("p",{children:["Understanding hour-long videos requires new architectures. Approaches include",e.jsx("strong",{children:"memory banks"})," that cache past context, ",e.jsx("strong",{children:"hierarchical encoding"})," at multiple temporal scales, and ",e.jsx("strong",{children:"retrieval-augmented"})," methods that select relevant clips. Models like MovieChat and LongViViT address this challenge, but efficient long-video understanding remains largely unsolved."]})})]})}const Q=Object.freeze(Object.defineProperty({__proto__:null,default:V},Symbol.toStringTag,{value:"Module"}));function B(){const[n,h]=p.useState("kinetics400"),i={kinetics400:{name:"Kinetics-400",classes:400,clips:"306K",duration:"~10s",source:"YouTube",topMethod:"VideoMAEv2 (90.0%)",challenge:"Web noise, class imbalance"},kinetics700:{name:"Kinetics-700",classes:700,clips:"650K",duration:"~10s",source:"YouTube",topMethod:"InternVideo2 (83.7%)",challenge:"Fine-grained + long-tail"},ssv2:{name:"Something-Something v2",classes:174,clips:"221K",duration:"2-6s",source:"Crowdsourced",topMethod:"VideoMAEv2 (77.0%)",challenge:"Temporal reasoning required"},moments:{name:"Moments in Time",classes:339,clips:"1M",duration:"3s",source:"Mixed",topMethod:"SlowFast + NL (34.4%)",challenge:"Multi-label, ambiguous actions"}},s=i[n];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Action Classification Benchmarks"}),e.jsx("div",{className:"flex flex-wrap gap-2 mb-4",children:Object.entries(i).map(([a,r])=>e.jsx("button",{onClick:()=>h(a),className:`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${n===a?"bg-violet-600 text-white":"bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400"}`,children:r.name},a))}),e.jsx("div",{className:"grid grid-cols-3 gap-3 text-sm",children:[["Classes",s.classes],["Clips",s.clips],["Duration",s.duration],["Source",s.source],["Top method",s.topMethod],["Challenge",s.challenge]].map(([a,r])=>e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3",children:[e.jsx("p",{className:"text-xs text-violet-600 dark:text-violet-400 font-semibold",children:a}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:r})]},a))})]})}function A(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Action classification assigns a single activity label to a trimmed video clip. While conceptually simple, it drives architecture innovation and serves as the primary benchmark for video understanding systems."}),e.jsxs(x,{title:"Video Classification Pipeline",children:[e.jsx("p",{children:"A standard video classification system predicts:"}),e.jsx(t.BlockMath,{math:"P(y | V) = \\text{softmax}(W \\cdot \\text{Pool}(f_\\theta(V)) + b)"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"V = \\{x_1, \\ldots, x_T\\}"})," is a clip of ",e.jsx(t.InlineMath,{math:"T"})," frames,",e.jsx(t.InlineMath,{math:"f_\\theta"})," is the spatiotemporal backbone (3D CNN or Video Transformer), and Pool is global average pooling over space and time."]})]}),e.jsx(B,{}),e.jsxs(_,{title:"Appearance vs Temporal Reasoning",id:"appearance-vs-temporal",children:[e.jsx("p",{children:"Datasets reveal different requirements. On Kinetics, a single frame achieves ~65% accuracy (appearance-biased). On Something-Something v2, temporal order matters:"}),e.jsx(t.BlockMath,{math:"\\text{SSv2: } \\text{Acc}_\\text{1-frame} \\approx 20\\% \\ll \\text{Acc}_\\text{video} \\approx 77\\%"}),e.jsx("p",{className:"mt-1",children:'Actions like "pushing something left to right" vs "pushing something right to left" require genuine temporal understanding, not just scene recognition. This makes SSv2 the standard test for temporal modeling.'})]}),e.jsxs(f,{title:"Multi-Crop Testing",children:[e.jsx("p",{children:"Standard practice for evaluation uses multiple temporal and spatial crops:"}),e.jsx(t.BlockMath,{math:"\\hat{y} = \\frac{1}{K}\\sum_{k=1}^{K} f_\\theta(\\text{crop}_k(V))"}),e.jsx("p",{className:"mt-1",children:"Typically 4 temporal crops (uniformly spaced) x 3 spatial crops (left, center, right) = 12 views. Scores are averaged for final prediction. This adds 12x compute at inference but consistently improves accuracy by 1-3%."})]}),e.jsx(u,{title:"Video Action Classification",code:`import torch
import torch.nn as nn

class VideoClassifier(nn.Module):
    def __init__(self, num_classes=400, backbone='r3d'):
        super().__init__()
        # 3D ResNet backbone (simplified)
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 128, (3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(128), nn.ReLU(),
            nn.Conv3d(128, 256, (3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(256), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, video):  # [B, C, T, H, W]
        features = self.features(video).flatten(1)
        return self.classifier(features)

# Training loop sketch
model = VideoClassifier(num_classes=400)
video = torch.randn(4, 3, 16, 224, 224)  # 4 clips, 16 frames
logits = model(video)
print(f"Logits: {logits.shape}")  # [4, 400]

# Multi-crop evaluation
def multi_crop_eval(model, video, num_temporal=4, num_spatial=3):
    """Average predictions over multiple crops."""
    T = video.shape[2]
    scores = []
    for t in range(num_temporal):
        start = t * (T - 16) // (num_temporal - 1) if num_temporal > 1 else 0
        clip = video[:, :, start:start+16]
        for s in range(num_spatial):
            # Spatial crops: left, center, right
            offset = s * (224 - 224) // max(num_spatial - 1, 1)
            crop = clip[:, :, :, :, offset:offset+224]
            with torch.no_grad():
                scores.append(model(crop).softmax(dim=-1))
    return torch.stack(scores).mean(dim=0)

avg_pred = multi_crop_eval(model, video)
print(f"Multi-crop prediction: {avg_pred.shape}")  # [4, 400]`}),e.jsx(g,{type:"note",title:"Foundation Models for Action Recognition",children:e.jsx("p",{children:"Large pre-trained video models (InternVideo, VideoMAEv2) now dominate benchmarks by leveraging massive pre-training followed by fine-tuning. Zero-shot classification using video-language models (VideoCLIP, X-CLIP) is also competitive, classifying actions from text descriptions without any video-specific training labels."})})]})}const Z=Object.freeze(Object.defineProperty({__proto__:null,default:A},Symbol.toStringTag,{value:"Module"}));function L(){const[n,h]=p.useState(.5),i=[{start:10,end:35,label:"Running",conf:.9},{start:45,end:70,label:"Jumping",conf:.85},{start:80,end:110,label:"Throwing",conf:.7}],s=[{start:12,end:38,label:"Running"},{start:50,end:68,label:"Jumping"},{start:85,end:120,label:"Throwing"}],a=(l,o)=>{const d=Math.max(0,Math.min(l.end,o.end)-Math.max(l.start,o.start)),m=l.end-l.start+(o.end-o.start)-d;return d/m},r=130,c=360;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Temporal Action Detection"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["IoU threshold: ",n.toFixed(2),e.jsx("input",{type:"range",min:.1,max:.9,step:.05,value:n,onChange:l=>h(Number(l.target.value)),className:"w-32 accent-violet-500"})]}),e.jsxs("svg",{width:c,height:100,className:"block",children:[e.jsx("text",{x:0,y:15,fontSize:10,fill:"#8b5cf6",children:"Ground Truth"}),s.map((l,o)=>e.jsx("rect",{x:l.start/r*c,y:20,width:(l.end-l.start)/r*c,height:14,fill:"#8b5cf6",opacity:.4,rx:3},`gt-${o}`)),e.jsx("text",{x:0,y:55,fontSize:10,fill:"#f97316",children:"Predictions"}),i.map((l,o)=>{const d=a(l,s[o]),m=d>=n;return e.jsxs("g",{children:[e.jsx("rect",{x:l.start/r*c,y:60,width:(l.end-l.start)/r*c,height:14,fill:m?"#22c55e":"#ef4444",opacity:.6,rx:3}),e.jsxs("text",{x:(l.start+l.end)/2/r*c,y:88,fontSize:8,fill:"#6b7280",textAnchor:"middle",children:["IoU=",d.toFixed(2)]})]},`pred-${o}`)})]}),e.jsxs("p",{className:"text-xs text-gray-500 mt-1",children:["Green = correct detection (IoU ≥ ",n.toFixed(2),"), Red = missed"]})]})}function I(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Temporal action detection localizes action instances in untrimmed videos, predicting both the temporal boundaries (start/end times) and the action class for each instance. This is the video equivalent of object detection in images."}),e.jsxs(x,{title:"Temporal Action Detection Task",children:[e.jsx("p",{children:"Given an untrimmed video, predict a set of action instances:"}),e.jsx(t.BlockMath,{math:"\\{(t_s^i, t_e^i, c^i, p^i)\\}_{i=1}^{N}"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"t_s^i, t_e^i"})," are start/end times, ",e.jsx(t.InlineMath,{math:"c^i"})," is the action class, and ",e.jsx(t.InlineMath,{math:"p^i"})," is a confidence score. Evaluation uses mAP at various temporal IoU thresholds (0.3, 0.5, 0.7)."]})]}),e.jsx(L,{}),e.jsxs(_,{title:"Anchor-Based Detection",id:"anchor-based-detection",children:[e.jsx("p",{children:"BMN (Boundary Matching Network) generates proposals by predicting start/end probability and proposal confidence for all candidate pairs:"}),e.jsx(t.BlockMath,{math:"\\text{BM}(t_s, t_e) = \\text{MLP}(\\text{Pool}(f[t_s : t_e]))"}),e.jsxs("p",{className:"mt-1",children:["Start and end boundaries are predicted independently, then combined. The boundary-matching confidence map ",e.jsx(t.InlineMath,{math:"\\text{BM} \\in \\mathbb{R}^{T \\times T}"}),"scores all ",e.jsx(t.InlineMath,{math:"(t_s, t_e)"})," pairs, filtered by NMS."]})]}),e.jsxs(f,{title:"ActionFormer: Anchor-Free Detection",children:[e.jsx("p",{children:"ActionFormer uses a Transformer encoder with multi-scale temporal feature pyramids:"}),e.jsx(t.BlockMath,{math:"(d_s^t, d_e^t, c_t) = \\text{Head}(f_l^t)"}),e.jsxs("p",{className:"mt-1",children:["At each temporal position ",e.jsx(t.InlineMath,{math:"t"})," and scale ",e.jsx(t.InlineMath,{math:"l"}),", it predicts distances to boundaries ",e.jsx(t.InlineMath,{math:"d_s, d_e"})," and class probabilities. This anchor-free approach achieved state-of-the-art results on ActivityNet and THUMOS."]})]}),e.jsx(u,{title:"Temporal Action Detection Pipeline",code:`import torch
import torch.nn as nn

class TemporalActionDetector(nn.Module):
    def __init__(self, feat_dim=2048, num_classes=20, num_scales=5):
        super().__init__()
        # Multi-scale temporal feature pyramid
        self.pyramids = nn.ModuleList()
        for s in range(num_scales):
            self.pyramids.append(nn.Sequential(
                nn.Conv1d(feat_dim if s == 0 else feat_dim // 2,
                          feat_dim // 2, 3, stride=2, padding=1),
                nn.ReLU(),
            ))
        # Detection heads per scale
        self.cls_head = nn.Conv1d(feat_dim // 2, num_classes, 1)
        self.reg_head = nn.Conv1d(feat_dim // 2, 2, 1)  # start/end offsets

    def forward(self, features):  # [B, D, T]
        detections = []
        x = features
        for i, pyramid in enumerate(self.pyramids):
            x = pyramid(x)
            cls = self.cls_head(x)       # [B, C, T_i]
            reg = self.reg_head(x).relu()  # [B, 2, T_i]
            detections.append((cls, reg, x.shape[-1]))
        return detections

# Feature extraction (from pre-trained backbone)
feat_dim = 2048
T = 256  # temporal positions
features = torch.randn(2, feat_dim, T)

detector = TemporalActionDetector(feat_dim=feat_dim, num_classes=20)
outputs = detector(features)

for i, (cls, reg, t_len) in enumerate(outputs):
    print(f"Scale {i}: cls={cls.shape}, reg={reg.shape}, T={t_len}")

# Decode predictions from scale 0
cls_probs = outputs[0][0].softmax(dim=1)  # [B, C, T]
offsets = outputs[0][1]  # [B, 2, T]
print(f"\\nPredicted classes: {cls_probs.shape}")
print(f"Boundary offsets: {offsets.shape}")`}),e.jsx(g,{type:"note",title:"From Detection to Dense Captioning",children:e.jsx("p",{children:"Dense video captioning extends temporal detection by generating natural language descriptions for each detected segment. Models like Vid2Seq unify temporal localization and captioning in a single sequence-to-sequence framework, predicting special time tokens interleaved with text tokens. This connects action detection with video-language understanding."})})]})}const ee=Object.freeze(Object.defineProperty({__proto__:null,default:I},Symbol.toStringTag,{value:"Module"}));function z(){const[n,h]=p.useState(0),i=200,s=250,a=[[100,30],[100,70],[70,70],[130,70],[50,120],[150,120],[40,160],[160,160],[85,140],[115,140],[80,190],[120,190],[75,230],[125,230]],r=[[0,1],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7],[1,8],[1,9],[8,10],[9,11],[10,12],[11,13],[8,9]],c=Math.sin(n*.3)*15,l=a.map(([o,d],m)=>m===6?[o+c*2,d-Math.abs(c)]:m===7?[o-c*2,d-Math.abs(c)]:m===4?[o+c,d]:m===5?[o-c,d]:[o,d]);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Skeleton Joint Visualization"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Frame: ",n,e.jsx("input",{type:"range",min:0,max:20,step:1,value:n,onChange:o=>h(Number(o.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("svg",{width:i,height:s,className:"mx-auto block",children:[r.map(([o,d],m)=>e.jsx("line",{x1:l[o][0],y1:l[o][1],x2:l[d][0],y2:l[d][1],stroke:"#8b5cf6",strokeWidth:2},m)),l.map(([o,d],m)=>e.jsx("circle",{cx:o,cy:d,r:4,fill:"#7c3aed",stroke:"white",strokeWidth:1.5},m))]}),e.jsxs("p",{className:"text-xs text-gray-500 text-center mt-1",children:["14 joints, ",r.length," bones — Input tensor: [N, C, T, V, M] = [batch, channels, frames, joints, persons]"]})]})}function F(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Skeleton-based action recognition uses body joint coordinates as input, offering robustness to appearance variations, viewpoint changes, and background clutter. Graph Convolutional Networks model the body as a graph for powerful spatial-temporal reasoning."}),e.jsxs(x,{title:"Skeleton Graph Representation",children:[e.jsx("p",{children:"The human skeleton is represented as a spatial-temporal graph:"}),e.jsx(t.BlockMath,{math:"G = (V, E_s \\cup E_t), \\quad V = \\{v_{t,j} | t \\in T, j \\in J\\}"}),e.jsxs("p",{className:"mt-2",children:["Vertices are joint positions over time. Spatial edges ",e.jsx(t.InlineMath,{math:"E_s"})," connect physically linked joints (bones), while temporal edges ",e.jsx(t.InlineMath,{math:"E_t"})," connect the same joint across consecutive frames. Input features are typically",e.jsx(t.InlineMath,{math:"(x, y, z)"})," coordinates or ",e.jsx(t.InlineMath,{math:"(x, y, \\text{confidence})"}),"."]})]}),e.jsx(z,{}),e.jsxs(_,{title:"Spatial-Temporal Graph Convolution (ST-GCN)",id:"stgcn",children:[e.jsx("p",{children:"ST-GCN applies graph convolution on the skeleton graph with a learnable adjacency matrix:"}),e.jsx(t.BlockMath,{math:"f_\\text{out} = \\sum_{k=0}^{K-1} \\hat{A}_k X W_k, \\quad \\hat{A}_k = D_k^{-1/2}(A_k + I)D_k^{-1/2}"}),e.jsxs("p",{className:"mt-1",children:["The adjacency matrix ",e.jsx(t.InlineMath,{math:"A"})," is partitioned into ",e.jsx(t.InlineMath,{math:"K"})," subsets based on distance from the root joint (centripetal, root, centrifugal). This spatial convolution is combined with temporal convolution for full spatiotemporal modeling:"]}),e.jsx(t.BlockMath,{math:"f = \\text{TemporalConv}(\\text{GraphConv}(X, A))"})]}),e.jsxs(f,{title:"Adaptive Graph Structures",children:[e.jsx("p",{children:"Modern approaches learn the graph topology rather than using the physical skeleton:"}),e.jsx(t.BlockMath,{math:"A_\\text{adaptive} = A_\\text{physical} + B_\\text{learnable} + C(X)"}),e.jsxs("p",{className:"mt-1",children:["where ",e.jsx(t.InlineMath,{math:"B"})," is a learnable residual graph and ",e.jsx(t.InlineMath,{math:"C(X)"})," is a data-dependent graph computed via dot-product attention between joint features. This allows discovering non-physical but semantically meaningful connections (e.g., left hand to right hand for clapping)."]})]}),e.jsx(u,{title:"Skeleton-Based Action Recognition with GCN",code:`import torch
import torch.nn as nn

class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_joints=25, A=None, t_kernel=9):
        super().__init__()
        self.num_joints = num_joints
        # Learnable adjacency matrix
        self.A = nn.Parameter(torch.eye(num_joints).unsqueeze(0))
        # Spatial graph convolution
        self.gcn = nn.Conv2d(in_ch, out_ch, 1)
        self.bn_s = nn.BatchNorm2d(out_ch)
        # Temporal convolution
        self.tcn = nn.Conv2d(out_ch, out_ch, (t_kernel, 1), padding=(t_kernel//2, 0))
        self.bn_t = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        # x: [B, C, T, V] where V=num_joints
        B, C, T, V = x.shape
        # Graph convolution: multiply by adjacency
        A = self.A.softmax(dim=-1)
        x_g = torch.einsum('bctv,kvw->bctw', x, A)
        x_g = torch.relu(self.bn_s(self.gcn(x_g)))
        # Temporal convolution
        x_t = torch.relu(self.bn_t(self.tcn(x_g)))
        return x_t

class SkeletonClassifier(nn.Module):
    def __init__(self, num_joints=25, num_classes=60):
        super().__init__()
        self.blocks = nn.Sequential(
            STGCNBlock(3, 64, num_joints),
            STGCNBlock(64, 128, num_joints),
            STGCNBlock(128, 256, num_joints),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

# NTU-RGBD skeleton input: [B, C, T, V]
B, C, T, V = 4, 3, 64, 25  # 3D coords, 64 frames, 25 joints
skeleton = torch.randn(B, C, T, V)
model = SkeletonClassifier(num_joints=V, num_classes=60)
logits = model(skeleton)
print(f"Skeleton input: {skeleton.shape}")
print(f"Action logits: {logits.shape}")  # [4, 60]`}),e.jsx(g,{type:"note",title:"Multi-Modal Fusion",children:e.jsx("p",{children:"Combining skeleton data with RGB features improves robustness: skeletons provide structural motion information while RGB captures appearance and context. Recent methods like PoseConv3D create 3D heatmap volumes from skeletons, enabling processing with standard 3D CNNs and easy fusion with RGB features at the backbone level."})})]})}const te=Object.freeze(Object.defineProperty({__proto__:null,default:F},Symbol.toStringTag,{value:"Module"}));function P(){const[n,h]=p.useState(5),[i,s]=p.useState("deterministic"),a={deterministic:{name:"Deterministic",blur:n*3,desc:"Single prediction, increasingly blurry"},stochastic:{name:"Stochastic (VAE)",blur:Math.min(n,3),desc:"Samples diverse futures, sharper"},diffusion:{name:"Diffusion",blur:Math.min(n,2),desc:"High quality, temporally consistent"}},r=a[i];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Video Prediction Quality vs Horizon"}),e.jsxs("div",{className:"flex flex-wrap gap-4 mb-4",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Prediction horizon: ",n," frames",e.jsx("input",{type:"range",min:1,max:20,step:1,value:n,onChange:c=>h(Number(c.target.value)),className:"w-28 accent-violet-500"})]}),e.jsx("div",{className:"flex gap-2",children:Object.entries(a).map(([c,l])=>e.jsx("button",{onClick:()=>s(c),className:`rounded-lg px-3 py-1 text-xs font-medium ${i===c?"bg-violet-600 text-white":"bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"}`,children:l.name},c))})]}),e.jsx("div",{className:"flex gap-1 items-end",children:Array.from({length:10}).map((c,l)=>{const o=l<5,d=l-5,m=o?100:Math.max(10,100-r.blur*(d+1)*2);return e.jsxs("div",{className:"flex-1 flex flex-col items-center",children:[e.jsx("div",{className:`w-full rounded-t ${o?"bg-violet-500":m>60?"bg-violet-400":m>30?"bg-violet-300":"bg-violet-200"}`,style:{height:`${m}px`,opacity:o?1:.5+m/200}}),e.jsx("span",{className:"text-xs text-gray-500 mt-1",children:o?`c${l}`:`p${d}`})]},l)})}),e.jsxs("p",{className:"text-xs text-gray-500 mt-2",children:[r.desc,". Estimated quality at horizon ",n,": ",Math.max(10,100-r.blur*n*2).toFixed(0),"%"]})]})}function q(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Video prediction generates future frames given past observations, testing a model's understanding of scene dynamics, physics, and object permanence. It serves as both a self-supervised learning task and a core component of world models."}),e.jsxs(x,{title:"Video Prediction Problem",children:[e.jsxs("p",{children:["Given context frames ",e.jsx(t.InlineMath,{math:"x_{1:T}"}),", predict future frames:"]}),e.jsx(t.BlockMath,{math:"P(x_{T+1:T+K} | x_{1:T})"}),e.jsx("p",{className:"mt-2",children:"The future is inherently uncertain: a ball at the edge of a table might fall or stay. Deterministic models produce blurry averages of possible futures, motivating stochastic approaches that model the full distribution of outcomes."})]}),e.jsx(P,{}),e.jsxs(_,{title:"Stochastic Video Prediction",id:"stochastic-prediction",children:[e.jsx("p",{children:"SVG (Stochastic Video Generation) uses a learned prior to sample diverse futures:"}),e.jsx(t.BlockMath,{math:"x_{t+1} = g_\\theta(x_t, z_t), \\quad z_t \\sim q_\\phi(z_t | x_{1:t+1}) \\text{ (training)}"}),e.jsxs("p",{className:"mt-1",children:["At test time, ",e.jsx(t.InlineMath,{math:"z_t \\sim p_\\psi(z_t | x_{1:t})"})," is sampled from a learned prior. The KL divergence between posterior and prior ensures the prior can generate meaningful latent codes:"]}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = \\|x_{t+1} - \\hat{x}_{t+1}\\|^2 + \\beta \\, D_\\text{KL}(q_\\phi \\| p_\\psi)"})]}),e.jsxs(f,{title:"Evaluation Metrics",children:[e.jsx("p",{children:"Video prediction quality is measured by multiple complementary metrics:"}),e.jsxs("ul",{className:"list-disc pl-5 mt-2 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"PSNR/SSIM:"})," Pixel-level quality (penalizes blur)"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"FVD (Frechet Video Distance):"})," Distribution-level quality using I3D features"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"LPIPS:"})," Perceptual quality from deep feature distances"]})]}),e.jsx(t.BlockMath,{math:"\\text{FVD} = \\|\\mu_r - \\mu_g\\|^2 + \\text{Tr}(\\Sigma_r + \\Sigma_g - 2(\\Sigma_r \\Sigma_g)^{1/2})"})]}),e.jsx(u,{title:"Simple Video Prediction Model",code:`import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=pad)
        self.hidden_ch = hidden_ch

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new

class VideoPredictionModel(nn.Module):
    def __init__(self, channels=3, hidden=64):
        super().__init__()
        self.encoder = nn.Conv2d(channels, hidden, 3, stride=2, padding=1)
        self.lstm = ConvLSTMCell(hidden, hidden)
        self.decoder = nn.ConvTranspose2d(hidden, channels, 4, stride=2, padding=1)

    def forward(self, context, n_future=5):
        B, T, C, H, W = context.shape
        h = torch.zeros(B, 64, H // 2, W // 2, device=context.device)
        c = torch.zeros_like(h)

        # Encode context frames
        for t in range(T):
            enc = torch.relu(self.encoder(context[:, t]))
            h, c = self.lstm(enc, (h, c))

        # Predict future frames
        predictions = []
        for _ in range(n_future):
            h, c = self.lstm(h, (h, c))
            pred = torch.sigmoid(self.decoder(h))
            predictions.append(pred)

        return torch.stack(predictions, dim=1)

model = VideoPredictionModel()
context = torch.rand(2, 5, 3, 64, 64)  # 5 context frames
future = model(context, n_future=10)
print(f"Context: {context.shape}")     # [2, 5, 3, 64, 64]
print(f"Predicted: {future.shape}")    # [2, 10, 3, 64, 64]`}),e.jsx(g,{type:"note",title:"World Models and Video Prediction",children:e.jsxs("p",{children:["Video prediction is a core capability of ",e.jsx("strong",{children:"world models"})," for autonomous agents. Models like GAIA-1 (Wayve) and DreamerV3 learn to predict future observations from actions, enabling planning in imagination. The transition from pixel prediction to latent-space prediction dramatically improves both quality and computational efficiency."]})})]})}const ae=Object.freeze(Object.defineProperty({__proto__:null,default:q},Symbol.toStringTag,{value:"Module"}));function E(){const[n,h]=p.useState("sora"),i={sora:{name:"Sora",arch:"DiT (Diffusion Transformer)",resolution:"Up to 1080p",duration:"Up to 60s",conditioning:"Text + image",keyInnovation:"Spacetime patches, variable resolution/duration"},runway:{name:"Gen-3 Alpha",arch:"DiT with temporal layers",resolution:"1080p",duration:"~10s",conditioning:"Text + image + video",keyInnovation:"Fine-grained temporal control and camera motion"},stablevideo:{name:"Stable Video Diffusion",arch:"UNet with temporal convs",resolution:"576x1024",duration:"~4s (25 frames)",conditioning:"Image (img2vid)",keyInnovation:"Curated pre-training, motion bucket conditioning"},cogvideo:{name:"CogVideoX",arch:"3D-VAE + DiT",resolution:"720p",duration:"~6s",conditioning:"Text",keyInnovation:"Expert Transformer with adaptive LayerNorm"}},s=i[n];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Video Diffusion Models"}),e.jsx("div",{className:"flex flex-wrap gap-2 mb-4",children:Object.entries(i).map(([a,r])=>e.jsx("button",{onClick:()=>h(a),className:`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${n===a?"bg-violet-600 text-white":"bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400"}`,children:r.name},a))}),e.jsx("div",{className:"grid grid-cols-3 gap-3 text-sm",children:[["Architecture",s.arch],["Resolution",s.resolution],["Duration",s.duration],["Conditioning",s.conditioning],["Key innovation",s.keyInnovation]].map(([a,r])=>e.jsxs("div",{className:`rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3 ${a==="Key innovation"?"col-span-2":""}`,children:[e.jsx("p",{className:"text-xs text-violet-600 dark:text-violet-400 font-semibold",children:a}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:r})]},a))})]})}function O(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Video diffusion models generate photorealistic video from text prompts by extending image diffusion to the temporal dimension. Systems like Sora demonstrate emergent understanding of 3D geometry, physics, and object permanence from pure video generation training."}),e.jsxs(x,{title:"Video Latent Diffusion",children:[e.jsx("p",{children:"Video diffusion operates on spatiotemporal latent representations:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = \\mathbb{E}_{z_0, \\epsilon, t}\\left[\\|\\epsilon - \\epsilon_\\theta(z_t, t, c)\\|^2\\right], \\quad z_0 = \\text{Enc}_\\text{3D}(V)"}),e.jsxs("p",{className:"mt-2",children:["A 3D VAE encodes the video ",e.jsx(t.InlineMath,{math:"V \\in \\mathbb{R}^{T \\times H \\times W \\times 3}"})," into a compressed latent ",e.jsx(t.InlineMath,{math:"z_0 \\in \\mathbb{R}^{T' \\times H' \\times W' \\times C}"}),". The denoising network ",e.jsx(t.InlineMath,{math:"\\epsilon_\\theta"})," is typically a DiT (Diffusion Transformer) processing spacetime patches."]})]}),e.jsx(E,{}),e.jsxs(f,{title:"Spacetime Patches (Sora)",children:[e.jsx("p",{children:"Sora tokenizes video into spacetime patches, treating video like a language of visual tokens:"}),e.jsx(t.BlockMath,{math:"z_\\text{patch} \\in \\mathbb{R}^{(T/t_p) \\times (H/h_p) \\times (W/w_p) \\times D}"}),e.jsxs("p",{className:"mt-1",children:["With patch size ",e.jsx(t.InlineMath,{math:"(t_p, h_p, w_p) = (1, 2, 2)"})," in latent space, a 16-frame 512x512 video becomes ~16K tokens. The DiT processes these with full attention, learning spatiotemporal relationships. Variable-resolution training enables generating at any aspect ratio and duration."]})]}),e.jsx(u,{title:"Simplified Video Diffusion Architecture",code:`import torch
import torch.nn as nn

class VideoDenoiser(nn.Module):
    """Simplified DiT-style video denoiser."""
    def __init__(self, latent_ch=4, dim=512, num_heads=8, depth=4):
        super().__init__()
        # Spacetime patch embedding
        self.patch_embed = nn.Conv3d(latent_ch, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # Time embedding
        self.time_mlp = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        # Text conditioning projection
        self.text_proj = nn.Linear(768, dim)
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
            for _ in range(depth)
        ])
        # Output projection
        self.output = nn.Linear(dim, latent_ch * 4)  # 2x2 patch

    def forward(self, z_t, t, text_emb):
        B = z_t.shape[0]
        # Patchify
        patches = self.patch_embed(z_t)  # [B, D, T, H/2, W/2]
        T_p, H_p, W_p = patches.shape[2:]
        x = patches.flatten(2).permute(0, 2, 1)  # [B, N, D]
        # Add conditioning
        t_emb = self.time_mlp(t.view(B, 1))  # [B, D]
        c_emb = self.text_proj(text_emb)      # [B, D]
        x = x + (t_emb + c_emb).unsqueeze(1)
        # Transformer
        for block in self.blocks:
            x = block(x)
        # Unpatchify
        noise_pred = self.output(x)
        noise_pred = noise_pred.view(B, T_p, H_p, W_p, 4, 2, 2)
        noise_pred = noise_pred.permute(0, 4, 1, 2, 5, 3, 6)
        return noise_pred.reshape(B, 4, T_p, H_p * 2, W_p * 2)

model = VideoDenoiser()
z_t = torch.randn(2, 4, 16, 32, 32)   # noisy latent video
t = torch.rand(2)                       # diffusion timestep
text_emb = torch.randn(2, 768)          # T5 text embedding
noise_pred = model(z_t, t, text_emb)
print(f"Noisy input: {z_t.shape}")
print(f"Noise prediction: {noise_pred.shape}")  # [2, 4, 16, 32, 32]`}),e.jsx(v,{title:"Computational Requirements",children:e.jsx("p",{children:"Training video diffusion models requires enormous compute: Sora is estimated at thousands of GPU-months. A single 60-second generation may take minutes on high-end hardware. Inference optimization through distillation, consistency models, and progressive generation is an active research area."})}),e.jsx(g,{type:"note",title:"Video as World Simulation",children:e.jsx("p",{children:`Sora's technical report describes the model as a "world simulator" that learns physical rules from video data. Emergent capabilities include consistent 3D geometry across camera movements, object persistence through occlusion, and realistic physical interactions. This suggests that scaling video generation may be a path toward understanding the physical world.`})})]})}const se=Object.freeze(Object.defineProperty({__proto__:null,default:O},Symbol.toStringTag,{value:"Module"}));function R(){const[n,h]=p.useState("instruct"),i={instruct:{name:"Text-Guided Editing",method:"Instruction-following with diffusion",temporalConsistency:"High (shared attention)",input:"Video + edit instruction",examples:"InstructVid2Vid, MagicEdit",useCase:'"Make the dog golden" on a video of a dog'},inpaint:{name:"Video Inpainting",method:"Masked diffusion generation",temporalConsistency:"Medium (propagation needed)",input:"Video + spatiotemporal mask",examples:"ProPainter, E2FGVI",useCase:"Remove an object from all frames seamlessly"},style:{name:"Style Transfer",method:"Latent space manipulation",temporalConsistency:"Medium (flickering risk)",input:"Video + style reference",examples:"Rerender-A-Video, CoDeF",useCase:"Apply Van Gogh style to a home video"},motion:{name:"Motion Transfer",method:"Pose/flow-guided generation",temporalConsistency:"High (motion prior)",input:"Source video + target motion",examples:"DreamPose, MotionCtrl",useCase:"Transfer dance moves to a different person"}},s=i[n];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Video Editing Techniques"}),e.jsx("div",{className:"flex flex-wrap gap-2 mb-4",children:Object.entries(i).map(([a,r])=>e.jsx("button",{onClick:()=>h(a),className:`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${n===a?"bg-violet-600 text-white":"bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400"}`,children:r.name},a))}),e.jsx("div",{className:"grid grid-cols-2 gap-3 text-sm",children:[["Method",s.method],["Temporal consistency",s.temporalConsistency],["Input",s.input],["Example models",s.examples],["Use case",s.useCase]].map(([a,r])=>e.jsxs("div",{className:`rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3 ${a==="Use case"?"col-span-2":""}`,children:[e.jsx("p",{className:"text-xs text-violet-600 dark:text-violet-400 font-semibold",children:a}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:r})]},a))})]})}function W(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Neural video editing enables modifying video content through text instructions, style transfer, and inpainting while maintaining temporal consistency. The key challenge is ensuring edits are coherent across frames without flickering or artifacts."}),e.jsxs(x,{title:"DDIM Inversion for Video Editing",children:[e.jsx("p",{children:"Video editing via diffusion models typically uses DDIM inversion to obtain an editable latent:"}),e.jsx(t.BlockMath,{math:"z_T = \\text{DDIM-Inv}(z_0, \\{\\epsilon_\\theta(\\cdot, t, c_\\text{src})\\}_{t=1}^{T})"}),e.jsxs("p",{className:"mt-2",children:["The source video is inverted to noise ",e.jsx(t.InlineMath,{math:"z_T"}),", then denoised with the edited prompt ",e.jsx(t.InlineMath,{math:"c_\\text{edit}"}),". Shared self-attention keys and values from the source denoising process preserve structure:"]}),e.jsx(t.BlockMath,{math:"\\text{Attn}(Q_\\text{edit}, K_\\text{src}, V_\\text{src})"})]}),e.jsx(R,{}),e.jsxs(f,{title:"Temporal Consistency via Cross-Frame Attention",children:[e.jsxs("p",{children:["To prevent flickering, video editing models replace per-frame self-attention with cross-frame attention. For frame ",e.jsx(t.InlineMath,{math:"t"}),", keys and values come from a reference frame (typically the first or previous):"]}),e.jsx(t.BlockMath,{math:"\\text{Attn}_t = \\text{softmax}\\!\\left(\\frac{Q_t K_\\text{ref}^\\top}{\\sqrt{d}}\\right) V_\\text{ref}"}),e.jsx("p",{className:"mt-1",children:"This forces all frames to attend to the same reference, propagating the edit consistently. Extended attention (attending to multiple reference frames) further improves long-video consistency."})]}),e.jsx(u,{title:"Video Editing with Cross-Frame Attention",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossFrameAttention(nn.Module):
    """Self-attention that uses keys/values from a reference frame."""
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, ref_frame=None):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        if ref_frame is not None:
            # Use keys/values from reference frame
            ref_qkv = self.qkv(ref_frame).reshape(B, N, 3, self.num_heads, self.head_dim)
            _, k, v = ref_qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)

class VideoEditor(nn.Module):
    def __init__(self, dim=512, num_frames=16):
        super().__init__()
        self.num_frames = num_frames
        self.attn = CrossFrameAttention(dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def edit_video(self, source_features, edit_features):
        """Apply edit while maintaining temporal consistency."""
        B_total, N, D = edit_features.shape
        B = B_total // self.num_frames

        # Use first frame as reference for cross-frame attention
        ref = source_features[:B * 1].repeat(self.num_frames, 1, 1)

        # Cross-frame attention (edit queries, source keys/values)
        x = self.norm(edit_features)
        x = edit_features + self.attn(x, ref_frame=ref)
        x = x + self.mlp(self.norm(x))
        return x

editor = VideoEditor(dim=512, num_frames=16)
source = torch.randn(32, 196, 512)  # 16 frames * 2 batch, 14x14 patches
edited = torch.randn(32, 196, 512)  # edited version (may flicker)
consistent = editor.edit_video(source, edited)
print(f"Source features: {source.shape}")
print(f"Consistent edit: {consistent.shape}")`}),e.jsx(v,{title:"Ethical Implications",children:e.jsx("p",{children:"Video editing and deepfake technology raise serious ethical concerns around misinformation, non-consensual content, and identity theft. Research in provenance tracking, watermarking (C2PA standard), and deepfake detection is critical. Responsible deployment requires robust detection tools alongside generation capabilities."})}),e.jsx(g,{type:"note",title:"Content-Deformation Fields (CoDeF)",children:e.jsx("p",{children:"CoDeF represents a video as a canonical content field plus a temporal deformation field, enabling editing the canonical image and propagating changes consistently across all frames. This neural representation approach avoids per-frame processing entirely, achieving superior temporal consistency for style transfer and object manipulation."})})]})}const ne=Object.freeze(Object.defineProperty({__proto__:null,default:W},Symbol.toStringTag,{value:"Module"}));export{$ as a,X as b,J as c,Y as d,Q as e,Z as f,ee as g,te as h,ae as i,se as j,ne as k,U as s};

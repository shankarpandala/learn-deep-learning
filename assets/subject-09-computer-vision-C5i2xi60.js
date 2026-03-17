import{j as e,r as x}from"./vendor-DpISuAX6.js";import{r as t}from"./vendor-katex-CbWCYdth.js";import{D as g,E as f,P as u,W as _,N as y,T as j}from"./subject-01-foundations-D0A1VJsr.js";function k(){const[n,m]=x.useState("original"),l={original:{label:"Original",tx:0,ty:0,scale:1,rotate:0,opacity:1},flip:{label:"Horizontal Flip",tx:0,ty:0,scale:-1,rotate:0,opacity:1},rotate:{label:"Rotation (+15)",tx:0,ty:0,scale:1,rotate:15,opacity:1},crop:{label:"Random Crop",tx:10,ty:10,scale:1.3,rotate:0,opacity:1},color:{label:"Color Jitter",tx:0,ty:0,scale:1,rotate:0,opacity:.7}},c=l[n];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Data Augmentation Preview"}),e.jsx("div",{className:"flex flex-wrap gap-2 mb-4",children:Object.entries(l).map(([a,r])=>e.jsx("button",{onClick:()=>m(a),className:`px-3 py-1 rounded text-sm ${n===a?"bg-violet-500 text-white":"bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300"}`,children:r.label},a))}),e.jsx("svg",{width:200,height:200,className:"mx-auto block border border-gray-200 dark:border-gray-700 rounded",children:e.jsxs("g",{transform:`translate(100,100) rotate(${c.rotate}) scale(${c.scale}) translate(${c.tx},${c.ty})`,opacity:c.opacity,children:[e.jsx("rect",{x:-40,y:-40,width:80,height:80,fill:"#8b5cf6",rx:4}),e.jsx("circle",{cx:-10,cy:-10,r:8,fill:"#fbbf24"}),e.jsx("polygon",{points:"0,10 20,-15 40,10",fill:"#34d399"})]})}),e.jsxs("p",{className:"mt-2 text-center text-sm text-gray-500 dark:text-gray-400",children:["Transform: ",c.label]})]})}function v(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"A robust image classification pipeline covers data loading, augmentation, model training, and evaluation. Each stage has a significant impact on final accuracy and generalization."}),e.jsxs(g,{title:"Standard Training Pipeline",children:[e.jsx("p",{children:"The pipeline consists of sequential stages:"}),e.jsx(t.BlockMath,{math:"\\text{Data} \\xrightarrow{\\text{augment}} \\text{Batch} \\xrightarrow{\\text{forward}} \\hat{y} \\xrightarrow{\\mathcal{L}} \\text{loss} \\xrightarrow{\\nabla} \\text{update}"}),e.jsxs("p",{className:"mt-2",children:["The cross-entropy loss for ",e.jsx(t.InlineMath,{math:"C"})," classes is:"]}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = -\\sum_{c=1}^{C} y_c \\log(\\hat{y}_c)"})]}),e.jsx(k,{}),e.jsxs(f,{title:"Common Augmentation Strategy",children:[e.jsx("p",{children:"For ImageNet-scale training, a typical augmentation stack includes:"}),e.jsxs("ul",{className:"list-disc ml-5 mt-2 space-y-1",children:[e.jsxs("li",{children:["Random resized crop to ",e.jsx(t.InlineMath,{math:"224 \\times 224"})]}),e.jsxs("li",{children:["Horizontal flip with ",e.jsx(t.InlineMath,{math:"p = 0.5"})]}),e.jsx("li",{children:"Color jitter (brightness, contrast, saturation)"}),e.jsx("li",{children:"RandAugment or AutoAugment policies"}),e.jsx("li",{children:"Mixup / CutMix regularization"})]})]}),e.jsx(u,{title:"PyTorch Training Pipeline",code:`import torch
import torch.nn as nn
from torchvision import transforms, datasets, models

# Data augmentation pipeline
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_set = datasets.ImageFolder('data/train', train_transform)
loader = torch.utils.data.DataLoader(
    train_set, batch_size=64, shuffle=True, num_workers=4)

model = models.resnet50(pretrained=False, num_classes=10)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Training loop
for epoch in range(100):
    model.train()
    for images, labels in loader:
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: loss={loss.item():.4f}")`}),e.jsxs(_,{title:"Label Smoothing",children:[e.jsxs("p",{children:["Hard one-hot labels can cause overconfident predictions. Label smoothing replaces the target ",e.jsx(t.InlineMath,{math:"y_c"})," with:"]}),e.jsx(t.BlockMath,{math:"y_c' = (1 - \\epsilon) \\cdot y_c + \\frac{\\epsilon}{C}"}),e.jsxs("p",{className:"mt-1",children:["Typical ",e.jsx(t.InlineMath,{math:"\\epsilon = 0.1"}),". This improves calibration and generalization."]})]}),e.jsxs(y,{type:"note",title:"Learning Rate Scheduling",children:[e.jsxs("p",{children:["Cosine annealing is the most popular schedule for image classification. It decays the learning rate from ",e.jsx(t.InlineMath,{math:"\\eta_{\\max}"})," to ",e.jsx(t.InlineMath,{math:"\\eta_{\\min}"})," following:"]}),e.jsx(t.BlockMath,{math:"\\eta_t = \\eta_{\\min} + \\frac{1}{2}(\\eta_{\\max} - \\eta_{\\min})\\left(1 + \\cos\\left(\\frac{t\\pi}{T}\\right)\\right)"}),e.jsx("p",{className:"mt-1",children:"Warm-up for the first 5-10 epochs stabilizes early training."})]})]})}const oe=Object.freeze(Object.defineProperty({__proto__:null,default:v},Symbol.toStringTag,{value:"Module"}));function w(){const[n,m]=x.useState("full"),l=["Conv1","Conv2","Conv3","Conv4","Conv5","FC"],c={full:[],"feature-extract":["Conv1","Conv2","Conv3","Conv4","Conv5"],"gradual-unfreeze":["Conv1","Conv2","Conv3"]};return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Fine-Tuning Strategy"}),e.jsx("div",{className:"flex flex-wrap gap-2 mb-4",children:[["full","Full Fine-Tune"],["feature-extract","Feature Extract"],["gradual-unfreeze","Gradual Unfreeze"]].map(([a,r])=>e.jsx("button",{onClick:()=>m(a),className:`px-3 py-1 rounded text-sm ${n===a?"bg-violet-500 text-white":"bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300"}`,children:r},a))}),e.jsx("div",{className:"flex justify-center gap-2",children:l.map(a=>{const r=c[n].includes(a);return e.jsxs("div",{className:"flex flex-col items-center gap-1",children:[e.jsx("div",{className:`w-14 h-10 rounded flex items-center justify-center text-xs font-mono text-white ${r?"bg-gray-400":"bg-violet-500"}`,children:a}),e.jsx("span",{className:"text-xs text-gray-500",children:r?"frozen":"train"})]},a)})})]})}function N(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Transfer learning leverages features learned on large-scale datasets (like ImageNet) and adapts them to new tasks with limited data, dramatically reducing training time and data requirements."}),e.jsxs(g,{title:"Transfer Learning",children:[e.jsxs("p",{children:["Given a source model ",e.jsx(t.InlineMath,{math:"f_\\theta"})," trained on task ",e.jsx(t.InlineMath,{math:"\\mathcal{T}_s"}),", transfer learning adapts parameters to a target task ",e.jsx(t.InlineMath,{math:"\\mathcal{T}_t"}),":"]}),e.jsx(t.BlockMath,{math:"\\theta^* = \\arg\\min_\\theta \\mathcal{L}_t(f_\\theta) \\quad \\text{initialized from } \\theta_s"}),e.jsx("p",{className:"mt-2",children:"Lower layers learn generic features (edges, textures) that transfer well across domains."})]}),e.jsx(w,{}),e.jsxs(j,{title:"Domain Shift Bound",id:"domain-shift",children:[e.jsx("p",{children:"The target risk is bounded by the source risk plus domain divergence:"}),e.jsx(t.BlockMath,{math:"\\epsilon_t(h) \\leq \\epsilon_s(h) + d_{\\mathcal{H}\\Delta\\mathcal{H}}(\\mathcal{D}_s, \\mathcal{D}_t) + \\lambda"}),e.jsxs("p",{className:"mt-1",children:["where ",e.jsx(t.InlineMath,{math:"d_{\\mathcal{H}\\Delta\\mathcal{H}}"})," measures the divergence between source and target distributions, and ",e.jsx(t.InlineMath,{math:"\\lambda"})," is the ideal joint error."]})]}),e.jsxs(f,{title:"Fine-Tuning Learning Rates",children:[e.jsx("p",{children:"Discriminative learning rates assign different rates per layer group:"}),e.jsx(t.BlockMath,{math:"\\eta_l = \\eta_{\\text{base}} \\cdot \\gamma^{L - l}"}),e.jsxs("p",{className:"mt-1",children:["With ",e.jsx(t.InlineMath,{math:"\\gamma = 0.1"}),", early layers train at 100x smaller learning rate than the classification head, preserving pretrained features."]})]}),e.jsx(u,{title:"Transfer Learning with PyTorch",code:`import torch
import torch.nn as nn
from torchvision import models

# Load pretrained ResNet-50
model = models.resnet50(weights='IMAGENET1K_V2')

# Strategy 1: Feature extraction (freeze backbone)
for param in model.parameters():
    param.requires_grad = False

# Replace classifier for new task (10 classes)
model.fc = nn.Sequential(
    nn.Linear(2048, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10),
)

# Strategy 2: Discriminative learning rates
param_groups = [
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3},
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

# Gradual unfreezing after warmup
def unfreeze_layer(model, layer_name):
    for name, param in model.named_parameters():
        if layer_name in name:
            param.requires_grad = True`}),e.jsx(y,{type:"note",title:"When to Fine-Tune vs Feature Extract",children:e.jsxs("p",{children:[e.jsx("strong",{children:"Feature extraction"})," works well when the target dataset is small and similar to the source. ",e.jsx("strong",{children:"Full fine-tuning"})," is preferred when you have sufficient target data or the domains differ significantly. Gradual unfreezing offers a middle ground that often achieves the best results on medium-sized datasets."]})})]})}const le=Object.freeze(Object.defineProperty({__proto__:null,default:N},Symbol.toStringTag,{value:"Module"}));function M(){const[n,m]=x.useState(1),a=((s,o)=>{const h=s.map(p=>Math.exp(p/o)),d=h.reduce((p,b)=>p+b,0);return h.map(p=>p/d)})([5,2,.5,-1],n),r=["Cat","Dog","Bird","Fish"],i=Math.max(...a);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Temperature Scaling Effect"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-4",children:["T = ",n.toFixed(1),e.jsx("input",{type:"range",min:.1,max:10,step:.1,value:n,onChange:s=>m(parseFloat(s.target.value)),className:"w-48 accent-violet-500"})]}),e.jsx("div",{className:"flex items-end gap-3 justify-center h-32",children:a.map((s,o)=>e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsxs("span",{className:"text-xs text-gray-500 mb-1",children:[(s*100).toFixed(1),"%"]}),e.jsx("div",{className:"w-12 bg-violet-500 rounded-t",style:{height:`${s/i*100}px`}}),e.jsx("span",{className:"text-xs mt-1 text-gray-600 dark:text-gray-400",children:r[o]})]},o))}),e.jsx("p",{className:"mt-2 text-center text-xs text-gray-500",children:"Higher T produces softer probability distributions"})]})}function S(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:`Knowledge distillation trains a smaller student model to mimic a larger teacher model, transferring "dark knowledge" encoded in the teacher's soft probability outputs.`}),e.jsxs(g,{title:"Distillation Loss",children:[e.jsx("p",{children:"The student is trained with a combination of hard label loss and soft target loss:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = (1 - \\alpha)\\,\\mathcal{L}_{\\text{CE}}(y, \\hat{y}_s) + \\alpha\\,T^2\\,\\text{KL}\\!\\left(\\sigma\\!\\left(\\frac{z_t}{T}\\right) \\| \\sigma\\!\\left(\\frac{z_s}{T}\\right)\\right)"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"T"})," is the temperature, ",e.jsx(t.InlineMath,{math:"\\alpha"})," balances the two losses, and ",e.jsx(t.InlineMath,{math:"z_t, z_s"})," are teacher/student logits."]})]}),e.jsx(M,{}),e.jsxs(j,{title:"Dark Knowledge",id:"dark-knowledge",children:[e.jsx("p",{children:"The soft targets from the teacher encode inter-class similarities. At high temperature:"}),e.jsx(t.BlockMath,{math:"\\frac{\\partial}{\\partial z_i}\\sigma(z/T)_i \\approx \\frac{1}{T \\cdot C}\\left(1 + \\frac{z_i}{T} - \\frac{\\bar{z}}{T}\\right)"}),e.jsx("p",{className:"mt-1",children:"This reveals that soft targets carry gradient information proportional to logit differences, not just class labels."})]}),e.jsxs(f,{title:"Compression Ratios",children:[e.jsx("p",{children:"Typical distillation results on ImageNet:"}),e.jsxs("ul",{className:"list-disc ml-5 mt-2 space-y-1",children:[e.jsx("li",{children:"Teacher: ResNet-152 (60M params, 78.3% top-1)"}),e.jsx("li",{children:"Student: ResNet-18 (11M params, 71.5% alone, 73.2% distilled)"}),e.jsx("li",{children:"5.5x compression with only 5.1% accuracy drop"})]})]}),e.jsx(u,{title:"Knowledge Distillation in PyTorch",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, T=4.0, alpha=0.7):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # Hard label loss
        hard_loss = self.ce(student_logits, labels)

        # Soft target loss (KL divergence)
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.T, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher,
                             reduction='batchmean') * (self.T ** 2)

        return (1 - self.alpha) * hard_loss + self.alpha * soft_loss

# Training loop
teacher.eval()
criterion = DistillationLoss(T=4.0, alpha=0.7)
for images, labels in loader:
    with torch.no_grad():
        teacher_logits = teacher(images)
    student_logits = student(images)
    loss = criterion(student_logits, teacher_logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()`}),e.jsx(y,{type:"note",title:"Feature Distillation",children:e.jsxs("p",{children:["Beyond logit-level distillation, intermediate feature maps can also be matched. Methods like FitNets align student hidden layers to teacher layers using",e.jsx(t.InlineMath,{math:"\\mathcal{L}_{\\text{hint}} = \\|W_s h_s - h_t\\|^2"}),". This provides richer supervision and often improves student performance further."]})})]})}const ce=Object.freeze(Object.defineProperty({__proto__:null,default:S},Symbol.toStringTag,{value:"Module"}));function T(){const[n,m]=x.useState(60),l={x:40,y:40,w:80,h:60},c={x:n,y:50,w:70,h:50},a=Math.max(l.x,c.x),r=Math.max(l.y,c.y),i=Math.min(l.x+l.w,c.x+c.w),s=Math.min(l.y+l.h,c.y+c.h),o=Math.max(0,i-a)*Math.max(0,s-r),h=l.w*l.h+c.w*c.h-o,d=h>0?o/h:0;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"IoU Interactive Demo"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Prediction X offset:",e.jsx("input",{type:"range",min:0,max:140,value:n,onChange:p=>m(parseInt(p.target.value)),className:"w-40 accent-violet-500"}),e.jsxs("span",{className:"font-mono",children:["IoU = ",d.toFixed(3)]})]}),e.jsxs("svg",{width:240,height:140,className:"mx-auto block",children:[e.jsx("rect",{x:l.x,y:l.y,width:l.w,height:l.h,fill:"none",stroke:"#8b5cf6",strokeWidth:2,strokeDasharray:"4,2"}),e.jsx("rect",{x:c.x,y:c.y,width:c.w,height:c.h,fill:"none",stroke:"#f97316",strokeWidth:2}),o>0&&e.jsx("rect",{x:a,y:r,width:i-a,height:s-r,fill:"#8b5cf6",opacity:.25}),e.jsx("text",{x:l.x,y:l.y-4,fontSize:10,fill:"#8b5cf6",children:"GT"}),e.jsx("text",{x:c.x,y:c.y-4,fontSize:10,fill:"#f97316",children:"Pred"})]})]})}function I(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Anchor-based detectors place predefined bounding box priors across the image and refine them to match objects. This two-stage paradigm (e.g., Faster R-CNN) remains highly accurate."}),e.jsxs(g,{title:"Intersection over Union (IoU)",children:[e.jsx(t.BlockMath,{math:"\\text{IoU}(A, B) = \\frac{|A \\cap B|}{|A \\cup B|} = \\frac{|A \\cap B|}{|A| + |B| - |A \\cap B|}"}),e.jsxs("p",{className:"mt-2",children:["IoU is the primary metric for matching predictions to ground truth boxes. A typical positive threshold is ",e.jsx(t.InlineMath,{math:"\\text{IoU} \\geq 0.5"}),"."]})]}),e.jsx(T,{}),e.jsxs(j,{title:"Anchor Box Regression",id:"anchor-regression",children:[e.jsxs("p",{children:["Given an anchor ",e.jsx(t.InlineMath,{math:"(x_a, y_a, w_a, h_a)"}),", the network predicts offsets:"]}),e.jsx(t.BlockMath,{math:"\\hat{x} = x_a + t_x w_a, \\quad \\hat{y} = y_a + t_y h_a"}),e.jsx(t.BlockMath,{math:"\\hat{w} = w_a e^{t_w}, \\quad \\hat{h} = h_a e^{t_h}"}),e.jsx("p",{className:"mt-1",children:"The smooth L1 loss penalizes box regression errors robustly."})]}),e.jsxs(f,{title:"Non-Maximum Suppression (NMS)",children:[e.jsx("p",{children:"NMS removes duplicate detections by iteratively:"}),e.jsxs("ol",{className:"list-decimal ml-5 mt-2 space-y-1",children:[e.jsx("li",{children:"Select the box with highest confidence score"}),e.jsxs("li",{children:["Remove all boxes with ",e.jsx(t.InlineMath,{math:"\\text{IoU} > \\tau_{\\text{nms}}"})," against it"]}),e.jsx("li",{children:"Repeat until no boxes remain"})]}),e.jsxs("p",{className:"mt-1",children:["Typical ",e.jsx(t.InlineMath,{math:"\\tau_{\\text{nms}} = 0.5"}),"."]})]}),e.jsx(u,{title:"Faster R-CNN with Torchvision",code:`import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.ops import nms, box_iou

# Load pretrained Faster R-CNN
model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
model.eval()

# Inference
images = [torch.rand(3, 640, 480)]
with torch.no_grad():
    predictions = model(images)

boxes = predictions[0]['boxes']    # (N, 4) xyxy format
scores = predictions[0]['scores']  # (N,)
labels = predictions[0]['labels']  # (N,)

# Apply NMS manually
keep = nms(boxes, scores, iou_threshold=0.5)
filtered_boxes = boxes[keep]

# Compute IoU matrix between predictions and GT
gt_boxes = torch.tensor([[50, 50, 200, 200]], dtype=torch.float)
ious = box_iou(filtered_boxes, gt_boxes)
print(f"IoU with GT: {ious.squeeze()}")`}),e.jsx(y,{type:"note",title:"Feature Pyramid Networks",children:e.jsx("p",{children:"FPN builds a multi-scale feature pyramid by combining top-down and lateral connections. This enables detecting objects at different scales: large objects from deep (low-res) features and small objects from shallow (high-res) features. Most modern anchor-based detectors use FPN as the backbone neck."})})]})}const de=Object.freeze(Object.defineProperty({__proto__:null,default:I},Symbol.toStringTag,{value:"Module"}));function D(){const[n,m]=x.useState(120),[l,c]=x.useState(80),a=260,r=180,i=20,s=[];for(let o=0;o<a;o+=4)for(let h=0;h<r;h+=4){const d=(o-n)**2+(h-l)**2,p=Math.exp(-d/(2*i*i));p>.05&&s.push({x:o,y:h,val:p})}return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"CenterNet Heatmap Demo"}),e.jsx("p",{className:"text-sm text-gray-500 dark:text-gray-400 mb-2",children:"Click to move the object center"}),e.jsxs("svg",{width:a,height:r,className:"mx-auto block border border-gray-200 dark:border-gray-700 rounded cursor-crosshair",onClick:o=>{const h=o.currentTarget.getBoundingClientRect();m(o.clientX-h.left),c(o.clientY-h.top)},children:[s.map((o,h)=>e.jsx("rect",{x:o.x,y:o.y,width:4,height:4,fill:"#8b5cf6",opacity:o.val*.8},h)),e.jsx("circle",{cx:n,cy:l,r:3,fill:"#f97316"}),e.jsx("rect",{x:n-40,y:l-30,width:80,height:60,fill:"none",stroke:"#f97316",strokeWidth:1.5}),e.jsx("text",{x:n+5,y:l-33,fontSize:10,fill:"#f97316",children:"center"})]})]})}function C(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Anchor-free detectors eliminate handcrafted anchor boxes by directly predicting object locations as keypoints or center points, simplifying the detection pipeline."}),e.jsxs(g,{title:"CenterNet Formulation",children:[e.jsxs("p",{children:["CenterNet predicts a heatmap ",e.jsx(t.InlineMath,{math:"\\hat{Y} \\in [0,1]^{H \\times W \\times C}"})," where peaks correspond to object centers. For each center, it regresses:"]}),e.jsx(t.BlockMath,{math:"\\hat{Y}_{xyc} = \\exp\\!\\left(-\\frac{(x - \\tilde{p}_x)^2 + (y - \\tilde{p}_y)^2}{2\\sigma_p^2}\\right)"}),e.jsxs("p",{className:"mt-2",children:["Plus offset ",e.jsx(t.InlineMath,{math:"\\hat{O} \\in \\mathbb{R}^{2}"})," and size ",e.jsx(t.InlineMath,{math:"\\hat{S} \\in \\mathbb{R}^{2}"})," at each center."]})]}),e.jsx(D,{}),e.jsxs(j,{title:"FCOS: Fully Convolutional One-Stage",id:"fcos",children:[e.jsxs("p",{children:["FCOS predicts, for each spatial location ",e.jsx(t.InlineMath,{math:"(x,y)"})," on the feature map:"]}),e.jsx(t.BlockMath,{math:"(l^*, t^*, r^*, b^*) = (x - x_0, y - y_0, x_1 - x, y_1 - y)"}),e.jsxs("p",{className:"mt-1",children:["These are distances from the location to the four sides of the bounding box. A centerness score ",e.jsx(t.InlineMath,{math:"\\sqrt{\\frac{\\min(l,r)}{\\max(l,r)} \\cdot \\frac{\\min(t,b)}{\\max(t,b)}}"})," suppresses low-quality predictions far from object centers."]})]}),e.jsxs(f,{title:"Anchor-Free vs Anchor-Based",children:[e.jsx("p",{children:"Key advantages of anchor-free methods:"}),e.jsxs("ul",{className:"list-disc ml-5 mt-2 space-y-1",children:[e.jsx("li",{children:"No hyperparameters for anchor sizes, ratios, or aspect ratios"}),e.jsx("li",{children:"Simpler training with fewer positive/negative sampling heuristics"}),e.jsx("li",{children:"Naturally handle objects of arbitrary shape"}),e.jsx("li",{children:"CenterNet achieves 45.1 AP on COCO at real-time speeds"})]})]}),e.jsx(u,{title:"CenterNet-Style Detection Head",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterNetHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Heatmap head (object centers)
        self.heatmap = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )
        # Box size head (width, height)
        self.size = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
        )
        # Offset head (sub-pixel refinement)
        self.offset = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
        )

    def forward(self, x):
        hm = torch.sigmoid(self.heatmap(x))
        sz = self.size(x)
        off = self.offset(x)
        return hm, sz, off

# Focal loss for heatmap training
def focal_loss(pred, gt, alpha=2, beta=4):
    pos = gt.eq(1).float()
    neg = gt.lt(1).float()
    pos_loss = -((1 - pred)**alpha * torch.log(pred + 1e-6)) * pos
    neg_loss = -((1 - gt)**beta * pred**alpha * torch.log(1 - pred + 1e-6)) * neg
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1)`}),e.jsx(y,{type:"note",title:"Keypoint Detection",children:e.jsx("p",{children:"Anchor-free methods naturally extend to keypoint detection (e.g., CornerNet detects top-left and bottom-right corners). This paradigm unifies object detection, pose estimation, and instance segmentation under a single keypoint-based framework."})})]})}const he=Object.freeze(Object.defineProperty({__proto__:null,default:C},Symbol.toStringTag,{value:"Module"}));function P(){const[n,m]=x.useState(0),l=280,c=200,a=Array.from({length:80},(i,s)=>({x:Math.cos(s*.45)*40+Math.random()*20,y:Math.sin(s*.45)*30+Math.random()*15,z:s%20*2-20+Math.random()*5})),r=n*Math.PI/180;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"3D Point Cloud View"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Rotation: ",n,"°",e.jsx("input",{type:"range",min:0,max:360,value:n,onChange:i=>m(parseInt(i.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("svg",{width:l,height:c,className:"mx-auto block",children:[a.map((i,s)=>{const o=i.x*Math.cos(r)-i.z*Math.sin(r),d=1+(i.x*Math.sin(r)+i.z*Math.cos(r))/100;return e.jsx("circle",{cx:l/2+o*d,cy:c/2-i.y*d,r:2*d,fill:"#8b5cf6",opacity:.4+d*.3},s)}),e.jsx("rect",{x:l/2-45,y:c/2-35,width:90,height:50,fill:"none",stroke:"#f97316",strokeWidth:1.5,strokeDasharray:"4,2"}),e.jsx("text",{x:l/2-44,y:c/2-38,fontSize:10,fill:"#f97316",children:"3D BBox"})]})]})}function z(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"3D object detection estimates object positions, dimensions, and orientations in three-dimensional space. Key representations include point clouds from LiDAR and bird's-eye view projections."}),e.jsxs(g,{title:"3D Bounding Box",children:[e.jsx("p",{children:"A 3D box is parameterized by 7 degrees of freedom:"}),e.jsx(t.BlockMath,{math:"\\mathbf{b} = (x, y, z, w, h, l, \\theta)"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"(x, y, z)"})," is the center, ",e.jsx(t.InlineMath,{math:"(w, h, l)"})," are dimensions, and ",e.jsx(t.InlineMath,{math:"\\theta"})," is yaw rotation. The 3D IoU considers volumetric overlap."]})]}),e.jsx(P,{}),e.jsxs(j,{title:"PointNet Feature Extraction",id:"pointnet",children:[e.jsxs("p",{children:["PointNet processes unordered point sets directly. For ",e.jsx(t.InlineMath,{math:"N"})," points with features ",e.jsx(t.InlineMath,{math:"x_i"}),":"]}),e.jsx(t.BlockMath,{math:"f = \\gamma\\!\\left(\\max_{i=1,\\ldots,N} h(x_i)\\right)"}),e.jsxs("p",{className:"mt-1",children:["where ",e.jsx(t.InlineMath,{math:"h"})," is a shared MLP and ",e.jsx(t.InlineMath,{math:"\\gamma"})," is another MLP. The max-pooling ensures permutation invariance over input points."]})]}),e.jsx(f,{title:"3D Detection Methods Comparison",children:e.jsxs("ul",{className:"list-disc ml-5 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Point-based"}),": PointRCNN processes raw points directly (accurate, slow)"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Voxel-based"}),": VoxelNet/SECOND discretize into voxel grid (fast, scalable)"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Pillar-based"}),": PointPillars uses vertical columns (real-time, autonomous driving)"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"BEV-based"}),": BEVFusion projects camera + LiDAR to bird's-eye view"]})]})}),e.jsx(u,{title:"Voxelization and 3D Detection",code:`import torch
import torch.nn as nn

class VoxelEncoder(nn.Module):
    """Simplified voxel feature encoder (VFE)."""
    def __init__(self, in_dim=4, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )

    def forward(self, voxel_features, voxel_count):
        # voxel_features: (num_voxels, max_points, C)
        # Aggregate points within each voxel
        x = self.mlp(voxel_features.view(-1, 4))
        x = x.view(*voxel_features.shape[:2], -1)
        # Max pooling over points in each voxel
        x = x.max(dim=1).values  # (num_voxels, hidden)
        return x

# Bird's Eye View projection
def scatter_to_bev(voxel_feats, coords, grid_size=(512, 512)):
    """Scatter voxel features to BEV grid."""
    bev = torch.zeros(1, voxel_feats.shape[1],
                      *grid_size, device=voxel_feats.device)
    bev[0, :, coords[:, 2], coords[:, 3]] = voxel_feats.T
    return bev  # Apply 2D conv backbone on BEV map

# Loss: smooth L1 for box regression + focal for class
def detection_loss(pred_boxes, gt_boxes, pred_cls, gt_cls):
    reg_loss = nn.SmoothL1Loss()(pred_boxes, gt_boxes)
    cls_loss = sigmoid_focal_loss(pred_cls, gt_cls)
    return reg_loss + cls_loss`}),e.jsx(y,{type:"note",title:"Multi-Modal Fusion",children:e.jsx("p",{children:"Modern autonomous driving systems fuse camera images with LiDAR point clouds. BEVFusion projects both modalities into a shared bird's-eye view space using camera-to-BEV transformations like LSS (Lift, Splat, Shoot), enabling unified feature extraction and detection."})})]})}const me=Object.freeze(Object.defineProperty({__proto__:null,default:z},Symbol.toStringTag,{value:"Module"}));function A(){const[n,m]=x.useState(0),l=[{name:"Input",boxes:12,color:"#9ca3af"},{name:"P-Net (12x12)",boxes:8,color:"#8b5cf6"},{name:"R-Net (24x24)",boxes:4,color:"#7c3aed"},{name:"O-Net (48x48)",boxes:2,color:"#6d28d9"}],c=300,a=140;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"MTCNN Cascade Stages"}),e.jsx("div",{className:"flex gap-2 mb-4",children:l.map((r,i)=>e.jsx("button",{onClick:()=>m(i),className:`px-3 py-1 rounded text-sm ${n===i?"bg-violet-500 text-white":"bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300"}`,children:r.name},i))}),e.jsxs("svg",{width:c,height:a,className:"mx-auto block",children:[e.jsx("rect",{x:10,y:10,width:120,height:120,fill:"#f3f4f6",stroke:"#d1d5db",rx:4}),e.jsx("text",{x:70,y:75,textAnchor:"middle",fontSize:11,fill:"#6b7280",children:"Image"}),Array.from({length:l[n].boxes}).map((r,i)=>{const s=150+i%4*35,o=20+Math.floor(i/4)*55;return e.jsx("rect",{x:s,y:o,width:28,height:28,fill:"none",stroke:l[n].color,strokeWidth:2,rx:2},i)}),e.jsxs("text",{x:220,y:a-8,fontSize:11,fill:"#6b7280",children:[l[n].boxes," candidates"]})]})]})}function L(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Face detection localizes all faces in an image with bounding boxes and optional landmark points. Modern detectors achieve robust performance across scales, poses, and occlusions."}),e.jsxs(g,{title:"Multi-Task Face Detection",children:[e.jsx("p",{children:"Face detectors jointly optimize multiple objectives:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = \\lambda_1 \\mathcal{L}_{\\text{cls}} + \\lambda_2 \\mathcal{L}_{\\text{box}} + \\lambda_3 \\mathcal{L}_{\\text{landmark}}"}),e.jsx("p",{className:"mt-2",children:"where classification loss determines face/non-face, box regression refines location, and landmark loss localizes facial keypoints (eyes, nose, mouth corners)."})]}),e.jsx(A,{}),e.jsxs(j,{title:"RetinaFace Multi-Task Loss",id:"retinaface",children:[e.jsx("p",{children:"RetinaFace adds a dense regression branch for 3D face vertices:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = \\mathcal{L}_{\\text{cls}} + \\lambda_1 \\mathcal{L}_{\\text{box}} + \\lambda_2 \\mathcal{L}_{\\text{pts}} + \\lambda_3 \\mathcal{L}_{\\text{mesh}}"}),e.jsxs("p",{className:"mt-1",children:["The mesh loss leverages a graph convolution decoder that predicts a 3D face shape ",e.jsx(t.InlineMath,{math:"\\mathbf{S} \\in \\mathbb{R}^{N \\times 3}"}),", providing self-supervision that improves 2D detection accuracy."]})]}),e.jsxs(f,{title:"MTCNN Pipeline",children:[e.jsxs("ol",{className:"list-decimal ml-5 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"P-Net"}),": Shallow CNN on image pyramid, produces candidate boxes at 12x12"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"R-Net"}),": Refines candidates at 24x24, rejects false positives"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"O-Net"}),": Final stage at 48x48, outputs boxes + 5 landmarks"]})]}),e.jsx("p",{className:"mt-2",children:"Each stage reduces candidates by roughly 50-80%."})]}),e.jsx(u,{title:"RetinaFace with InsightFace",code:`import torch
import torch.nn as nn

class RetinaFaceHead(nn.Module):
    """Simplified RetinaFace detection head."""
    def __init__(self, in_channels=256, num_anchors=2):
        super().__init__()
        self.cls = nn.Conv2d(in_channels, num_anchors * 2, 1)
        self.box = nn.Conv2d(in_channels, num_anchors * 4, 1)
        self.landmark = nn.Conv2d(in_channels, num_anchors * 10, 1)

    def forward(self, x):
        return self.cls(x), self.box(x), self.landmark(x)

# Multi-scale feature pyramid for face detection
class FaceFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([
            RetinaFaceHead() for _ in range(3)  # 3 FPN levels
        ])

    def forward(self, fpn_features):
        results = []
        for feat, head in zip(fpn_features, self.heads):
            cls, box, lmk = head(feat)
            results.append({
                'cls': cls,     # (B, 2A, H, W)
                'box': box,     # (B, 4A, H, W)
                'lmk': lmk,    # (B, 10A, H, W) - 5 landmarks
            })
        return results

# Evaluation: compute AP at IoU=0.5
def compute_ap(pred_boxes, gt_boxes, iou_thresh=0.5):
    """Average precision for face detection."""
    from torchvision.ops import box_iou
    ious = box_iou(pred_boxes, gt_boxes)
    matches = ious.max(dim=1).values >= iou_thresh
    return matches.float().mean()`}),e.jsx(y,{type:"note",title:"Handling Tiny Faces",children:e.jsx("p",{children:"Detecting small faces (under 20px) remains challenging. Key strategies include using high-resolution feature maps from FPN, training with image pyramids, and employing special anchor designs for small scales. RetinaFace achieves 91.4% AP on WIDER FACE hard set by leveraging these techniques."})})]})}const xe=Object.freeze(Object.defineProperty({__proto__:null,default:L},Symbol.toStringTag,{value:"Module"}));function B(){const[n,m]=x.useState(.3),l=280,c=200,a={x:100,y:100},r={x:130,y:80},i={x:200,y:140},s=Math.sqrt((a.x-r.x)**2+(a.y-r.y)**2),o=Math.sqrt((a.x-i.x)**2+(a.y-i.y)**2),h=n*200;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Triplet Loss Embedding Space"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Margin: ",n.toFixed(2),e.jsx("input",{type:"range",min:.05,max:.8,step:.05,value:n,onChange:d=>m(parseFloat(d.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("svg",{width:l,height:c,className:"mx-auto block",children:[e.jsx("circle",{cx:a.x,cy:a.y,r:h,fill:"none",stroke:"#8b5cf6",strokeWidth:1,strokeDasharray:"4,3",opacity:.5}),e.jsx("line",{x1:a.x,y1:a.y,x2:r.x,y2:r.y,stroke:"#22c55e",strokeWidth:1.5}),e.jsx("line",{x1:a.x,y1:a.y,x2:i.x,y2:i.y,stroke:"#ef4444",strokeWidth:1.5}),e.jsx("circle",{cx:a.x,cy:a.y,r:6,fill:"#8b5cf6"}),e.jsx("circle",{cx:r.x,cy:r.y,r:6,fill:"#22c55e"}),e.jsx("circle",{cx:i.x,cy:i.y,r:6,fill:"#ef4444"}),e.jsx("text",{x:a.x-20,y:a.y+18,fontSize:10,fill:"#8b5cf6",children:"Anchor"}),e.jsx("text",{x:r.x-5,y:r.y-10,fontSize:10,fill:"#22c55e",children:"Positive"}),e.jsx("text",{x:i.x-5,y:i.y+16,fontSize:10,fill:"#ef4444",children:"Negative"})]}),e.jsxs("p",{className:"mt-1 text-center text-xs text-gray-500",children:["d(A,P) = ",(s/200).toFixed(2)," | d(A,N) = ",(o/200).toFixed(2)," | loss = ",Math.max(0,s/200-o/200+n).toFixed(3)]})]})}function F(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Face recognition maps face images to compact embedding vectors where distance reflects identity similarity. Modern systems use metric learning losses to achieve superhuman accuracy."}),e.jsxs(g,{title:"Triplet Loss (FaceNet)",children:[e.jsxs("p",{children:["Given an anchor ",e.jsx(t.InlineMath,{math:"a"}),", positive ",e.jsx(t.InlineMath,{math:"p"})," (same identity), and negative ",e.jsx(t.InlineMath,{math:"n"})," (different identity):"]}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_{\\text{triplet}} = \\max\\!\\left(0,\\; \\|f(a) - f(p)\\|^2 - \\|f(a) - f(n)\\|^2 + m\\right)"}),e.jsxs("p",{className:"mt-2",children:["The margin ",e.jsx(t.InlineMath,{math:"m"})," enforces a minimum gap between positive and negative pairs in the embedding space."]})]}),e.jsx(B,{}),e.jsxs(j,{title:"ArcFace Angular Margin",id:"arcface",children:[e.jsx("p",{children:"ArcFace adds an angular margin to the softmax classification loss:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_{\\text{arc}} = -\\log \\frac{e^{s \\cos(\\theta_{y_i} + m)}}{e^{s \\cos(\\theta_{y_i} + m)} + \\sum_{j \\neq y_i} e^{s \\cos \\theta_j}}"}),e.jsxs("p",{className:"mt-1",children:["where ",e.jsx(t.InlineMath,{math:"\\theta_j = \\arccos(W_j^T f(x))"})," is the angle between the feature and class center, ",e.jsx(t.InlineMath,{math:"s"})," is a scale factor, and ",e.jsx(t.InlineMath,{math:"m"})," is the additive angular margin."]})]}),e.jsx(f,{title:"Face Verification Pipeline",children:e.jsxs("ol",{className:"list-decimal ml-5 space-y-1",children:[e.jsx("li",{children:"Detect and align faces using landmarks (5-point alignment)"}),e.jsxs("li",{children:["Extract 512-d embedding: ",e.jsx(t.InlineMath,{math:"v = f_\\theta(\\text{face})"})]}),e.jsxs("li",{children:["L2-normalize: ",e.jsx(t.InlineMath,{math:"\\hat{v} = v / \\|v\\|"})]}),e.jsxs("li",{children:["Compare cosine similarity: ",e.jsx(t.InlineMath,{math:"\\text{sim} = \\hat{v}_1 \\cdot \\hat{v}_2"})]}),e.jsxs("li",{children:["Threshold at ",e.jsx(t.InlineMath,{math:"\\tau \\approx 0.4"})," for same/different identity"]})]})}),e.jsx(u,{title:"ArcFace Loss Implementation",code:`import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    def __init__(self, embed_dim=512, num_classes=10000,
                 s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, embeddings, labels):
        # Normalize weights and features
        W = F.normalize(self.W, dim=1)
        x = F.normalize(embeddings, dim=1)

        # Compute cos(theta)
        cosine = x @ W.T  # (B, num_classes)

        # Add angular margin to target class
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        target_logits = torch.cos(theta[range(len(labels)), labels] + self.m)
        cosine[range(len(labels)), labels] = target_logits

        # Scale and compute cross-entropy
        logits = cosine * self.s
        return F.cross_entropy(logits, labels)

# Usage
loss_fn = ArcFaceLoss(embed_dim=512, num_classes=85742)
embeddings = backbone(face_images)  # (B, 512)
loss = loss_fn(embeddings, identity_labels)`}),e.jsx(y,{type:"note",title:"Hard Mining Strategies",children:e.jsx("p",{children:"Training efficiency depends heavily on selecting informative triplets. Online hard mining selects the hardest positive (farthest same-identity) and hardest negative (closest different-identity) within each mini-batch. Semi-hard mining selects negatives that are farther than the positive but still within the margin, providing more stable gradients."})})]})}const pe=Object.freeze(Object.defineProperty({__proto__:null,default:F},Symbol.toStringTag,{value:"Module"}));function R(){const[n,m]=x.useState(50),[l,c]=x.useState(50),a=240,r=160,i=l*3.6;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Style Mixing Visualization"}),e.jsxs("div",{className:"flex flex-col gap-2 mb-4",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Coarse style (pose, shape):",e.jsx("input",{type:"range",min:0,max:100,value:n,onChange:s=>m(parseInt(s.target.value)),className:"w-36 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Fine style (color, texture):",e.jsx("input",{type:"range",min:0,max:100,value:l,onChange:s=>c(parseInt(s.target.value)),className:"w-36 accent-violet-500"})]})]}),e.jsxs("svg",{width:a,height:r,className:"mx-auto block",children:[e.jsx("ellipse",{cx:a/2,cy:r/2-10,rx:35+n*.15,ry:45+n*.1,fill:`hsl(${i}, 60%, 75%)`,stroke:"#8b5cf6",strokeWidth:1.5}),e.jsx("circle",{cx:a/2-12,cy:r/2-20,r:4,fill:`hsl(${i}, 70%, 40%)`}),e.jsx("circle",{cx:a/2+12,cy:r/2-20,r:4,fill:`hsl(${i}, 70%, 40%)`}),e.jsx("path",{d:`M${a/2-8},${r/2+5} Q${a/2},${r/2+15} ${a/2+8},${r/2+5}`,fill:"none",stroke:`hsl(${i}, 60%, 50%)`,strokeWidth:1.5}),e.jsxs("text",{x:a/2,y:r-5,textAnchor:"middle",fontSize:10,fill:"#6b7280",children:["Coarse: ",n,"% | Fine: ",l,"%"]})]})]})}function W(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Face generation using GANs (particularly StyleGAN) can create photorealistic synthetic faces. This technology enables face editing, but also raises concerns about deepfakes."}),e.jsxs(g,{title:"StyleGAN Architecture",children:[e.jsx("p",{children:"StyleGAN maps a latent code through a mapping network to a style space:"}),e.jsx(t.BlockMath,{math:"z \\in \\mathcal{Z} \\xrightarrow{f} w \\in \\mathcal{W} \\xrightarrow{\\text{AdaIN}} \\text{synthesis}"}),e.jsx("p",{className:"mt-2",children:"Adaptive Instance Normalization (AdaIN) injects style at each layer:"}),e.jsx(t.BlockMath,{math:"\\text{AdaIN}(x_i, y) = y_{s,i}\\frac{x_i - \\mu(x_i)}{\\sigma(x_i)} + y_{b,i}"})]}),e.jsx(R,{}),e.jsxs(j,{title:"W+ Space for Editing",id:"w-plus",children:[e.jsxs("p",{children:["The extended ",e.jsx(t.InlineMath,{math:"\\mathcal{W}^+"})," space uses different ",e.jsx(t.InlineMath,{math:"w"})," vectors per layer, enabling fine-grained control:"]}),e.jsx(t.BlockMath,{math:"\\mathcal{W}^+ = \\{(w_1, w_2, \\ldots, w_L) \\mid w_i \\in \\mathcal{W}\\}"}),e.jsxs("p",{className:"mt-1",children:["Editing directions ",e.jsx(t.InlineMath,{math:"n"})," in ",e.jsx(t.InlineMath,{math:"\\mathcal{W}"})," correspond to semantic attributes: ",e.jsx(t.InlineMath,{math:"w' = w + \\alpha \\cdot n"})," (e.g., age, smile, pose)."]})]}),e.jsxs(f,{title:"StyleGAN Layers and Attributes",children:[e.jsxs("ul",{className:"list-disc ml-5 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Coarse layers (4-8)"}),": pose, face shape, hairstyle"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Middle layers (16-32)"}),": facial features, eye shape, nose"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Fine layers (64-1024)"}),": color scheme, skin texture, lighting"]})]}),e.jsxs("p",{className:"mt-2",children:["Style mixing applies different ",e.jsx(t.InlineMath,{math:"w"})," codes at different resolution layers."]})]}),e.jsx(u,{title:"StyleGAN2 Inference and Editing",code:`import torch
import torch.nn as nn

class MappingNetwork(nn.Module):
    """StyleGAN2 mapping network: Z -> W."""
    def __init__(self, z_dim=512, w_dim=512, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = z_dim if i == 0 else w_dim
            layers.extend([nn.Linear(in_d, w_dim), nn.LeakyReLU(0.2)])
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)  # w: (B, 512)

# Face editing in W space
def edit_face(w, direction, alpha=3.0):
    """Edit a face attribute by moving in latent space."""
    return w + alpha * direction

# Deepfake detection features
def extract_frequency_features(image):
    """DCT-based frequency analysis for deepfake detection."""
    # Real faces have consistent high-frequency patterns
    # GAN-generated faces often lack certain frequencies
    dct = torch.fft.fft2(image)
    magnitude = torch.abs(dct)
    # High-frequency energy ratio as detection feature
    h, w = magnitude.shape[-2:]
    center = magnitude[..., h//4:3*h//4, w//4:3*w//4].sum()
    total = magnitude.sum()
    return 1.0 - center / total  # Higher = more HF content`}),e.jsx(_,{title:"Deepfake Detection",children:e.jsxs("p",{children:["Detecting AI-generated faces relies on artifacts invisible to humans: inconsistent specular reflections, frequency spectrum anomalies, and temporal flickering in videos. Binary classifiers trained on real vs generated faces achieve over 95% detection accuracy but struggle with unseen generators. Multi-spectral analysis using ",e.jsx(t.InlineMath,{math:"\\mathcal{F}\\{I\\}"})," (frequency domain) provides more robust detection signals."]})}),e.jsx(y,{type:"note",title:"Ethical Considerations",children:e.jsx("p",{children:"Face generation technology requires responsible use. Key concerns include non-consensual deepfakes, identity fraud, and misinformation. Research in provenance tracking, watermarking, and robust detection methods is critical for mitigating misuse while preserving beneficial applications."})})]})}const ge=Object.freeze(Object.defineProperty({__proto__:null,default:W},Symbol.toStringTag,{value:"Module"}));function E(){const[n,m]=x.useState(1),l=7,c=3,a=220,r=a/l,i=(s,o)=>{const h=Math.abs(s-c),d=Math.abs(o-c);return h===0&&d<=n||d===0&&h<=n||h===n&&d===n||h<=1&&d<=1};return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Dilated Convolution Receptive Field"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Dilation rate: ",n,e.jsx("input",{type:"range",min:1,max:3,step:1,value:n,onChange:s=>m(parseInt(s.target.value)),className:"w-32 accent-violet-500"})]}),e.jsx("svg",{width:a,height:a,className:"mx-auto block",children:Array.from({length:l}).map((s,o)=>Array.from({length:l}).map((h,d)=>e.jsx("rect",{x:d*r,y:o*r,width:r-1,height:r-1,fill:o===c&&d===c?"#8b5cf6":i(o,d)?"#c4b5fd":"#f3f4f6",stroke:"#d1d5db",strokeWidth:.5,rx:2},`${o}-${d}`)))}),e.jsxs("p",{className:"mt-2 text-center text-xs text-gray-500",children:["Receptive field: ",2*n+1,"x",2*n+1," | No resolution loss"]})]})}function H(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Semantic segmentation assigns a class label to every pixel in an image. Encoder-decoder architectures and dilated convolutions are the foundational techniques for this task."}),e.jsxs(g,{title:"Semantic Segmentation",children:[e.jsxs("p",{children:["Given an image ",e.jsx(t.InlineMath,{math:"I \\in \\mathbb{R}^{H \\times W \\times 3}"}),", predict a label map ",e.jsx(t.InlineMath,{math:"Y \\in \\{1, \\ldots, C\\}^{H \\times W}"}),":"]}),e.jsx(t.BlockMath,{math:"Y_{ij} = \\arg\\max_c \\; f_\\theta(I)_{ijc}"}),e.jsx("p",{className:"mt-2",children:"The per-pixel cross-entropy loss sums over all spatial locations:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = -\\frac{1}{HW}\\sum_{i,j}\\sum_{c} y_{ijc}\\log\\hat{y}_{ijc}"})]}),e.jsx(E,{}),e.jsxs(j,{title:"Dilated (Atrous) Convolution",id:"dilated-conv",children:[e.jsxs("p",{children:["Dilated convolution with rate ",e.jsx(t.InlineMath,{math:"r"})," expands the kernel without adding parameters:"]}),e.jsx(t.BlockMath,{math:"(f *_r k)(p) = \\sum_{s+rt=p} f(s) \\cdot k(t)"}),e.jsxs("p",{className:"mt-1",children:["Effective receptive field for kernel size ",e.jsx(t.InlineMath,{math:"k"})," with dilation ",e.jsx(t.InlineMath,{math:"r"}),":",e.jsx(t.InlineMath,{math:"\\; k_{\\text{eff}} = k + (k-1)(r-1)"}),". This captures multi-scale context without downsampling."]})]}),e.jsx(f,{title:"Key Architectures",children:e.jsxs("ul",{className:"list-disc ml-5 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"FCN"}),": First fully convolutional approach, upsamples via transposed convolutions"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"U-Net"}),": Skip connections between encoder and decoder at each scale"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"DeepLabv3+"}),": ASPP module with parallel dilated convolutions (rates 6, 12, 18)"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"PSPNet"}),": Pyramid pooling at multiple scales for global context"]})]})}),e.jsx(u,{title:"DeepLabv3+ Segmentation",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""
    def __init__(self, in_ch=2048, out_ch=256, rates=[6, 12, 18]):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, 1)
        self.atrous = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r)
            for r in rates
        ])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1),
        )
        self.project = nn.Conv2d(out_ch * (len(rates) + 2), out_ch, 1)

    def forward(self, x):
        h, w = x.shape[2:]
        feats = [self.conv1x1(x)]
        feats += [conv(x) for conv in self.atrous]
        feats.append(F.interpolate(self.pool(x), (h, w),
                                   mode='bilinear'))
        return self.project(torch.cat(feats, dim=1))

# Mean IoU evaluation metric
def mean_iou(pred, target, num_classes):
    ious = []
    for c in range(num_classes):
        inter = ((pred == c) & (target == c)).sum().float()
        union = ((pred == c) | (target == c)).sum().float()
        ious.append((inter / union.clamp(min=1)).item())
    return sum(ious) / len(ious)`}),e.jsx(y,{type:"note",title:"Class Imbalance in Segmentation",children:e.jsxs("p",{children:["Pixel-level class imbalance is severe (e.g., road vs sign in driving scenes). Solutions include weighted cross-entropy, focal loss, and Dice loss:",e.jsx(t.InlineMath,{math:"\\mathcal{L}_{\\text{Dice}} = 1 - \\frac{2|P \\cap G|}{|P| + |G|}"}),". Combining cross-entropy with Dice loss often gives the best results."]})})]})}const fe=Object.freeze(Object.defineProperty({__proto__:null,default:H},Symbol.toStringTag,{value:"Module"}));function q(){const[n,m]=x.useState(!0),[l,c]=x.useState(!0),a=280,r=180,i=[{x:30,y:30,w:70,h:90,color:"#8b5cf6",label:"Person 1"},{x:120,y:50,w:60,h:80,color:"#f97316",label:"Person 2"},{x:200,y:60,w:55,h:70,color:"#22c55e",label:"Dog"}];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Instance Segmentation Output"}),e.jsxs("div",{className:"flex gap-4 mb-3",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:[e.jsx("input",{type:"checkbox",checked:n,onChange:s=>m(s.target.checked),className:"accent-violet-500"}),"Show masks"]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:[e.jsx("input",{type:"checkbox",checked:l,onChange:s=>c(s.target.checked),className:"accent-violet-500"}),"Show boxes"]})]}),e.jsx("svg",{width:a,height:r,className:"mx-auto block bg-gray-50 dark:bg-gray-800 rounded",children:i.map((s,o)=>e.jsxs("g",{children:[n&&e.jsx("ellipse",{cx:s.x+s.w/2,cy:s.y+s.h/2,rx:s.w/2.2,ry:s.h/2.2,fill:s.color,opacity:.3}),l&&e.jsx("rect",{x:s.x,y:s.y,width:s.w,height:s.h,fill:"none",stroke:s.color,strokeWidth:2}),e.jsx("text",{x:s.x,y:s.y-4,fontSize:10,fill:s.color,fontWeight:"bold",children:s.label})]},o))})]})}function O(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Instance segmentation combines object detection with pixel-level segmentation, producing individual masks for each object instance. Mask R-CNN is the foundational approach."}),e.jsxs(g,{title:"Instance vs Semantic Segmentation",children:[e.jsx("p",{children:"Semantic segmentation labels pixels by class. Instance segmentation additionally distinguishes individual objects of the same class:"}),e.jsx(t.BlockMath,{math:"\\text{Output} = \\{(c_k, m_k, s_k)\\}_{k=1}^{K}"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"c_k"})," is the class, ",e.jsx(t.InlineMath,{math:"m_k \\in \\{0,1\\}^{H \\times W}"})," is the binary mask, and ",e.jsx(t.InlineMath,{math:"s_k"})," is the confidence score for the ",e.jsx(t.InlineMath,{math:"k"}),"-th instance."]})]}),e.jsx(q,{}),e.jsxs(j,{title:"Mask R-CNN Architecture",id:"mask-rcnn",children:[e.jsx("p",{children:"Mask R-CNN extends Faster R-CNN with a parallel mask prediction branch:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = \\mathcal{L}_{\\text{cls}} + \\mathcal{L}_{\\text{box}} + \\mathcal{L}_{\\text{mask}}"}),e.jsx("p",{className:"mt-1",children:"The mask loss is per-pixel binary cross-entropy applied only to the ground truth class:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_{\\text{mask}} = -\\frac{1}{m^2}\\sum_{ij}\\left[y_{ij}\\log\\hat{m}_{ij}^c + (1-y_{ij})\\log(1-\\hat{m}_{ij}^c)\\right]"}),e.jsxs("p",{className:"mt-1",children:["where ",e.jsx(t.InlineMath,{math:"m = 28"})," is the mask resolution and ",e.jsx(t.InlineMath,{math:"c"})," is the predicted class."]})]}),e.jsxs(f,{title:"RoIAlign vs RoIPool",children:[e.jsx("p",{children:"RoIPool introduces quantization errors by snapping to grid cells. RoIAlign uses bilinear interpolation at exact floating-point coordinates:"}),e.jsx(t.BlockMath,{math:"\\text{RoIAlign}(x, y) = \\sum_{ij} \\max(0, 1 - |x - x_i|) \\cdot \\max(0, 1 - |y - y_j|) \\cdot f_{ij}"}),e.jsx("p",{className:"mt-1",children:"This eliminates misalignment artifacts and improves mask AP by 1-3 points."})]}),e.jsx(u,{title:"Mask R-CNN with Torchvision",code:`import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

# Load pretrained Mask R-CNN
model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')
model.eval()

# Inference
images = [torch.rand(3, 640, 480)]
with torch.no_grad():
    predictions = model(images)

# Each prediction contains:
boxes = predictions[0]['boxes']    # (N, 4) bounding boxes
labels = predictions[0]['labels']  # (N,) class labels
scores = predictions[0]['scores']  # (N,) confidence scores
masks = predictions[0]['masks']    # (N, 1, H, W) instance masks

# Filter by confidence
keep = scores > 0.7
final_masks = masks[keep] > 0.5  # Binary masks at threshold 0.5

# Panoptic segmentation: combine instance + semantic
def merge_to_panoptic(instance_masks, semantic_pred):
    """Merge instance and semantic for panoptic output."""
    panoptic = semantic_pred.clone()
    for i, mask in enumerate(instance_masks):
        panoptic[mask.squeeze()] = 1000 + i  # Unique instance ID
    return panoptic

print(f"Detected {keep.sum()} instances")
print(f"Mask shape: {final_masks.shape}")`}),e.jsx(y,{type:"note",title:"Panoptic Segmentation",children:e.jsx("p",{children:'Panoptic segmentation unifies instance and semantic segmentation into a single task. Every pixel gets both a class label and an instance ID. "Things" (countable objects) get instance IDs while "stuff" (amorphous regions like sky, road) share a single ID per class. Modern approaches like Panoptic FPN and MaskFormer handle both in one model.'})})]})}const ue=Object.freeze(Object.defineProperty({__proto__:null,default:O},Symbol.toStringTag,{value:"Module"}));function V(){const[n,m]=x.useState("point"),[l,c]=x.useState({x:140,y:90}),a=280,r=180,i={x:140,y:90},s=d=>{const p=d.currentTarget.getBoundingClientRect();c({x:d.clientX-p.left,y:d.clientY-p.top})},o=Math.sqrt((l.x-i.x)**2+(l.y-i.y)**2),h=n==="point"?o<80:!0;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"SAM Prompt Types"}),e.jsx("div",{className:"flex gap-2 mb-3",children:["point","box","auto"].map(d=>e.jsx("button",{onClick:()=>m(d),className:`px-3 py-1 rounded text-sm capitalize ${n===d?"bg-violet-500 text-white":"bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300"}`,children:d},d))}),e.jsxs("svg",{width:a,height:r,className:"mx-auto block bg-gray-50 dark:bg-gray-800 rounded cursor-crosshair",onClick:s,children:[h&&e.jsx("ellipse",{cx:i.x,cy:i.y,rx:60,ry:45,fill:"#8b5cf6",opacity:.25,stroke:"#8b5cf6",strokeWidth:1.5}),n==="point"&&e.jsxs(e.Fragment,{children:[e.jsx("circle",{cx:l.x,cy:l.y,r:5,fill:"#22c55e",stroke:"white",strokeWidth:1.5}),e.jsx("text",{x:l.x+8,y:l.y+4,fontSize:10,fill:"#22c55e",children:"click"})]}),n==="box"&&e.jsx("rect",{x:80,y:45,width:120,height:90,fill:"none",stroke:"#f97316",strokeWidth:2,strokeDasharray:"4,2"}),n==="auto"&&e.jsxs(e.Fragment,{children:[e.jsx("ellipse",{cx:80,cy:70,rx:30,ry:25,fill:"#8b5cf6",opacity:.2,stroke:"#8b5cf6",strokeWidth:1}),e.jsx("ellipse",{cx:200,cy:110,rx:35,ry:28,fill:"#f97316",opacity:.2,stroke:"#f97316",strokeWidth:1}),e.jsx("text",{x:a/2,y:r-8,textAnchor:"middle",fontSize:10,fill:"#6b7280",children:"Automatic everything mode"})]})]})]})}function U(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"The Segment Anything Model (SAM) is a foundation model for image segmentation that can segment any object given a prompt. Trained on 1B+ masks, it generalizes to unseen domains."}),e.jsxs(g,{title:"SAM Architecture",children:[e.jsx("p",{children:"SAM consists of three components:"}),e.jsx(t.BlockMath,{math:"\\text{Image} \\xrightarrow{\\text{ViT Encoder}} \\mathbf{E} \\xrightarrow[\\text{Prompt}]{\\text{Decoder}} \\text{Masks}"}),e.jsx("p",{className:"mt-2",children:"The image encoder runs once, then the lightweight mask decoder produces masks for any number of prompts (points, boxes, or text) in real-time."})]}),e.jsx(V,{}),e.jsxs(j,{title:"Promptable Segmentation",id:"promptable-seg",children:[e.jsx("p",{children:"SAM's mask decoder uses cross-attention between prompt tokens and image embeddings:"}),e.jsx(t.BlockMath,{math:"\\mathbf{Q} = \\text{Prompt Tokens}, \\quad \\mathbf{K} = \\mathbf{V} = \\mathbf{E}_{\\text{img}}"}),e.jsx(t.BlockMath,{math:"\\text{Mask} = \\text{MLP}\\!\\left(\\text{CrossAttn}(\\mathbf{Q}, \\mathbf{K}, \\mathbf{V})\\right) \\cdot \\mathbf{E}_{\\text{img}}"}),e.jsxs("p",{className:"mt-1",children:["The output is a dot product between updated prompt tokens and image embeddings, producing ",e.jsx(t.InlineMath,{math:"256 \\times 256"})," mask logits per prompt."]})]}),e.jsx(f,{title:"SAM Training Data (SA-1B)",children:e.jsxs("ul",{className:"list-disc ml-5 space-y-1",children:[e.jsx("li",{children:"11 million images with 1.1 billion masks"}),e.jsx("li",{children:"Three-stage annotation: assisted-manual, semi-automatic, fully automatic"}),e.jsx("li",{children:"99.1% mask quality measured by human evaluation"}),e.jsx("li",{children:"400x more masks than any previous segmentation dataset"})]})}),e.jsx(u,{title:"Using SAM for Segmentation",code:`import torch
from segment_anything import sam_model_registry, SamPredictor

# Load SAM model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

# Set image (encodes once with ViT)
image = load_image("photo.jpg")  # (H, W, 3) numpy
predictor.set_image(image)

# Point prompt: positive click at (x=500, y=300)
masks, scores, logits = predictor.predict(
    point_coords=torch.tensor([[500, 300]]),
    point_labels=torch.tensor([1]),  # 1=foreground
    multimask_output=True,  # Returns 3 mask candidates
)
best_mask = masks[scores.argmax()]  # Pick highest IoU

# Box prompt
masks_box, _, _ = predictor.predict(
    box=torch.tensor([100, 100, 400, 350]),  # xyxy
    multimask_output=False,
)

# Automatic mask generation (segment everything)
from segment_anything import SamAutomaticMaskGenerator
generator = SamAutomaticMaskGenerator(sam)
all_masks = generator.generate(image)
print(f"Found {len(all_masks)} segments")`}),e.jsx(y,{type:"note",title:"SAM 2 and Video Segmentation",children:e.jsx("p",{children:"SAM 2 extends the foundation model paradigm to video, tracking segments across frames with memory-based attention. It handles occlusions, appearance changes, and new objects appearing mid-video. The streaming architecture processes frames sequentially while maintaining a memory bank of past predictions and image features."})})]})}const ye=Object.freeze(Object.defineProperty({__proto__:null,default:U},Symbol.toStringTag,{value:"Module"}));function K(){const[n,m]=x.useState(0),l=260,c=240,a=n*Math.PI/180,r={head:[130,35],neck:[130,60],lShoulder:[100,70],rShoulder:[160,70],lElbow:[80+Math.sin(a)*15,105],rElbow:[180-Math.sin(a)*15,105],lWrist:[65+Math.sin(a)*25,140],rWrist:[195-Math.sin(a)*25,140],lHip:[110,140],rHip:[150,140],lKnee:[105,180],rKnee:[155,180],lAnkle:[100,215],rAnkle:[160,215]},i=[["head","neck"],["neck","lShoulder"],["neck","rShoulder"],["lShoulder","lElbow"],["rShoulder","rElbow"],["lElbow","lWrist"],["rElbow","rWrist"],["neck","lHip"],["neck","rHip"],["lHip","lKnee"],["rHip","rKnee"],["lKnee","lAnkle"],["rKnee","rAnkle"]];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"2D Pose Skeleton"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Arm angle:",e.jsx("input",{type:"range",min:-60,max:60,value:n,onChange:s=>m(parseInt(s.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("svg",{width:l,height:c,className:"mx-auto block",children:[i.map(([s,o],h)=>e.jsx("line",{x1:r[s][0],y1:r[s][1],x2:r[o][0],y2:r[o][1],stroke:"#8b5cf6",strokeWidth:2.5,strokeLinecap:"round"},h)),Object.values(r).map((s,o)=>e.jsx("circle",{cx:s[0],cy:s[1],r:4,fill:"#7c3aed",stroke:"white",strokeWidth:1.5},o))]})]})}function G(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"2D pose estimation localizes body keypoints (joints) in image coordinates. The two main approaches are heatmap-based regression and direct coordinate regression."}),e.jsxs(g,{title:"Heatmap-Based Pose Estimation",children:[e.jsxs("p",{children:["For each keypoint ",e.jsx(t.InlineMath,{math:"k"}),", predict a heatmap ",e.jsx(t.InlineMath,{math:"H_k \\in \\mathbb{R}^{h \\times w}"})," where the target is a 2D Gaussian centered at the ground truth location:"]}),e.jsx(t.BlockMath,{math:"H_k^*(x, y) = \\exp\\!\\left(-\\frac{(x - x_k)^2 + (y - y_k)^2}{2\\sigma^2}\\right)"}),e.jsxs("p",{className:"mt-2",children:["The predicted keypoint is at ",e.jsx(t.InlineMath,{math:"\\arg\\max_{x,y} H_k(x, y)"}),"."]})]}),e.jsx(K,{}),e.jsxs(j,{title:"Top-Down vs Bottom-Up",id:"pose-paradigm",children:[e.jsxs("p",{children:[e.jsx("strong",{children:"Top-down"}),": Detect persons first, then estimate pose per crop."]}),e.jsx(t.BlockMath,{math:"\\text{Detector} \\rightarrow \\text{Crop} \\rightarrow \\text{Pose Net} \\rightarrow K \\text{ keypoints}"}),e.jsxs("p",{className:"mt-2",children:[e.jsx("strong",{children:"Bottom-up"}),": Detect all keypoints first, then group into persons."]}),e.jsx(t.BlockMath,{math:"\\text{All Keypoints} \\xrightarrow{\\text{association}} \\text{Person Instances}"}),e.jsx("p",{className:"mt-1",children:"Top-down is more accurate; bottom-up is faster for multi-person scenes."})]}),e.jsxs(f,{title:"COCO Keypoint Format",children:[e.jsx("p",{children:"The COCO dataset defines 17 body keypoints. The evaluation metric OKS (Object Keypoint Similarity):"}),e.jsx(t.BlockMath,{math:"\\text{OKS} = \\frac{\\sum_k \\exp\\!\\left(-d_k^2 / (2s^2\\kappa_k^2)\\right) \\cdot \\delta(v_k > 0)}{\\sum_k \\delta(v_k > 0)}"}),e.jsxs("p",{className:"mt-1",children:["where ",e.jsx(t.InlineMath,{math:"d_k"})," is the Euclidean distance, ",e.jsx(t.InlineMath,{math:"s"})," is object scale, and ",e.jsx(t.InlineMath,{math:"\\kappa_k"})," is a per-keypoint constant."]})]}),e.jsx(u,{title:"HRNet Pose Estimation",code:`import torch
import torch.nn as nn

class SimpleHeatmapHead(nn.Module):
    """Heatmap prediction head for pose estimation."""
    def __init__(self, in_channels=256, num_keypoints=17):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(256, num_keypoints, 1)

    def forward(self, x):
        x = self.deconv(x)
        heatmaps = self.head(x)  # (B, K, H, W)
        return heatmaps

def decode_heatmaps(heatmaps):
    """Get keypoint coordinates from heatmaps."""
    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)
    max_idx = flat.argmax(dim=2)
    y = max_idx // W
    x = max_idx % W
    conf = flat.max(dim=2).values
    return torch.stack([x, y, conf], dim=2)  # (B, K, 3)

# Loss: MSE between predicted and target heatmaps
criterion = nn.MSELoss()
loss = criterion(pred_heatmaps, target_heatmaps)`}),e.jsx(y,{type:"note",title:"High-Resolution Networks (HRNet)",children:e.jsx("p",{children:"HRNet maintains high-resolution representations throughout the network by running parallel multi-resolution branches with repeated feature exchange. This avoids the information loss from downsampling-then-upsampling in encoder-decoder designs, achieving state-of-the-art results: 77.0 AP on COCO keypoint detection."})})]})}const je=Object.freeze(Object.defineProperty({__proto__:null,default:G},Symbol.toStringTag,{value:"Module"}));function $(){const[n,m]=x.useState(0),l=280,c=220,a=n*Math.PI/180,r=[[0,-.8,0],[0,-.5,0],[-.25,-.4,0],[.25,-.4,0],[-.45,-.1,.1],[.45,-.1,-.1],[0,0,0],[-.15,.3,0],[.15,.3,0],[-.15,.7,0],[.15,.7,0]],i=([o,h,d])=>{const p=o*Math.cos(a)-d*Math.sin(a);return[l/2+p*100,c/2+h*110]},s=[[0,1],[1,2],[1,3],[2,4],[3,5],[1,6],[6,7],[6,8],[7,9],[8,10]];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"3D Pose Rotation"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Y-axis rotation: ",n,"°",e.jsx("input",{type:"range",min:-90,max:90,value:n,onChange:o=>m(parseInt(o.target.value)),className:"w-40 accent-violet-500"})]}),e.jsxs("svg",{width:l,height:c,className:"mx-auto block",children:[s.map(([o,h],d)=>{const p=i(r[o]),b=i(r[h]);return e.jsx("line",{x1:p[0],y1:p[1],x2:b[0],y2:b[1],stroke:"#8b5cf6",strokeWidth:2.5,strokeLinecap:"round"},d)}),r.map((o,h)=>{const d=i(o);return e.jsx("circle",{cx:d[0],cy:d[1],r:4,fill:"#7c3aed",stroke:"white",strokeWidth:1.5},h)})]})]})}function J(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"3D pose estimation recovers the three-dimensional positions of body joints from images or video. The key challenge is the inherent depth ambiguity in 2D projections."}),e.jsxs(g,{title:"2D-to-3D Lifting",children:[e.jsxs("p",{children:["Given 2D keypoints ",e.jsx(t.InlineMath,{math:"p \\in \\mathbb{R}^{K \\times 2}"}),", predict 3D positions ",e.jsx(t.InlineMath,{math:"P \\in \\mathbb{R}^{K \\times 3}"})," via a lifting network:"]}),e.jsx(t.BlockMath,{math:"P = g_\\phi(p) \\quad \\text{where} \\quad g: \\mathbb{R}^{K \\times 2} \\rightarrow \\mathbb{R}^{K \\times 3}"}),e.jsx("p",{className:"mt-2",children:"The 2D-to-3D projection relationship under weak perspective:"}),e.jsx(t.BlockMath,{math:"p = \\Pi P = \\begin{bmatrix} f & 0 \\\\ 0 & f \\end{bmatrix} \\begin{bmatrix} X/Z \\\\ Y/Z \\end{bmatrix}"})]}),e.jsx($,{}),e.jsxs(j,{title:"Depth Ambiguity",id:"depth-ambiguity",children:[e.jsx("p",{children:"A single 2D projection has infinitely many 3D interpretations. The reprojection loss alone is insufficient:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_{\\text{reproj}} = \\sum_k \\|\\Pi(P_k) - p_k\\|^2"}),e.jsx("p",{className:"mt-1",children:"Structural priors (bone lengths, joint angle limits) and temporal consistency constrain the solution space:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = \\mathcal{L}_{\\text{3D}} + \\lambda_1 \\mathcal{L}_{\\text{bone}} + \\lambda_2 \\mathcal{L}_{\\text{smooth}}"})]}),e.jsx(f,{title:"3D Pose Approaches",children:e.jsxs("ul",{className:"list-disc ml-5 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Lifting"}),": SimpleBL lifts 2D detections with a residual MLP (MPJPE: 36.5mm)"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Volumetric"}),": Predict 3D heatmaps ",e.jsx(t.InlineMath,{math:"H \\in \\mathbb{R}^{D \\times H \\times W}"})]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Temporal"}),": VideoPose3D uses dilated convolutions over 2D pose sequences"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Multi-view"}),": Triangulate from calibrated cameras for ground truth"]})]})}),e.jsx(u,{title:"Simple 2D-to-3D Lifting Network",code:`import torch
import torch.nn as nn

class LiftingNet(nn.Module):
    """2D-to-3D lifting with residual blocks (SimpleBL)."""
    def __init__(self, num_joints=17, hidden=1024, num_blocks=2):
        super().__init__()
        self.input_proj = nn.Linear(num_joints * 2, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
            ) for _ in range(num_blocks)
        ])
        self.output = nn.Linear(hidden, num_joints * 3)

    def forward(self, x_2d):
        # x_2d: (B, 17, 2) -> flatten
        x = self.input_proj(x_2d.view(x_2d.shape[0], -1))
        for block in self.blocks:
            x = x + block(x)  # Residual connection
        out = self.output(x)
        return out.view(-1, 17, 3)  # (B, 17, 3)

# MPJPE loss (mean per-joint position error)
def mpjpe(pred, target):
    return torch.norm(pred - target, dim=-1).mean()

# Procrustes-aligned MPJPE (P-MPJPE)
def p_mpjpe(pred, target):
    """Align pred to target via Procrustes then compute MPJPE."""
    # Center both
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    tgt_c = target - target.mean(dim=1, keepdim=True)
    # SVD for optimal rotation
    U, S, V = torch.svd(tgt_c.transpose(1, 2) @ pred_c)
    R = V @ U.transpose(1, 2)
    return torch.norm(pred_c @ R.transpose(1, 2) - tgt_c, dim=-1).mean()`}),e.jsx(y,{type:"note",title:"Temporal Models",children:e.jsx("p",{children:"Processing 2D poses over time dramatically improves 3D accuracy. VideoPose3D uses temporal convolutions with receptive fields of 243 frames, reducing MPJPE from 52mm (single-frame) to 37mm. The temporal smoothness naturally resolves depth ambiguities and reduces jitter in predictions."})})]})}const be=Object.freeze(Object.defineProperty({__proto__:null,default:J},Symbol.toStringTag,{value:"Module"}));function Y(){const[n,m]=x.useState(0),l=240,c=220,a=[120,160],r=[{base:[85,140],mid:[75,110],tip:[70-n*.6,75]},{base:[100,125],mid:[95,85],tip:[92-n*.3,50]},{base:[118,120],mid:[118,78],tip:[118,42]},{base:[136,125],mid:[140,85],tip:[143+n*.3,50]},{base:[152,135],mid:[162,105],tip:[170+n*.6,75]}];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Hand Keypoint Tracking"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Finger spread:",e.jsx("input",{type:"range",min:0,max:30,value:n,onChange:i=>m(parseInt(i.target.value)),className:"w-36 accent-violet-500"})]}),e.jsxs("svg",{width:l,height:c,className:"mx-auto block",children:[e.jsx("circle",{cx:a[0],cy:a[1],r:22,fill:"#ede9fe",stroke:"#8b5cf6",strokeWidth:1.5}),r.map((i,s)=>e.jsxs("g",{children:[e.jsx("line",{x1:a[0],y1:a[1]-15,x2:i.base[0],y2:i.base[1],stroke:"#c4b5fd",strokeWidth:2}),e.jsx("line",{x1:i.base[0],y1:i.base[1],x2:i.mid[0],y2:i.mid[1],stroke:"#8b5cf6",strokeWidth:2}),e.jsx("line",{x1:i.mid[0],y1:i.mid[1],x2:i.tip[0],y2:i.tip[1],stroke:"#7c3aed",strokeWidth:2}),[i.base,i.mid,i.tip].map((o,h)=>e.jsx("circle",{cx:o[0],cy:o[1],r:3,fill:"#7c3aed",stroke:"white",strokeWidth:1},h))]},s))]})]})}function Q(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Dense pose estimation, hand tracking, and body mesh recovery go beyond sparse keypoints to produce detailed surface representations of the human body and hands."}),e.jsxs(g,{title:"SMPL Body Model",children:[e.jsxs("p",{children:["SMPL is a parametric body model controlled by pose ",e.jsx(t.InlineMath,{math:"\\theta \\in \\mathbb{R}^{72}"})," and shape ",e.jsx(t.InlineMath,{math:"\\beta \\in \\mathbb{R}^{10}"}),":"]}),e.jsx(t.BlockMath,{math:"M(\\beta, \\theta) = W(T_P(\\beta, \\theta), J(\\beta), \\theta, \\mathbf{W})"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"T_P"})," is the posed template, ",e.jsx(t.InlineMath,{math:"J"})," are joint locations, and ",e.jsx(t.InlineMath,{math:"\\mathbf{W}"})," are blend skinning weights. Output: 6890 vertices forming a triangulated mesh."]})]}),e.jsx(Y,{}),e.jsxs(j,{title:"MANO Hand Model",id:"mano",children:[e.jsxs("p",{children:["MANO parameterizes hand meshes with pose ",e.jsx(t.InlineMath,{math:"\\theta \\in \\mathbb{R}^{48}"})," (16 joints x 3) and shape ",e.jsx(t.InlineMath,{math:"\\beta \\in \\mathbb{R}^{10}"}),":"]}),e.jsx(t.BlockMath,{math:"V = \\bar{V} + B_S(\\beta) + B_P(\\theta)"}),e.jsxs("p",{className:"mt-1",children:["where ",e.jsx(t.InlineMath,{math:"\\bar{V}"})," is the mean hand, ",e.jsx(t.InlineMath,{math:"B_S"})," are shape blend shapes, and ",e.jsx(t.InlineMath,{math:"B_P"})," are pose-dependent correctives. The final mesh has 778 vertices and 21 joints."]})]}),e.jsxs(f,{title:"DensePose: Dense Correspondence",children:[e.jsx("p",{children:"DensePose maps every visible pixel to a UV coordinate on the body surface:"}),e.jsx(t.BlockMath,{math:"f: (x, y) \\rightarrow (I, U, V)"}),e.jsxs("ul",{className:"list-disc ml-5 mt-2 space-y-1",children:[e.jsxs("li",{children:[e.jsx(t.InlineMath,{math:"I"}),": body part index (1 of 24 parts)"]}),e.jsxs("li",{children:[e.jsx(t.InlineMath,{math:"(U, V) \\in [0, 1]^2"}),": surface coordinates within that part"]}),e.jsx("li",{children:"Enables pixel-level body surface correspondence across images"})]})]}),e.jsx(u,{title:"Body Mesh Recovery (HMR)",code:`import torch
import torch.nn as nn

class HMRHead(nn.Module):
    """Human Mesh Recovery regression head."""
    def __init__(self, feat_dim=2048):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )
        # SMPL parameters
        self.pose = nn.Linear(1024, 72)     # 24 joints * 3
        self.shape = nn.Linear(1024, 10)    # Shape coefficients
        self.cam = nn.Linear(1024, 3)       # Weak-perspective camera

    def forward(self, features):
        x = self.fc(features)
        return {
            'pose': self.pose(x),
            'shape': self.shape(x),
            'camera': self.cam(x),
        }

# Hand landmark detection (MediaPipe-style)
class HandLandmarkNet(nn.Module):
    def __init__(self, num_landmarks=21):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.regressor = nn.Linear(64, num_landmarks * 3)

    def forward(self, hand_crop):
        feat = self.backbone(hand_crop).flatten(1)
        landmarks = self.regressor(feat)
        return landmarks.view(-1, 21, 3)  # (B, 21, xyz)`}),e.jsx(y,{type:"note",title:"Whole-Body Estimation",children:e.jsx("p",{children:"Modern whole-body models like SMPL-X jointly model the body (22 joints), hands (30 joints), and face (3 jaw joints + expression). This enables capturing complete human behavior including gestures and facial expressions in a single forward pass. Applications span AR/VR, sign language recognition, and motion capture."})})]})}const _e=Object.freeze(Object.defineProperty({__proto__:null,default:Q},Symbol.toStringTag,{value:"Module"}));function X(){const[n,m]=x.useState(4),l=16,c=14,a=l*c,r=Math.floor(l/n),i=["#8b5cf6","#f97316","#22c55e","#ef4444","#3b82f6","#eab308","#ec4899","#14b8a6","#a855f7","#f59e0b","#10b981","#f43f5e","#6366f1","#d97706","#059669","#e11d48"];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Image Patch Embedding"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Patch size: ",n,"x",n,e.jsx("input",{type:"range",min:2,max:8,step:2,value:n,onChange:s=>m(parseInt(s.target.value)),className:"w-32 accent-violet-500"}),e.jsxs("span",{className:"ml-2",children:["= ",r*r," tokens"]})]}),e.jsx("svg",{width:a,height:a,className:"mx-auto block",children:Array.from({length:r}).map((s,o)=>Array.from({length:r}).map((h,d)=>e.jsx("rect",{x:d*n*c,y:o*n*c,width:n*c-1,height:n*c-1,fill:i[(o*r+d)%i.length],opacity:.3,stroke:i[(o*r+d)%i.length],strokeWidth:1.5,rx:2},`${o}-${d}`)))}),e.jsx("p",{className:"mt-2 text-center text-xs text-gray-500",children:"Each colored patch becomes a token via linear projection"})]})}function Z(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"The Vision Transformer (ViT) applies the transformer architecture directly to image patches, demonstrating that pure attention-based models can match or exceed CNNs for image classification."}),e.jsxs(g,{title:"ViT Architecture",children:[e.jsxs("p",{children:["An image ",e.jsx(t.InlineMath,{math:"x \\in \\mathbb{R}^{H \\times W \\times C}"})," is split into",e.jsx(t.InlineMath,{math:"N = HW/P^2"})," patches of size ",e.jsx(t.InlineMath,{math:"P \\times P"}),":"]}),e.jsx(t.BlockMath,{math:"z_0 = [\\mathbf{x}_{\\text{cls}};\\; x_1\\mathbf{E};\\; x_2\\mathbf{E};\\; \\ldots;\\; x_N\\mathbf{E}] + \\mathbf{E}_{\\text{pos}}"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"\\mathbf{E} \\in \\mathbb{R}^{P^2 C \\times D}"})," is the patch embedding projection and ",e.jsx(t.InlineMath,{math:"\\mathbf{E}_{\\text{pos}}"})," are learned positional embeddings."]})]}),e.jsx(X,{}),e.jsxs(j,{title:"Self-Attention Complexity",id:"vit-complexity",children:[e.jsxs("p",{children:["Self-attention over ",e.jsx(t.InlineMath,{math:"N"})," patch tokens has quadratic complexity:"]}),e.jsx(t.BlockMath,{math:"\\text{Attention}(Q, K, V) = \\text{softmax}\\!\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V"}),e.jsx(t.BlockMath,{math:"\\mathcal{O}(N^2 \\cdot D) = \\mathcal{O}\\!\\left(\\frac{H^2 W^2}{P^4} \\cdot D\\right)"}),e.jsxs("p",{className:"mt-1",children:["Larger patch sizes reduce sequence length but lose spatial resolution. ViT-B/16 uses ",e.jsx(t.InlineMath,{math:"P=16"})," yielding ",e.jsx(t.InlineMath,{math:"N=196"})," tokens for 224x224 images."]})]}),e.jsxs(f,{title:"ViT Model Variants",children:[e.jsxs("ul",{className:"list-disc ml-5 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"ViT-B/16"}),": 12 layers, 768 dim, 12 heads, 86M params (81.8% ImageNet)"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"ViT-L/16"}),": 24 layers, 1024 dim, 16 heads, 307M params (85.2%)"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"ViT-H/14"}),": 32 layers, 1280 dim, 16 heads, 632M params (88.6% w/ JFT)"]})]}),e.jsx("p",{className:"mt-1",children:"ViTs require large-scale pretraining (JFT-300M) or strong regularization to match CNNs on ImageNet alone."})]}),e.jsx(u,{title:"Vision Transformer from Scratch",code:`import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, D)
        return x + self.pos_embed

class ViT(nn.Module):
    def __init__(self, num_classes=1000, dim=768, depth=12,
                 heads=12, patch_size=16):
        super().__init__()
        self.embed = PatchEmbedding(patch_size=patch_size, dim=dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=0.1, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        return self.head(x[:, 0])  # CLS token output`}),e.jsx(y,{type:"note",title:"CNN vs Transformer Inductive Biases",children:e.jsx("p",{children:"CNNs have built-in translation equivariance and locality. ViTs lack these biases, learning spatial structure entirely from data. This makes ViTs more flexible but data-hungry. Hybrid approaches (CNN stem + transformer layers) combine the efficiency of convolutions at early stages with the global reasoning of attention."})})]})}const ke=Object.freeze(Object.defineProperty({__proto__:null,default:Z},Symbol.toStringTag,{value:"Module"}));function ee(){const[n,m]=x.useState(!1),l=8,c=4,a=28,r=l*a,i=n?c/2:0,s=(h,d)=>{const p=Math.floor((h+i)%l/c),b=Math.floor((d+i)%l/c);return p*2+b},o=["#8b5cf6","#f97316","#22c55e","#3b82f6"];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Swin Transformer Windows"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:[e.jsx("input",{type:"checkbox",checked:n,onChange:h=>m(h.target.checked),className:"accent-violet-500"}),"Apply window shift (SW-MSA)"]}),e.jsxs("svg",{width:r,height:r,className:"mx-auto block",children:[Array.from({length:l}).map((h,d)=>Array.from({length:l}).map((p,b)=>e.jsx("rect",{x:b*a,y:d*a,width:a-1,height:a-1,fill:o[s(d,b)],opacity:.25,stroke:o[s(d,b)],strokeWidth:1,rx:1},`${d}-${b}`))),!n&&Array.from({length:2}).map((h,d)=>e.jsxs("g",{children:[e.jsx("line",{x1:0,y1:(d+1)*c*a,x2:r,y2:(d+1)*c*a,stroke:"#374151",strokeWidth:2}),e.jsx("line",{x1:(d+1)*c*a,y1:0,x2:(d+1)*c*a,y2:r,stroke:"#374151",strokeWidth:2})]},d))]}),e.jsx("p",{className:"mt-2 text-center text-xs text-gray-500",children:n?"Shifted windows enable cross-window connections":"Regular non-overlapping windows"})]})}function te(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"DeiT brings data-efficient training to ViT through distillation, while Swin Transformer introduces hierarchical vision with shifted windows, making transformers practical for dense prediction tasks."}),e.jsxs(g,{title:"DeiT: Data-Efficient Image Transformer",children:[e.jsx("p",{children:"DeiT adds a distillation token alongside the CLS token to learn from a CNN teacher:"}),e.jsx(t.BlockMath,{math:"z_0 = [\\mathbf{x}_{\\text{cls}};\\; \\mathbf{x}_{\\text{dist}};\\; x_1\\mathbf{E};\\; \\ldots;\\; x_N\\mathbf{E}] + \\mathbf{E}_{\\text{pos}}"}),e.jsx("p",{className:"mt-2",children:"The distillation loss combines hard label and teacher supervision:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = \\frac{1}{2}\\mathcal{L}_{\\text{CE}}(y, \\psi(z_{\\text{cls}})) + \\frac{1}{2}\\mathcal{L}_{\\text{CE}}(y_t, \\psi(z_{\\text{dist}}))"})]}),e.jsx(ee,{}),e.jsxs(j,{title:"Shifted Window Attention",id:"swin-attention",children:[e.jsxs("p",{children:["Swin computes self-attention within local windows of size ",e.jsx(t.InlineMath,{math:"M \\times M"}),":"]}),e.jsx(t.BlockMath,{math:"\\Omega(\\text{W-MSA}) = 4hwC^2 + 2M^2hwC"}),e.jsx(t.BlockMath,{math:"\\Omega(\\text{Global MSA}) = 4hwC^2 + 2(hw)^2C"}),e.jsxs("p",{className:"mt-1",children:["Window attention is linear in image size (",e.jsx(t.InlineMath,{math:"hw"}),") vs quadratic for global attention. Shifted windows in alternating layers provide cross-window connections."]})]}),e.jsxs(f,{title:"Swin Hierarchical Stages",children:[e.jsxs("ul",{className:"list-disc ml-5 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Stage 1"}),": ",e.jsx(t.InlineMath,{math:"H/4 \\times W/4"}),", dim=96, 2 blocks"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Stage 2"}),": ",e.jsx(t.InlineMath,{math:"H/8 \\times W/8"}),", dim=192, 2 blocks (patch merge 2x2)"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Stage 3"}),": ",e.jsx(t.InlineMath,{math:"H/16 \\times W/16"}),", dim=384, 6 blocks"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Stage 4"}),": ",e.jsx(t.InlineMath,{math:"H/32 \\times W/32"}),", dim=768, 2 blocks"]})]}),e.jsx("p",{className:"mt-1",children:"This mimics the multi-scale structure of CNN backbones like ResNet."})]}),e.jsx(u,{title:"Swin Transformer Block",code:`import torch
import torch.nn as nn

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=7, num_heads=8):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        # Relative position bias
        self.rel_pos_bias = nn.Parameter(
            torch.zeros((2*window_size-1) * (2*window_size-1), num_heads))

    def forward(self, x):
        B, N, C = x.shape  # N = window_size^2
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        attn = (q @ k.transpose(-2, -1)) / (C // self.num_heads) ** 0.5
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=7, shift=False):
        super().__init__()
        self.shift = shift
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # W-MSA or SW-MSA
        x = x + self.mlp(self.norm2(x))
        return x`}),e.jsx(y,{type:"note",title:"DeiT Training Recipe",children:e.jsx("p",{children:"DeiT achieves 83.1% ImageNet accuracy training only on ImageNet-1K (no JFT) by using aggressive augmentation (RandAugment, CutMix, Mixup), regularization (stochastic depth, repeated augmentation), and knowledge distillation from a RegNetY teacher. This recipe made ViTs accessible to researchers without massive datasets."})})]})}const ve=Object.freeze(Object.defineProperty({__proto__:null,default:te},Symbol.toStringTag,{value:"Module"}));function se(){const[n,m]=x.useState(0),l=300,c=160,a=[{x:50,y:50,label:"q1"},{x:150,y:80,label:"q2"},{x:250,y:50,label:"q3"},{x:100,y:120,label:"q4"}],r=[{x:60,y:60,label:"gt1"},{x:240,y:55,label:"gt2"}],i=n>=1?[[0,0],[2,1]]:[];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Hungarian Matching"}),e.jsx("div",{className:"flex gap-2 mb-3",children:["Queries + GT","Bipartite Match"].map((s,o)=>e.jsx("button",{onClick:()=>m(o),className:`px-3 py-1 rounded text-sm ${n===o?"bg-violet-500 text-white":"bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300"}`,children:s},o))}),e.jsxs("svg",{width:l,height:c,className:"mx-auto block",children:[i.map(([s,o],h)=>e.jsx("line",{x1:a[s].x,y1:a[s].y,x2:r[o].x,y2:r[o].y,stroke:"#22c55e",strokeWidth:2,strokeDasharray:"4,2"},h)),a.map((s,o)=>{const h=i.some(([d])=>d===o);return e.jsxs("g",{children:[e.jsx("circle",{cx:s.x,cy:s.y,r:8,fill:h?"#8b5cf6":"#d1d5db",opacity:.8}),e.jsx("text",{x:s.x,y:s.y+3,textAnchor:"middle",fontSize:8,fill:"white",children:s.label})]},`p${o}`)}),r.map((s,o)=>e.jsxs("g",{children:[e.jsx("rect",{x:s.x-10,y:s.y-10,width:20,height:20,fill:"none",stroke:"#f97316",strokeWidth:2}),e.jsx("text",{x:s.x,y:s.y+25,textAnchor:"middle",fontSize:9,fill:"#f97316",children:s.label})]},`g${o}`)),e.jsx("text",{x:10,y:c-5,fontSize:9,fill:"#6b7280",children:n===0?"N queries, M ground truths":"Optimal 1-to-1 assignment (unmatched = no-object)"})]})]})}function ae(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"DETR (DEtection TRansformer) reformulates object detection as a direct set prediction problem, eliminating anchor boxes, NMS, and hand-designed components entirely."}),e.jsxs(g,{title:"DETR Architecture",children:[e.jsx("p",{children:"DETR processes images through a CNN backbone, transformer encoder-decoder, and prediction heads:"}),e.jsx(t.BlockMath,{math:"\\text{Image} \\xrightarrow{\\text{CNN}} \\mathbf{F} \\xrightarrow{\\text{Encoder}} \\mathbf{Z} \\xrightarrow[\\text{Object Queries}]{\\text{Decoder}} \\{(\\hat{c}_i, \\hat{b}_i)\\}_{i=1}^{N}"}),e.jsxs("p",{className:"mt-2",children:[e.jsx(t.InlineMath,{math:"N"})," learnable object queries attend to image features via cross-attention, each predicting a class ",e.jsx(t.InlineMath,{math:"\\hat{c}"})," and box ",e.jsx(t.InlineMath,{math:"\\hat{b}"}),' (or "no object").']})]}),e.jsx(se,{}),e.jsxs(j,{title:"Bipartite Matching Loss",id:"hungarian",children:[e.jsx("p",{children:"DETR finds the optimal one-to-one assignment between predictions and ground truth:"}),e.jsx(t.BlockMath,{math:"\\hat{\\sigma} = \\arg\\min_{\\sigma \\in \\mathfrak{S}_N} \\sum_{i=1}^{N} \\mathcal{L}_{\\text{match}}(y_i, \\hat{y}_{\\sigma(i)})"}),e.jsx("p",{className:"mt-1",children:"where the matching cost combines classification and box terms:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_{\\text{match}} = -\\mathbb{1}_{c_i \\neq \\varnothing}\\hat{p}_{\\sigma(i)}(c_i) + \\mathbb{1}_{c_i \\neq \\varnothing}\\mathcal{L}_{\\text{box}}(b_i, \\hat{b}_{\\sigma(i)})"}),e.jsxs("p",{className:"mt-1",children:["Solved efficiently using the Hungarian algorithm in ",e.jsx(t.InlineMath,{math:"\\mathcal{O}(N^3)"}),"."]})]}),e.jsx(f,{title:"Deformable DETR Improvements",children:e.jsxs("ul",{className:"list-disc ml-5 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Deformable attention"}),": attends to a small set of sampling points instead of all tokens"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Multi-scale features"}),": processes FPN features at multiple resolutions"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"10x faster convergence"}),": 50 epochs vs 500 for vanilla DETR"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Better small object detection"}),": multi-scale attention at high-res features"]})]})}),e.jsx(u,{title:"DETR-Style Detection",code:`import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

class DETRHead(nn.Module):
    def __init__(self, d_model=256, num_classes=91, num_queries=100):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=1024,
            batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.class_head = nn.Linear(d_model, num_classes + 1)  # +1 for no-obj
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 4),  # (cx, cy, w, h) normalized
        )

    def forward(self, encoder_output):
        B = encoder_output.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        hs = self.decoder(queries, encoder_output)
        return self.class_head(hs), self.box_head(hs).sigmoid()

def hungarian_loss(pred_cls, pred_box, gt_cls, gt_box):
    """Compute loss with Hungarian matching."""
    # Cost matrix: classification + L1 + GIoU
    cost_cls = -pred_cls.softmax(-1)[..., gt_cls]  # (N, M)
    cost_box = torch.cdist(pred_box, gt_box, p=1)
    cost = cost_cls + 5 * cost_box
    # Hungarian matching
    row_idx, col_idx = linear_sum_assignment(cost.detach().cpu())
    return row_idx, col_idx  # Matched indices`}),e.jsx(y,{type:"note",title:"End-to-End Detection",children:e.jsx("p",{children:"DETR's key insight is replacing hand-designed components (anchors, NMS, proposal generation) with learned set prediction. This simplifies the pipeline but originally required very long training (500 epochs). Deformable DETR, DAB-DETR, and DINO have progressively addressed convergence speed, achieving state-of-the-art results (63.3 AP on COCO) with practical training schedules."})})]})}const we=Object.freeze(Object.defineProperty({__proto__:null,default:ae},Symbol.toStringTag,{value:"Module"}));export{le as a,ce as b,de as c,he as d,me as e,xe as f,pe as g,ge as h,fe as i,ue as j,ye as k,je as l,be as m,_e as n,ke as o,ve as p,we as q,oe as s};

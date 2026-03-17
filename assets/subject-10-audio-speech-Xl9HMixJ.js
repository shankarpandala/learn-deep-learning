import{j as e,r as m}from"./vendor-DpISuAX6.js";import{r as t}from"./vendor-katex-CbWCYdth.js";import{D as x,E as g,P as u,N as f,T as b,W as j}from"./subject-01-foundations-D0A1VJsr.js";function _(){const[n,c]=m.useState(1024),[a,r]=m.useState(256),[s,o]=m.useState(!1),d=Math.floor(16e3/a),l=s?80:n/2;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Spectrogram Parameter Explorer"}),e.jsxs("div",{className:"flex flex-wrap items-center gap-4 mb-3",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Window: ",n,e.jsx("input",{type:"range",min:256,max:4096,step:256,value:n,onChange:i=>c(Number(i.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Hop: ",a,e.jsx("input",{type:"range",min:64,max:1024,step:64,value:a,onChange:i=>r(Number(i.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:[e.jsx("input",{type:"checkbox",checked:s,onChange:i=>o(i.target.checked),className:"accent-violet-500"}),"Mel scale"]})]}),e.jsxs("div",{className:"grid grid-cols-2 gap-4 text-sm text-gray-700 dark:text-gray-300",children:[e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3",children:[e.jsx("p",{className:"font-semibold text-violet-700 dark:text-violet-300",children:"Time frames (1s audio)"}),e.jsx("p",{className:"text-2xl font-bold text-violet-600",children:d})]}),e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3",children:[e.jsx("p",{className:"font-semibold text-violet-700 dark:text-violet-300",children:"Frequency bins"}),e.jsx("p",{className:"text-2xl font-bold text-violet-600",children:l})]})]}),e.jsxs("p",{className:"mt-2 text-xs text-gray-500 dark:text-gray-400",children:["Output shape: ",e.jsxs("strong",{children:["[",l,", ",d,"]"]})," — freq resolution: ",(16e3/n).toFixed(1)," Hz, time resolution: ",(a/16e3*1e3).toFixed(1)," ms"]})]})}function k(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Raw audio waveforms are high-dimensional time-domain signals. Spectrograms convert them into compact time-frequency representations that are far more effective as neural network inputs, revealing structure invisible in the waveform."}),e.jsxs(x,{title:"Short-Time Fourier Transform (STFT)",children:[e.jsx("p",{children:"The STFT decomposes a signal into overlapping windowed segments and applies the DFT to each:"}),e.jsx(t.BlockMath,{math:"X(t, f) = \\sum_{n=0}^{N-1} x[n + tH] \\cdot w[n] \\cdot e^{-j2\\pi fn/N}"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"w[n]"})," is the window function of length ",e.jsx(t.InlineMath,{math:"N"}),", and ",e.jsx(t.InlineMath,{math:"H"})," is the hop length. The ",e.jsx("strong",{children:"spectrogram"})," is ",e.jsx(t.InlineMath,{math:"|X(t,f)|^2"}),"."]})]}),e.jsxs(x,{title:"Mel Scale",children:[e.jsx("p",{children:"The mel scale maps frequencies to approximate human pitch perception:"}),e.jsx(t.BlockMath,{math:"m = 2595 \\log_{10}\\left(1 + \\frac{f}{700}\\right)"}),e.jsx("p",{className:"mt-2",children:"A mel spectrogram applies triangular filter banks spaced linearly on the mel scale, compressing high frequencies where human perception has lower resolution."})]}),e.jsx(_,{}),e.jsxs(g,{title:"Time-Frequency Trade-off",children:[e.jsxs("p",{children:["With a window of ",e.jsx(t.InlineMath,{math:"N = 1024"})," at 16 kHz sample rate, the frequency resolution is ",e.jsx(t.InlineMath,{math:"\\Delta f = 16000/1024 \\approx 15.6"})," Hz, but the time resolution is ",e.jsx(t.InlineMath,{math:"1024/16000 = 64"})," ms. Smaller windows improve temporal precision at the cost of frequency resolution:"]}),e.jsx(t.BlockMath,{math:"\\Delta f \\cdot \\Delta t \\geq \\frac{1}{4\\pi}"})]}),e.jsx(u,{title:"Computing Spectrograms with torchaudio",code:`import torch
import torchaudio
import torchaudio.transforms as T

# Load audio (16 kHz mono)
waveform, sr = torchaudio.load("speech.wav")
print(f"Waveform: {waveform.shape}")  # [1, num_samples]

# Standard spectrogram (STFT)
spectrogram = T.Spectrogram(n_fft=1024, hop_length=256)
spec = spectrogram(waveform)
print(f"Spectrogram: {spec.shape}")  # [1, 513, time_frames]

# Mel spectrogram (80 mel bins)
mel_spectrogram = T.MelSpectrogram(
    sample_rate=sr, n_fft=1024, hop_length=256, n_mels=80
)
mel_spec = mel_spectrogram(waveform)
print(f"Mel spectrogram: {mel_spec.shape}")  # [1, 80, time_frames]

# Log-mel spectrogram (standard input for speech models)
log_mel = torch.log(mel_spec.clamp(min=1e-9))
print(f"Log-mel: min={log_mel.min():.2f}, max={log_mel.max():.2f}")`}),e.jsx(f,{type:"note",title:"Why Log-Mel Spectrograms?",children:e.jsxs("p",{children:["Nearly all modern speech and audio models use ",e.jsx("strong",{children:"log-mel spectrograms"})," as input. The mel scale matches human perception, the log transform compresses dynamic range (mimicking the ear's logarithmic loudness response), and the 2D representation enables reuse of powerful vision architectures like CNNs and Vision Transformers."]})})]})}const te=Object.freeze(Object.defineProperty({__proto__:null,default:k},Symbol.toStringTag,{value:"Module"}));function N(){const[n,c]=m.useState(13),[a,r]=m.useState(40),[s,o]=m.useState(4),d=[{name:"Waveform",shape:"[T]",desc:"Raw audio signal"},{name:"STFT",shape:"[F, N]",desc:"Short-Time Fourier Transform"},{name:"Power Spectrum",shape:`[${a>64?512:256}, N]`,desc:"|STFT|^2"},{name:"Mel Filter Bank",shape:`[${a}, N]`,desc:`${a} triangular filters`},{name:"Log Mel",shape:`[${a}, N]`,desc:"Log compression"},{name:"DCT",shape:`[${n}, N]`,desc:`Keep first ${n} coefficients`}];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"MFCC Extraction Pipeline"}),e.jsxs("div",{className:"flex flex-wrap gap-4 mb-4",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Mel bins: ",a,e.jsx("input",{type:"range",min:20,max:128,step:4,value:a,onChange:l=>r(Number(l.target.value)),className:"w-24 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Coefficients: ",n,e.jsx("input",{type:"range",min:6,max:40,step:1,value:n,onChange:l=>c(Number(l.target.value)),className:"w-24 accent-violet-500"})]})]}),e.jsx("div",{className:"flex flex-wrap gap-2",children:d.map((l,i)=>e.jsxs("button",{onClick:()=>o(i),className:`rounded-lg px-3 py-2 text-xs font-medium transition-colors ${i===s?"bg-violet-600 text-white":"bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400"}`,children:[i+1,". ",l.name]},i))}),e.jsxs("div",{className:"mt-3 rounded-lg bg-violet-50 dark:bg-violet-900/20 p-4",children:[e.jsx("p",{className:"font-semibold text-violet-700 dark:text-violet-300",children:d[s].name}),e.jsx("p",{className:"text-sm text-gray-600 dark:text-gray-400",children:d[s].desc}),e.jsxs("p",{className:"text-sm font-mono mt-1 text-violet-600 dark:text-violet-400",children:["Shape: ",d[s].shape]})]})]})}function w(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Mel-Frequency Cepstral Coefficients (MFCCs) were the dominant audio feature for decades. While deep learning models now often learn features end-to-end, understanding MFCCs provides insight into perceptual audio processing."}),e.jsxs(x,{title:"MFCC Computation",children:[e.jsx("p",{children:"MFCCs apply the Discrete Cosine Transform to log-mel filter bank energies:"}),e.jsx(t.BlockMath,{math:"c_k = \\sum_{m=1}^{M} \\log(S_m) \\cos\\!\\left[\\frac{\\pi k(m - 0.5)}{M}\\right]"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"S_m"})," is the energy in the ",e.jsx(t.InlineMath,{math:"m"}),"-th mel filter bank, and we keep only the first ",e.jsx(t.InlineMath,{math:"K"})," coefficients (typically 13)."]})]}),e.jsx(N,{}),e.jsx(b,{title:"Decorrelation via DCT",id:"dct-decorrelation",children:e.jsxs("p",{children:["The DCT approximately decorrelates the log-mel features, acting as a compact representation of the spectral envelope. The first coefficient ",e.jsx(t.InlineMath,{math:"c_0"})," captures overall energy, while higher coefficients capture increasingly fine spectral detail. This decorrelation was critical for GMM-HMM systems with diagonal covariances."]})}),e.jsxs(g,{title:"Delta and Delta-Delta Features",children:[e.jsx("p",{children:"Standard practice appends first and second derivatives to capture dynamics:"}),e.jsx(t.BlockMath,{math:"\\Delta c_k[t] = \\frac{\\sum_{n=1}^{N} n(c_k[t+n] - c_k[t-n])}{2\\sum_{n=1}^{N} n^2}"}),e.jsx("p",{className:"mt-1",children:"This triples the feature dimension from 13 to 39, providing velocity and acceleration of spectral changes."})]}),e.jsx(u,{title:"Extracting MFCCs with torchaudio",code:`import torch
import torchaudio
import torchaudio.transforms as T

waveform, sr = torchaudio.load("speech.wav")

# MFCC extraction
mfcc_transform = T.MFCC(
    sample_rate=sr,
    n_mfcc=13,
    melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 40}
)
mfccs = mfcc_transform(waveform)
print(f"MFCCs shape: {mfccs.shape}")  # [1, 13, time_frames]

# Compute deltas and delta-deltas
deltas = torchaudio.functional.compute_deltas(mfccs)
delta_deltas = torchaudio.functional.compute_deltas(deltas)

# Stack: [1, 39, time_frames]
features = torch.cat([mfccs, deltas, delta_deltas], dim=1)
print(f"Full features: {features.shape}")

# Compare with log-mel (modern preference)
mel_spec = T.MelSpectrogram(sample_rate=sr, n_mels=80, hop_length=256)(waveform)
log_mel = torch.log(mel_spec.clamp(min=1e-9))
print(f"Log-mel shape: {log_mel.shape}")  # [1, 80, time_frames]`}),e.jsx(f,{type:"note",title:"MFCCs vs Log-Mel in Deep Learning",children:e.jsxs("p",{children:["Modern deep learning systems typically prefer ",e.jsx("strong",{children:"log-mel spectrograms"})," over MFCCs. The DCT step in MFCCs discards information that neural networks can learn to use. However, MFCCs remain relevant in low-resource settings and as a pedagogical bridge to understanding perceptual audio features."]})})]})}const se=Object.freeze(Object.defineProperty({__proto__:null,default:w},Symbol.toStringTag,{value:"Module"}));function M(){const[n,c]=m.useState("wav2vec2"),a={wav2vec2:{name:"wav2vec 2.0",input:"Raw waveform",dim:768,pretraining:"Contrastive + masked prediction",data:"960h LibriSpeech"},hubert:{name:"HuBERT",input:"Raw waveform",dim:768,pretraining:"Offline clustering + masked prediction",data:"960h LibriSpeech"},whisper:{name:"Whisper encoder",input:"Log-mel spectrogram",dim:1280,pretraining:"Supervised multitask",data:"680k hours web audio"},beats:{name:"BEATs",input:"Log-mel spectrogram",dim:768,pretraining:"Audio event tokenizer + masked prediction",data:"AudioSet"}},r=a[n];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Self-Supervised Audio Models"}),e.jsx("div",{className:"flex flex-wrap gap-2 mb-4",children:Object.entries(a).map(([s,o])=>e.jsx("button",{onClick:()=>c(s),className:`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${n===s?"bg-violet-600 text-white":"bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400"}`,children:o.name},s))}),e.jsxs("div",{className:"grid grid-cols-2 gap-3 text-sm",children:[e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3",children:[e.jsx("p",{className:"text-xs text-violet-600 dark:text-violet-400 font-semibold",children:"Input"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:r.input})]}),e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3",children:[e.jsx("p",{className:"text-xs text-violet-600 dark:text-violet-400 font-semibold",children:"Hidden dim"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:r.dim})]}),e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3",children:[e.jsx("p",{className:"text-xs text-violet-600 dark:text-violet-400 font-semibold",children:"Pre-training"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:r.pretraining})]}),e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3",children:[e.jsx("p",{className:"text-xs text-violet-600 dark:text-violet-400 font-semibold",children:"Training data"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:r.data})]})]})]})}function T(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Rather than hand-crafted spectral features, modern systems learn audio representations directly from raw waveforms or spectrograms using self-supervised pre-training, analogous to BERT and GPT in NLP."}),e.jsxs(x,{title:"Convolutional Feature Encoder",children:[e.jsx("p",{children:"Models like wav2vec 2.0 process raw waveforms with a multi-layer 1D CNN:"}),e.jsx(t.BlockMath,{math:"z_t = \\text{CNN}(x_{t \\cdot s : t \\cdot s + k})"}),e.jsxs("p",{className:"mt-2",children:["The encoder uses ",e.jsx(t.InlineMath,{math:"7"})," temporal convolution blocks with strides that downsample 16 kHz audio to 50 Hz (one vector every 20 ms), producing latent representations ",e.jsx(t.InlineMath,{math:"z_t \\in \\mathbb{R}^{512}"}),"."]})]}),e.jsx(M,{}),e.jsxs(g,{title:"wav2vec 2.0 Contrastive Loss",children:[e.jsx("p",{children:"During pre-training, masked positions are predicted via contrastive learning:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = -\\log \\frac{\\exp(\\text{sim}(c_t, q_t) / \\kappa)}{\\sum_{q' \\in Q_t} \\exp(\\text{sim}(c_t, q') / \\kappa)}"}),e.jsxs("p",{className:"mt-1",children:["where ",e.jsx(t.InlineMath,{math:"c_t"})," is the Transformer context output, ",e.jsx(t.InlineMath,{math:"q_t"})," is the quantized target, and ",e.jsx(t.InlineMath,{math:"Q_t"})," includes distractors from other masked positions."]})]}),e.jsx(u,{title:"Extracting Learned Features with HuggingFace",code:`import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# Load pre-trained wav2vec 2.0
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# Process raw audio (16kHz)
waveform = torch.randn(1, 16000)  # 1 second of audio
inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# CNN features: [batch, time_steps, 512]
cnn_features = outputs.extract_features
print(f"CNN features: {cnn_features.shape}")

# Transformer features: [batch, time_steps, 768]
hidden_states = outputs.last_hidden_state
print(f"Hidden states: {hidden_states.shape}")

# Use as features for downstream tasks
# e.g., add a classification head for speaker ID
classifier = torch.nn.Linear(768, 100)  # 100 speakers
pooled = hidden_states.mean(dim=1)       # mean pooling
logits = classifier(pooled)
print(f"Speaker logits: {logits.shape}")`}),e.jsx(j,{title:"Computational Cost",children:e.jsx("p",{children:"Self-supervised audio models are computationally expensive. wav2vec 2.0 Base has 95M parameters, and processing a 10-second clip requires significant GPU memory. For resource-constrained settings, distilled models or log-mel features remain practical."})}),e.jsx(f,{type:"note",title:"Feature Extraction vs Fine-tuning",children:e.jsxs("p",{children:["Pre-trained audio models can be used in two modes: ",e.jsx("strong",{children:"frozen feature extraction"})," (fast, good for small datasets) or ",e.jsx("strong",{children:"full fine-tuning"})," (better performance, needs more data). Intermediate approaches like fine-tuning only the top layers offer a balance."]})})]})}const ae=Object.freeze(Object.defineProperty({__proto__:null,default:T},Symbol.toStringTag,{value:"Module"}));function S(){const[n,c]=m.useState(!0),a="cat",r=[{path:["-","c","c","a","-","t","-"],collapsed:"cat"},{path:["c","-","a","a","a","t","-"],collapsed:"cat"},{path:["-","-","c","a","t","-","-"],collapsed:"cat"},{path:["c","a","-","-","-","t","t"],collapsed:"cat"}];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"CTC Alignment Paths"}),e.jsxs("p",{className:"text-sm text-gray-500 dark:text-gray-400 mb-3",children:['Multiple alignments collapse to the same output "',a,'"']}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:[e.jsx("input",{type:"checkbox",checked:n,onChange:s=>c(s.target.checked),className:"accent-violet-500"}),"Highlight blank tokens"]}),e.jsx("div",{className:"space-y-2",children:r.map((s,o)=>e.jsxs("div",{className:"flex items-center gap-1",children:[e.jsxs("span",{className:"text-xs text-gray-400 w-6",children:["#",o+1]}),s.path.map((d,l)=>e.jsx("span",{className:`w-8 h-8 flex items-center justify-center rounded text-sm font-mono font-bold
                ${d==="-"?n?"bg-violet-100 text-violet-400 dark:bg-violet-900/30 dark:text-violet-500":"bg-gray-100 text-gray-400 dark:bg-gray-800":"bg-violet-500 text-white"}`,children:d==="-"?"ε":d},l)),e.jsxs("span",{className:"text-sm text-gray-500 ml-2",children:['→ "',s.collapsed,'"']})]},o))})]})}function C(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Connectionist Temporal Classification (CTC) enables training sequence models without explicit alignment between input and output. It sums over all valid alignments, making it the foundation of modern end-to-end ASR systems."}),e.jsxs(x,{title:"CTC Loss",children:[e.jsxs("p",{children:["Given input sequence ",e.jsx(t.InlineMath,{math:"X"})," of length ",e.jsx(t.InlineMath,{math:"T"})," and target label sequence ",e.jsx(t.InlineMath,{math:"Y"}),", CTC defines:"]}),e.jsx(t.BlockMath,{math:"P(Y|X) = \\sum_{\\pi \\in \\mathcal{B}^{-1}(Y)} \\prod_{t=1}^{T} P(\\pi_t | X)"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"\\mathcal{B}"})," is the collapsing function that removes blanks (",e.jsx(t.InlineMath,{math:"\\epsilon"}),") and merges repeated characters. The loss is ",e.jsx(t.InlineMath,{math:"\\mathcal{L} = -\\log P(Y|X)"}),"."]})]}),e.jsx(S,{}),e.jsxs(b,{title:"Forward-Backward Algorithm",id:"ctc-forward-backward",children:[e.jsxs("p",{children:["Computing the CTC loss exactly via enumeration is intractable. The forward variable",e.jsx(t.InlineMath,{math:"\\alpha(t, s)"})," gives the probability of emitting the first ",e.jsx(t.InlineMath,{math:"s"})," labels in ",e.jsx(t.InlineMath,{math:"t"})," steps:"]}),e.jsx(t.BlockMath,{math:"\\alpha(t, s) = [\\alpha(t{-}1, s) + \\alpha(t{-}1, s{-}1) + \\alpha(t{-}1, s{-}2)] \\cdot P(l'_s | x_t)"}),e.jsxs("p",{className:"mt-1",children:["The third term is included only if ",e.jsx(t.InlineMath,{math:"l'_s \\neq \\epsilon"})," and",e.jsx(t.InlineMath,{math:"l'_s \\neq l'_{s-2}"}),". This runs in ",e.jsx(t.InlineMath,{math:"O(T \\cdot |Y'|)"})," time."]})]}),e.jsxs(g,{title:"CTC Decoding Strategies",children:[e.jsxs("p",{children:[e.jsx("strong",{children:"Greedy decoding:"})," Take ",e.jsx(t.InlineMath,{math:"\\arg\\max"})," at each timestep, then collapse."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Beam search:"})," Maintain top-k paths, merging those that collapse to the same output."]}),e.jsx("p",{children:e.jsx("strong",{children:"With language model:"})}),e.jsx(t.BlockMath,{math:"\\hat{Y} = \\arg\\max_Y \\log P_\\text{CTC}(Y|X) + \\lambda \\log P_\\text{LM}(Y) + \\beta |Y|"})]}),e.jsx(u,{title:"CTC Loss in PyTorch",code:`import torch
import torch.nn as nn

# Simulated ASR output
T, B, C = 50, 2, 29  # time, batch, vocab (26 chars + space + apostrophe + blank)
log_probs = torch.randn(T, B, C).log_softmax(dim=2)

# Targets (variable length)
targets = torch.tensor([3, 1, 20, 0, 8, 5, 12, 12, 15])  # "cat hello"
target_lengths = torch.tensor([3, 5])  # "cat" and "hello"
input_lengths = torch.tensor([T, T])

# CTC loss
ctc_loss = nn.CTCLoss(blank=28, zero_infinity=True)
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
print(f"CTC loss: {loss.item():.4f}")

# Greedy decoding
predictions = log_probs[:, 0, :].argmax(dim=-1)
# Collapse: remove blanks and repeated characters
collapsed = []
prev = -1
for p in predictions:
    if p != 28 and p != prev:
        collapsed.append(p.item())
    prev = p
print(f"Decoded indices: {collapsed}")`}),e.jsx(f,{type:"note",title:"CTC Assumptions & Limitations",children:e.jsxs("p",{children:["CTC assumes ",e.jsx("strong",{children:"conditional independence"})," between output tokens at each timestep, which limits its ability to model language structure. This is why CTC is often combined with an external language model or used alongside attention-based decoders in hybrid systems."]})})]})}const ne=Object.freeze(Object.defineProperty({__proto__:null,default:C},Symbol.toStringTag,{value:"Module"}));function A(){const[n,c]=m.useState("las"),a={las:{name:"Listen, Attend, Spell",encoder:"Pyramidal BiLSTM",decoder:"LSTM + attention",alignment:"Soft attention",strengths:"Strong language modeling",weaknesses:"Slow autoregressive decoding"},rnnt:{name:"RNN-Transducer",encoder:"LSTM / Conformer",decoder:"Prediction network (LSTM)",alignment:"RNN-T loss (CTC-like)",strengths:"Streaming capable",weaknesses:"Complex training"},transformer:{name:"Transformer ASR",encoder:"Conformer / Transformer",decoder:"Transformer decoder",alignment:"Cross-attention",strengths:"Best offline accuracy",weaknesses:"High memory, non-streaming"}},r=a[n];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"ASR Architecture Comparison"}),e.jsx("div",{className:"flex flex-wrap gap-2 mb-4",children:Object.entries(a).map(([s,o])=>e.jsx("button",{onClick:()=>c(s),className:`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${n===s?"bg-violet-600 text-white":"bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400"}`,children:o.name},s))}),e.jsx("div",{className:"grid grid-cols-2 gap-3 text-sm",children:[["Encoder",r.encoder],["Decoder",r.decoder],["Alignment",r.alignment],["Strengths",r.strengths],["Weaknesses",r.weaknesses]].map(([s,o])=>e.jsxs("div",{className:`rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3 ${s==="Weaknesses"?"col-span-2 sm:col-span-1":""}`,children:[e.jsx("p",{className:"text-xs text-violet-600 dark:text-violet-400 font-semibold",children:s}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:o})]},s))})]})}function z(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Attention-based models replace CTC's conditional independence assumption with an autoregressive decoder that attends to encoder outputs, enabling the model to learn implicit language modeling jointly with acoustic modeling."}),e.jsxs(x,{title:"Listen, Attend and Spell (LAS)",children:[e.jsx("p",{children:"LAS consists of three components:"}),e.jsx(t.BlockMath,{math:"h = \\text{Encoder}(X), \\quad c_i = \\text{Attention}(s_{i-1}, h), \\quad y_i = \\text{Decoder}(s_{i-1}, y_{i-1}, c_i)"}),e.jsxs("p",{className:"mt-2",children:["The encoder (listener) processes audio features, attention computes a context vector ",e.jsx(t.InlineMath,{math:"c_i"}),", and the decoder (speller) generates tokens autoregressively."]})]}),e.jsx(A,{}),e.jsxs(g,{title:"Conformer: Convolution-Augmented Transformer",children:[e.jsx("p",{children:"The Conformer block combines self-attention with depthwise convolutions:"}),e.jsx(t.BlockMath,{math:"y = x + \\tfrac{1}{2}\\text{FFN}(x) + \\text{MHSA}(x) + \\text{Conv}(x) + \\tfrac{1}{2}\\text{FFN}(x)"}),e.jsx("p",{className:"mt-1",children:"This captures both global context (via attention) and local patterns (via convolution), achieving state-of-the-art results on LibriSpeech with a 1.9% WER."})]}),e.jsx(u,{title:"Conformer-based ASR with torchaudio",code:`import torch
import torchaudio

# Conformer encoder (available in torchaudio)
conformer = torchaudio.models.Conformer(
    input_dim=80,         # mel features
    num_heads=4,
    ffn_dim=256,
    num_layers=8,
    depthwise_conv_kernel_size=31,
)

# Simulated log-mel input: [batch, time, features]
features = torch.randn(2, 200, 80)
lengths = torch.tensor([200, 180])

# Encode
encoded, out_lengths = conformer(features, lengths)
print(f"Encoder output: {encoded.shape}")  # [2, 200, 80]

# Simple CTC head on top of conformer
ctc_head = torch.nn.Linear(80, 29)  # vocab size
logits = ctc_head(encoded)
log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)  # [T, B, C]
print(f"CTC logits: {log_probs.shape}")`}),e.jsx(j,{title:"Attention Failures in Long Audio",children:e.jsx("p",{children:"Pure attention-based ASR can fail on very long utterances because the attention mechanism may not learn a monotonic left-to-right alignment. Solutions include monotonic attention constraints, CTC-attention joint training, or chunked processing."})}),e.jsx(f,{type:"note",title:"CTC-Attention Hybrid",children:e.jsxs("p",{children:["Modern ASR systems often combine CTC and attention losses:",e.jsx(t.InlineMath,{math:"\\mathcal{L} = \\lambda \\mathcal{L}_\\text{CTC} + (1-\\lambda)\\mathcal{L}_\\text{attention}"}),". The CTC loss enforces monotonic alignment as a regularizer, while the attention decoder provides superior language modeling. This is the default approach in ESPnet and other toolkits."]})})]})}const re=Object.freeze(Object.defineProperty({__proto__:null,default:z},Symbol.toStringTag,{value:"Module"}));function L(){const[n,c]=m.useState("base"),a={tiny:{params:"39M",layers:"4+4",dim:384,heads:6,englishWER:"7.6%",multiWER:"14.2%"},base:{params:"74M",layers:"6+6",dim:512,heads:8,englishWER:"5.0%",multiWER:"10.5%"},small:{params:"244M",layers:"12+12",dim:768,heads:12,englishWER:"3.4%",multiWER:"7.6%"},medium:{params:"769M",layers:"24+24",dim:1024,heads:16,englishWER:"2.9%",multiWER:"5.8%"},large:{params:"1550M",layers:"32+32",dim:1280,heads:20,englishWER:"2.7%",multiWER:"4.2%"}},r=a[n];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Whisper Model Variants"}),e.jsx("div",{className:"flex flex-wrap gap-2 mb-4",children:Object.keys(a).map(s=>e.jsx("button",{onClick:()=>c(s),className:`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors capitalize ${n===s?"bg-violet-600 text-white":"bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400"}`,children:s},s))}),e.jsx("div",{className:"grid grid-cols-3 gap-3 text-sm",children:[["Parameters",r.params],["Layers (enc+dec)",r.layers],["Hidden dim",r.dim],["Attention heads",r.heads],["English WER",r.englishWER],["Multilingual WER",r.multiWER]].map(([s,o])=>e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3",children:[e.jsx("p",{className:"text-xs text-violet-600 dark:text-violet-400 font-semibold",children:s}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300 font-bold",children:o})]},s))})]})}function E(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Whisper demonstrates that scaling weakly supervised training data to 680,000 hours produces remarkably robust speech recognition without self-supervised pre-training, approaching human-level performance across languages and acoustic conditions."}),e.jsxs(x,{title:"Whisper Architecture",children:[e.jsxs("p",{children:["Whisper uses a standard encoder-decoder Transformer. The encoder processes 30-second log-mel spectrogram chunks (",e.jsx(t.InlineMath,{math:"80 \\times 3000"})," frames), and the decoder generates text tokens autoregressively:"]}),e.jsx(t.BlockMath,{math:"P(y_1, \\ldots, y_N | X) = \\prod_{i=1}^{N} P(y_i | y_{<i}, \\text{Enc}(X))"}),e.jsxs("p",{className:"mt-2",children:["Special tokens encode the task: ",e.jsx("code",{children:"<|language|>"}),", ",e.jsx("code",{children:"<|transcribe|>"})," or",e.jsx("code",{children:"<|translate|>"}),", and ",e.jsx("code",{children:"<|timestamps|>"}),"."]})]}),e.jsx(L,{}),e.jsxs(b,{title:"Robustness Through Diversity",id:"whisper-robustness",children:[e.jsx("p",{children:"Whisper achieves robustness without domain-specific fine-tuning by training on diverse internet audio. On out-of-distribution benchmarks, Whisper's effective error rate decreases where fine-tuned models degrade:"}),e.jsx(t.BlockMath,{math:"\\text{WER}_{\\text{OOD}} \\propto \\frac{1}{\\sqrt{|\\mathcal{D}_{\\text{train}}|}}"}),e.jsx("p",{className:"mt-1",children:"This scaling suggests that data diversity, not just quantity, drives generalization."})]}),e.jsxs(g,{title:"Multitask Training Format",children:[e.jsx("p",{children:"Whisper's decoder handles multiple tasks via prompt tokens:"}),e.jsx("p",{className:"font-mono text-sm mt-2 bg-gray-100 dark:bg-gray-800 p-2 rounded",children:"<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|> Hello world <|endoftext|>"}),e.jsxs("p",{className:"mt-2",children:["For translation: replace ",e.jsx("code",{children:"<|transcribe|>"})," with ",e.jsx("code",{children:"<|translate|>"})," to translate any language to English."]})]}),e.jsx(u,{title:"Using Whisper for Speech Recognition",code:`import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load Whisper model
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

# Simulated 30s audio at 16kHz
audio = torch.randn(16000 * 30)

# Process audio to log-mel spectrogram
input_features = processor(
    audio.numpy(), sampling_rate=16000, return_tensors="pt"
).input_features
print(f"Input features: {input_features.shape}")  # [1, 80, 3000]

# Transcribe
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="en", task="transcribe"
)
generated_ids = model.generate(
    input_features, forced_decoder_ids=forced_decoder_ids
)
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(f"Transcription: {transcription[0]}")`}),e.jsx(f,{type:"note",title:"Beyond Whisper: Universal Speech Models",children:e.jsxs("p",{children:["Following Whisper, models like USM (Google) and MMS (Meta) scale to 1000+ languages. The trend is toward ",e.jsx("strong",{children:"universal speech foundation models"})," that handle ASR, translation, language ID, and speaker tasks in a single architecture, trained on millions of hours of diverse audio data."]})})]})}const ie=Object.freeze(Object.defineProperty({__proto__:null,default:E},Symbol.toStringTag,{value:"Module"}));function F(){const[n,c]=m.useState(0),a=[{name:"Text Input",desc:"Raw text or phoneme sequence",output:'"Hello world" or /h ə l oʊ w ɜːr l d/',color:"bg-gray-100 dark:bg-gray-800"},{name:"Encoder",desc:"Character/phoneme embeddings + Transformer/LSTM",output:"Hidden states [seq_len, 512]",color:"bg-violet-100 dark:bg-violet-900/30"},{name:"Attention",desc:"Location-sensitive attention for monotonic alignment",output:"Context vector per mel frame",color:"bg-violet-200 dark:bg-violet-900/40"},{name:"Decoder",desc:"Autoregressive mel prediction (2 frames/step)",output:"Mel spectrogram [80, T]",color:"bg-violet-300 dark:bg-violet-800/40"},{name:"Vocoder",desc:"WaveNet / HiFi-GAN converts mel to waveform",output:"Audio waveform [1, T*256]",color:"bg-violet-400 dark:bg-violet-700/40"}];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Tacotron 2 Pipeline"}),e.jsx("div",{className:"flex flex-wrap gap-2 mb-4",children:a.map((r,s)=>e.jsxs("button",{onClick:()=>c(s),className:`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${s===n?"bg-violet-600 text-white":"bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400"}`,children:[s+1,". ",r.name]},s))}),e.jsxs("div",{className:`rounded-lg p-4 ${a[n].color}`,children:[e.jsx("p",{className:"font-semibold text-violet-700 dark:text-violet-300",children:a[n].name}),e.jsx("p",{className:"text-sm text-gray-600 dark:text-gray-400 mt-1",children:a[n].desc}),e.jsx("p",{className:"text-sm font-mono mt-2 text-violet-600 dark:text-violet-400",children:a[n].output})]})]})}function I(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Tacotron introduced end-to-end neural TTS by predicting mel spectrograms from text, replacing complex traditional pipelines. Combined with neural vocoders, it produces near-human quality speech synthesis."}),e.jsxs(x,{title:"Tacotron 2 Architecture",children:[e.jsx("p",{children:"Tacotron 2 uses an encoder-decoder with location-sensitive attention:"}),e.jsx(t.BlockMath,{math:"\\text{mel}_t = \\text{Decoder}(s_{t-1}, \\text{mel}_{t-1}, c_t)"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"c_t = \\text{Attention}(s_{t-1}, h, \\alpha_{t-1})"})," uses previous alignment weights ",e.jsx(t.InlineMath,{math:"\\alpha_{t-1}"})," to encourage monotonic progression. The decoder predicts 2 mel frames per step with a stop token."]})]}),e.jsx(F,{}),e.jsxs(g,{title:"Tacotron 2 Loss",children:[e.jsx("p",{children:"Training minimizes the MSE on mel spectrograms plus a binary cross-entropy stop token loss:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = \\frac{1}{T}\\sum_{t=1}^{T} \\|\\hat{m}_t - m_t\\|^2 + \\lambda \\text{BCE}(\\hat{p}_t^{\\text{stop}}, p_t^{\\text{stop}})"}),e.jsx("p",{className:"mt-1",children:"The model also uses a post-net (5-layer CNN) that predicts a residual to refine the mel output."})]}),e.jsxs(x,{title:"Neural Vocoders",children:[e.jsx("p",{children:"Vocoders convert mel spectrograms to audio waveforms. Key architectures include:"}),e.jsxs("p",{className:"mt-2",children:[e.jsx("strong",{children:"Griffin-Lim:"})," Iterative phase reconstruction (fast but low quality)"]}),e.jsxs("p",{children:[e.jsx("strong",{children:"WaveNet vocoder:"})," Autoregressive, high quality, very slow"]}),e.jsxs("p",{children:[e.jsx("strong",{children:"HiFi-GAN:"})," GAN-based, real-time, near-WaveNet quality"]}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_\\text{HiFi-GAN} = \\mathcal{L}_\\text{adv} + \\lambda_\\text{fm}\\mathcal{L}_\\text{feature} + \\lambda_\\text{mel}\\mathcal{L}_\\text{mel}"})]}),e.jsx(u,{title:"TTS Inference with Tacotron 2 + HiFi-GAN",code:`import torch

# Tacotron 2 model (simplified structure)
class SimpleTacotron2(torch.nn.Module):
    def __init__(self, vocab_size=80, mel_dim=80, hidden=512):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden)
        self.encoder = torch.nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.decoder = torch.nn.LSTMCell(mel_dim + hidden, hidden)
        self.mel_proj = torch.nn.Linear(hidden, mel_dim)
        self.stop_proj = torch.nn.Linear(hidden, 1)

    def forward(self, text_ids, max_steps=200):
        enc_out, _ = self.encoder(self.embedding(text_ids))
        # Simplified: use mean context (real model uses attention)
        context = enc_out.mean(dim=1)
        mel_input = torch.zeros(text_ids.size(0), 80)
        h = torch.zeros(text_ids.size(0), 512)
        c = torch.zeros_like(h)
        mels = []
        for _ in range(max_steps):
            h, c = self.decoder(torch.cat([mel_input, context], -1), (h, c))
            mel_frame = self.mel_proj(h)
            mels.append(mel_frame)
            mel_input = mel_frame
        return torch.stack(mels, dim=1)  # [B, T, 80]

model = SimpleTacotron2()
text = torch.randint(0, 80, (1, 20))
mel_out = model(text)
print(f"Generated mel: {mel_out.shape}")  # [1, 200, 80]`}),e.jsx(j,{title:"Attention Alignment Issues",children:e.jsx("p",{children:"Tacotron's attention mechanism can fail to learn proper alignment, causing repeated words, skipped phrases, or babbling. Training tricks include guided attention loss, pre-trained aligners (like Montreal Forced Aligner), and replacing attention with duration predictors (as in FastSpeech)."})}),e.jsx(f,{type:"note",title:"FastSpeech: Non-Autoregressive TTS",children:e.jsx("p",{children:"FastSpeech replaces autoregressive decoding and attention with a duration predictor, enabling parallel mel generation. This is 100-300x faster than Tacotron 2 and avoids alignment failures. FastSpeech 2 adds pitch and energy predictors for better prosody."})})]})}const oe=Object.freeze(Object.defineProperty({__proto__:null,default:I},Symbol.toStringTag,{value:"Module"}));function R(){const[n,c]=m.useState(0),a=Math.pow(2,n),r=Math.pow(2,n+1)-1,s=400,o=160,d=s/16,l=4,i=o/(l+1);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Causal Dilated Convolutions"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Layer: ",n," (dilation=",a,")",e.jsx("input",{type:"range",min:0,max:3,step:1,value:n,onChange:h=>c(Number(h.target.value)),className:"w-32 accent-violet-500"})]}),e.jsxs("span",{className:"text-sm text-violet-600 dark:text-violet-400 font-semibold",children:["Receptive field: ",r," samples"]})]}),e.jsxs("svg",{width:s,height:o,className:"mx-auto block",children:[Array.from({length:16}).map((h,p)=>e.jsxs("g",{children:[e.jsx("circle",{cx:p*d+d/2,cy:o-i,r:4,fill:"#d1d5db"}),p+a<16&&e.jsx("line",{x1:p*d+d/2,y1:o-i,x2:(p+a)*d+d/2,y2:o-2*i,stroke:"#8b5cf6",strokeWidth:1.5,opacity:.6}),p-a>=0&&p<16&&e.jsx("line",{x1:p*d+d/2,y1:o-i,x2:p*d+d/2,y2:o-2*i,stroke:"#8b5cf6",strokeWidth:1.5,opacity:.6}),e.jsx("circle",{cx:p*d+d/2,cy:o-2*i,r:4,fill:"#8b5cf6"})]},p)),e.jsx("text",{x:5,y:o-i+4,fontSize:10,fill:"#9ca3af",children:"Input"}),e.jsxs("text",{x:5,y:o-2*i+4,fontSize:10,fill:"#8b5cf6",children:["Layer ",n]})]}),e.jsxs("p",{className:"text-xs text-center text-gray-500 mt-2",children:["After ",l," layers with doubling dilation: receptive field = ",Math.pow(2,l+1)-1," samples"]})]})}function q(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"WaveNet generates raw audio waveforms sample-by-sample using causal dilated convolutions, producing speech quality that was indistinguishable from human recordings and revolutionizing both TTS and generative audio modeling."}),e.jsxs(x,{title:"WaveNet Autoregressive Model",children:[e.jsx("p",{children:"WaveNet models the joint probability of a waveform as:"}),e.jsx(t.BlockMath,{math:"P(x) = \\prod_{t=1}^{T} P(x_t | x_1, \\ldots, x_{t-1})"}),e.jsx("p",{className:"mt-2",children:"Each sample is predicted using a stack of causal dilated convolutions with gated activations. At 16 kHz, this means generating 16,000 samples per second of audio."})]}),e.jsx(R,{}),e.jsxs(b,{title:"Exponential Receptive Field Growth",id:"dilated-receptive-field",children:[e.jsxs("p",{children:["With ",e.jsx(t.InlineMath,{math:"L"})," layers of dilation rates ",e.jsx(t.InlineMath,{math:"1, 2, 4, \\ldots, 2^{L-1}"}),", the receptive field grows exponentially:"]}),e.jsx(t.BlockMath,{math:"R = 2^L - 1 + (k - 1)(2^L - 1)"}),e.jsxs("p",{className:"mt-1",children:["where ",e.jsx(t.InlineMath,{math:"k"})," is the kernel size. With 10 layers repeated 3 times (30 layers total) and ",e.jsx(t.InlineMath,{math:"k=2"}),", this covers ",e.jsx(t.InlineMath,{math:"\\sim 300"})," ms at 16 kHz."]})]}),e.jsxs(g,{title:"Gated Activation Unit",children:[e.jsx("p",{children:"Each WaveNet layer uses a gated activation inspired by LSTMs:"}),e.jsx(t.BlockMath,{math:"z = \\tanh(W_f * x + V_f * h) \\odot \\sigma(W_g * x + V_g * h)"}),e.jsxs("p",{className:"mt-1",children:["where ",e.jsx(t.InlineMath,{math:"*"})," denotes dilated convolution, ",e.jsx(t.InlineMath,{math:"h"})," is the conditioning input (e.g., mel spectrogram or speaker embedding), and ",e.jsx(t.InlineMath,{math:"\\odot"})," is element-wise multiplication. Skip connections from each layer feed into the output."]})]}),e.jsx(u,{title:"WaveNet Causal Dilated Conv Block",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveNetBlock(nn.Module):
    def __init__(self, channels=64, kernel_size=2, dilation=1):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            channels, 2 * channels, kernel_size,
            padding=dilation * (kernel_size - 1),  # causal padding
            dilation=dilation
        )
        self.cond_proj = nn.Conv1d(80, 2 * channels, 1)  # mel conditioning
        self.res_conv = nn.Conv1d(channels, channels, 1)
        self.skip_conv = nn.Conv1d(channels, channels, 1)

    def forward(self, x, cond):
        h = self.dilated_conv(x)[..., :x.size(-1)]  # causal trim
        h = h + self.cond_proj(cond)
        gate, filt = h.chunk(2, dim=1)
        h = torch.tanh(filt) * torch.sigmoid(gate)
        skip = self.skip_conv(h)
        res = self.res_conv(h) + x
        return res, skip

# Stack of dilated convolutions
channels = 64
blocks = nn.ModuleList([
    WaveNetBlock(channels, dilation=2**i) for i in range(10)
])

x = torch.randn(1, channels, 1000)
cond = torch.randn(1, 80, 1000)  # mel spectrogram
skip_sum = 0
for block in blocks:
    x, skip = block(x, cond)
    skip_sum = skip_sum + skip
print(f"Output: {skip_sum.shape}")  # [1, 64, 1000]`}),e.jsx(f,{type:"note",title:"From WaveNet to Real-Time Vocoders",children:e.jsxs("p",{children:["WaveNet's autoregressive generation is extremely slow (minutes per second of audio). Parallel WaveNet uses inverse autoregressive flows for real-time synthesis. Modern vocoders like ",e.jsx("strong",{children:"HiFi-GAN"})," and ",e.jsx("strong",{children:"WaveGlow"})," achieve real-time speeds with comparable quality using GAN training or flow-based methods."]})})]})}const le=Object.freeze(Object.defineProperty({__proto__:null,default:q},Symbol.toStringTag,{value:"Module"}));function D(){const[n,c]=m.useState("vits"),a={vits:{name:"VITS",type:"VAE + Flow + GAN",endToEnd:!0,streaming:!1,zeroshort:!1,quality:"Excellent",speed:"Real-time"},valle:{name:"VALL-E",type:"Codec language model",endToEnd:!0,streaming:!0,zeroshort:!0,quality:"Near-human",speed:"Moderate"},voicebox:{name:"Voicebox",type:"Flow matching",endToEnd:!0,streaming:!1,zeroshort:!0,quality:"Near-human",speed:"Fast"},styletts2:{name:"StyleTTS 2",type:"Diffusion + style",endToEnd:!0,streaming:!1,zeroshort:!1,quality:"Human-level",speed:"Real-time"}},r=a[n];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Modern TTS Systems"}),e.jsx("div",{className:"flex flex-wrap gap-2 mb-4",children:Object.entries(a).map(([s,o])=>e.jsx("button",{onClick:()=>c(s),className:`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${n===s?"bg-violet-600 text-white":"bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400"}`,children:o.name},s))}),e.jsx("div",{className:"grid grid-cols-3 gap-3 text-sm",children:[["Architecture",r.type],["End-to-end",r.endToEnd?"Yes":"No"],["Zero-shot",r.zeroshort?"Yes":"No"],["Quality",r.quality],["Speed",r.speed],["Streaming",r.streaming?"Yes":"No"]].map(([s,o])=>e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3",children:[e.jsx("p",{className:"text-xs text-violet-600 dark:text-violet-400 font-semibold",children:s}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300 font-medium",children:o})]},s))})]})}function V(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Modern TTS systems have achieved human-level naturalness through end-to-end architectures that combine variational inference, normalizing flows, and codec language modeling, enabling zero-shot voice cloning from just seconds of reference audio."}),e.jsxs(x,{title:"VITS: Variational Inference with Adversarial Learning",children:[e.jsx("p",{children:"VITS combines a VAE, normalizing flow, and HiFi-GAN in a single end-to-end model:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_\\text{VITS} = \\mathcal{L}_\\text{recon} + D_\\text{KL}(q(z|x) \\| p(z|c)) + \\mathcal{L}_\\text{adv} + \\mathcal{L}_\\text{dur}"}),e.jsxs("p",{className:"mt-2",children:["The posterior encoder maps from linear spectrograms to latent ",e.jsx(t.InlineMath,{math:"z"}),", the prior encoder maps from text to the same latent space via normalizing flows, and the decoder generates waveforms directly from ",e.jsx(t.InlineMath,{math:"z"}),"."]})]}),e.jsx(D,{}),e.jsxs(b,{title:"VALL-E: Language Model Approach",id:"valle-approach",children:[e.jsx("p",{children:"VALL-E frames TTS as a conditional language model over neural audio codec tokens. Given a 3-second enrollment clip, it generates speech tokens autoregressively:"}),e.jsx(t.BlockMath,{math:"P(\\mathbf{c} | \\mathbf{t}, \\tilde{\\mathbf{c}}) = \\prod_{j=1}^{8} P(c^j | c^{<j}, \\mathbf{t}, \\tilde{\\mathbf{c}})"}),e.jsxs("p",{className:"mt-1",children:["where ",e.jsx(t.InlineMath,{math:"c^j"})," are the ",e.jsx(t.InlineMath,{math:"j"}),"-th codebook tokens,",e.jsx(t.InlineMath,{math:"\\mathbf{t}"})," is text, and ",e.jsx(t.InlineMath,{math:"\\tilde{\\mathbf{c}}"})," is the enrollment audio codec. This enables zero-shot voice cloning."]})]}),e.jsxs(g,{title:"Zero-Shot Voice Cloning",children:[e.jsx("p",{children:"Given a 3-second reference clip of an unseen speaker, VALL-E can:"}),e.jsxs("ul",{className:"list-disc pl-5 mt-2 space-y-1",children:[e.jsx("li",{children:"Preserve the speaker's voice characteristics (timbre, pitch range)"}),e.jsx("li",{children:"Maintain emotional tone and speaking style"}),e.jsx("li",{children:"Generate arbitrary text in that voice"}),e.jsx("li",{children:"Handle multiple languages with a multilingual variant"})]}),e.jsx("p",{className:"mt-2",children:"The key insight: discrete audio tokens allow treating speech as a language modeling problem, leveraging the power of large Transformer LMs."})]}),e.jsx(u,{title:"VITS-style End-to-End TTS (Simplified)",code:`import torch
import torch.nn as nn

class SimpleVITS(nn.Module):
    def __init__(self, vocab_size=100, hidden=192, latent=192):
        super().__init__()
        # Text encoder
        self.text_enc = nn.Sequential(
            nn.Embedding(vocab_size, hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Prior: text -> latent distribution (with flow)
        self.prior_mean = nn.Linear(hidden, latent)
        self.prior_logvar = nn.Linear(hidden, latent)
        # Decoder: latent -> waveform (simplified)
        self.decoder = nn.ConvTranspose1d(latent, 1, kernel_size=256, stride=256)

    def forward(self, text_ids):
        h = self.text_enc(text_ids)
        mu = self.prior_mean(h)
        logvar = self.prior_logvar(h)
        # Reparameterization trick
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        # Generate waveform from latent (real VITS uses HiFi-GAN decoder)
        waveform = self.decoder(z.transpose(1, 2))
        return waveform, mu, logvar

model = SimpleVITS()
text = torch.randint(0, 100, (1, 30))
waveform, mu, logvar = model(text)
print(f"Generated waveform: {waveform.shape}")  # [1, 1, 30*256]
print(f"Latent mean: {mu.shape}")  # [1, 30, 192]`}),e.jsx(f,{type:"note",title:"The Codec Language Model Paradigm",children:e.jsxs("p",{children:["The shift from mel spectrogram prediction to ",e.jsx("strong",{children:"neural audio codec token prediction"})," is transforming TTS. By using Encodec or SoundStream tokens, speech synthesis becomes a sequence-to-sequence language modeling task, enabling scaling laws similar to LLMs and naturally supporting zero-shot capabilities through in-context learning."]})})]})}const de=Object.freeze(Object.defineProperty({__proto__:null,default:V},Symbol.toStringTag,{value:"Module"}));function B(){const[n,c]=m.useState("jukebox"),a={musenet:{name:"MuseNet",arch:"Sparse Transformer",input:"MIDI tokens",output:"MIDI",training:"Autoregressive LM",duration:"~4 min",quality:"Good (symbolic)"},jukebox:{name:"Jukebox",arch:"VQ-VAE + Transformer",input:"Raw audio",output:"Raw audio",training:"Hierarchical VQ-VAE + autoregressive priors",duration:"~1 min",quality:"Good (audio artifacts)"},musiclm:{name:"MusicLM",arch:"AudioLM + MuLan",input:"Text description",output:"Audio tokens",training:"Hierarchical token prediction",duration:"~30s",quality:"High fidelity"},musicgen:{name:"MusicGen",arch:"Single Transformer",input:"Text / melody",output:"Codec tokens",training:"Codebook delay pattern",duration:"~30s",quality:"High fidelity"}},r=a[n];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Music Generation Models"}),e.jsx("div",{className:"flex flex-wrap gap-2 mb-4",children:Object.entries(a).map(([s,o])=>e.jsx("button",{onClick:()=>c(s),className:`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${n===s?"bg-violet-600 text-white":"bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400"}`,children:o.name},s))}),e.jsx("div",{className:"grid grid-cols-3 gap-3 text-sm",children:[["Architecture",r.arch],["Input",r.input],["Output",r.output],["Training",r.training],["Max duration",r.duration],["Quality",r.quality]].map(([s,o])=>e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3",children:[e.jsx("p",{className:"text-xs text-violet-600 dark:text-violet-400 font-semibold",children:s}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:o})]},s))})]})}function W(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Music generation has evolved from symbolic MIDI models to systems that produce high-fidelity audio directly from text descriptions, leveraging advances in audio tokenization and large-scale language modeling."}),e.jsxs(x,{title:"Jukebox: Hierarchical VQ-VAE",children:[e.jsx("p",{children:"Jukebox uses a three-level VQ-VAE to compress raw audio at different temporal resolutions:"}),e.jsx(t.BlockMath,{math:"x \\xrightarrow{\\text{Enc}_1} z_1 \\xrightarrow{\\text{Enc}_2} z_2 \\xrightarrow{\\text{Enc}_3} z_3"}),e.jsxs("p",{className:"mt-2",children:["Level 3 captures high-level musical structure at 8x compression, while level 1 captures fine acoustic details. Autoregressive Transformers generate tokens top-down:",e.jsx(t.InlineMath,{math:"z_3 \\to z_2 \\to z_1 \\to \\hat{x}"}),"."]})]}),e.jsx(B,{}),e.jsxs(b,{title:"MusicGen Codebook Interleaving",id:"musicgen-interleave",children:[e.jsxs("p",{children:["MusicGen avoids the need for multiple Transformer passes by interleaving codebook tokens with a delay pattern. For ",e.jsx(t.InlineMath,{math:"K"})," codebooks, each timestep",e.jsx(t.InlineMath,{math:"t"})," generates codebook ",e.jsx(t.InlineMath,{math:"k"})," at position ",e.jsx(t.InlineMath,{math:"t - k"}),":"]}),e.jsx(t.BlockMath,{math:"P(c_{t,k} | c_{<t}, \\text{text}) \\quad \\text{with delay } d_k = k"}),e.jsxs("p",{className:"mt-1",children:["This reduces ",e.jsx(t.InlineMath,{math:"K"})," sequential decoding passes to a single pass with only",e.jsx(t.InlineMath,{math:"K{-}1"})," steps of additional latency."]})]}),e.jsxs(g,{title:"Text-to-Music Pipeline",children:[e.jsx("p",{children:"A modern text-to-music system follows these steps:"}),e.jsxs("ol",{className:"list-decimal pl-5 mt-2 space-y-1",children:[e.jsx("li",{children:"Encode text description with a text encoder (T5 or CLAP)"}),e.jsx("li",{children:"Generate audio tokens conditioned on text embeddings"}),e.jsx("li",{children:"Decode tokens to waveform using neural audio codec decoder"}),e.jsx("li",{children:"Optional: apply post-processing (loudness normalization, effects)"})]}),e.jsx(t.BlockMath,{math:"\\text{``upbeat jazz piano''} \\xrightarrow{T5} e_\\text{text} \\xrightarrow{\\text{Transformer}} c_{1:T} \\xrightarrow{\\text{Encodec}} \\hat{x}"})]}),e.jsx(u,{title:"Music Generation with MusicGen",code:`import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# Load MusicGen model
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# Text-conditioned generation
inputs = processor(
    text=["upbeat jazz piano solo", "calm ambient electronic"],
    padding=True,
    return_tensors="pt",
)

# Generate 8 seconds of audio at 32kHz
audio_values = model.generate(**inputs, max_new_tokens=256)
print(f"Generated audio: {audio_values.shape}")
# Shape: [2, 1, 256000] (2 samples, mono, 8s * 32kHz)

# Sampling rate for MusicGen
sampling_rate = model.config.audio_encoder.sampling_rate
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Duration: {audio_values.shape[-1] / sampling_rate:.1f}s")`}),e.jsx(f,{type:"note",title:"Symbolic vs Audio Generation",children:e.jsxs("p",{children:[e.jsx("strong",{children:"Symbolic models"})," (MuseNet, Music Transformer) generate MIDI and offer precise control over notes, instruments, and structure, but require a separate synthesizer.",e.jsx("strong",{children:"Audio models"})," (Jukebox, MusicLM, MusicGen) generate waveforms directly with realistic timbres but less structural control. Hybrid approaches are an active research area."]})})]})}const ce=Object.freeze(Object.defineProperty({__proto__:null,default:W},Symbol.toStringTag,{value:"Module"}));function P(){const[n,c]=m.useState(50),[a,r]=m.useState(3),s=Array.from({length:5},(o,d)=>{const l=d/4;return{t:(l*n).toFixed(0),noise:(l*100).toFixed(0),signal:((1-l)*100).toFixed(0)}});return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Audio Diffusion Parameters"}),e.jsxs("div",{className:"flex flex-wrap gap-4 mb-4",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Denoising steps: ",n,e.jsx("input",{type:"range",min:10,max:200,step:10,value:n,onChange:o=>c(Number(o.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["CFG scale: ",a.toFixed(1),e.jsx("input",{type:"range",min:1,max:10,step:.5,value:a,onChange:o=>r(Number(o.target.value)),className:"w-28 accent-violet-500"})]})]}),e.jsx("div",{className:"flex gap-2",children:s.map((o,d)=>e.jsxs("div",{className:"flex-1 rounded-lg overflow-hidden",children:[e.jsxs("div",{className:"bg-violet-500 text-white text-center text-xs py-1",style:{height:`${100-Number(o.noise)}%`,minHeight:"20px"},children:[o.signal,"%"]}),e.jsxs("div",{className:"bg-gray-300 dark:bg-gray-600 text-center text-xs py-1",style:{height:`${Number(o.noise)}%`,minHeight:"20px"},children:[o.noise,"%"]}),e.jsxs("p",{className:"text-xs text-center text-gray-500 mt-1",children:["t=",o.t]})]},d))}),e.jsxs("p",{className:"text-xs text-gray-500 mt-2",children:["CFG strength ",a.toFixed(1),": ",a<3?"More diverse, less adherent to prompt":a<6?"Balanced quality and diversity":"High adherence, may reduce diversity"]})]})}function G(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Diffusion models have emerged as a powerful paradigm for audio generation, producing high-quality sound effects, music, and speech by iteratively denoising latent representations conditioned on text descriptions."}),e.jsxs(x,{title:"Latent Diffusion for Audio (AudioLDM)",children:[e.jsx("p",{children:"AudioLDM applies latent diffusion to mel spectrogram generation:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = \\mathbb{E}_{z_0, \\epsilon, t}\\left[\\|\\epsilon - \\epsilon_\\theta(z_t, t, c_\\text{text})\\|^2\\right]"}),e.jsxs("p",{className:"mt-2",children:["A VAE encodes mel spectrograms into latent space ",e.jsx(t.InlineMath,{math:"z_0"}),", the diffusion model operates in this compressed space, and a vocoder (HiFi-GAN) converts the decoded mel back to audio. Text conditioning uses CLAP embeddings."]})]}),e.jsx(P,{}),e.jsxs(g,{title:"Classifier-Free Guidance for Audio",children:[e.jsx("p",{children:"Audio diffusion models use CFG to balance quality and diversity:"}),e.jsx(t.BlockMath,{math:"\\hat{\\epsilon}_\\theta(z_t, c) = \\epsilon_\\theta(z_t, \\varnothing) + s \\cdot [\\epsilon_\\theta(z_t, c) - \\epsilon_\\theta(z_t, \\varnothing)]"}),e.jsxs("p",{className:"mt-1",children:["where ",e.jsx(t.InlineMath,{math:"s"})," is the guidance scale. During training, the text condition",e.jsx(t.InlineMath,{math:"c"})," is randomly dropped 10-20% of the time to enable unconditional generation."]})]}),e.jsx(u,{title:"Audio Generation with Diffusion (Simplified)",code:`import torch
import torch.nn as nn

class SimpleAudioUNet(nn.Module):
    """Simplified U-Net for audio latent diffusion."""
    def __init__(self, latent_dim=64, cond_dim=512, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.cond_proj = nn.Linear(cond_dim, time_dim)
        # Simplified encoder-decoder
        self.encoder = nn.Conv1d(latent_dim, 128, 3, padding=1)
        self.mid = nn.Conv1d(128, 128, 3, padding=1)
        self.decoder = nn.Conv1d(128, latent_dim, 3, padding=1)

    def forward(self, z_t, t, cond):
        t_emb = self.time_mlp(t.unsqueeze(-1)) + self.cond_proj(cond)
        h = self.encoder(z_t) + t_emb.unsqueeze(-1)
        h = torch.relu(self.mid(h))
        return self.decoder(h)

# Diffusion sampling loop
model = SimpleAudioUNet()
cond = torch.randn(1, 512)  # text embedding
z_t = torch.randn(1, 64, 100)  # start from noise
steps = 50

for i in range(steps, 0, -1):
    t = torch.tensor([i / steps])
    with torch.no_grad():
        noise_pred = model(z_t, t, cond)
    # Simplified DDPM step
    alpha = 1 - (i / steps) * 0.02
    z_t = (z_t - (1 - alpha) * noise_pred) / alpha**0.5

print(f"Denoised latent: {z_t.shape}")  # [1, 64, 100]`}),e.jsx(j,{title:"Audio Diffusion Challenges",children:e.jsx("p",{children:"Audio diffusion faces unique challenges: long sequences (30s audio = 1.5M samples), temporal coherence requirements, and the need for perceptually meaningful losses. Latent diffusion mitigates the computational cost, but generating long, coherent audio remains an open problem."})}),e.jsx(f,{type:"note",title:"Stable Audio and Beyond",children:e.jsx("p",{children:"Stable Audio from Stability AI applies latent diffusion with timing conditioning, enabling generation of variable-length audio up to 95 seconds. The model conditions on both text and timing embeddings (start time, total duration), providing fine-grained control over the generated audio's temporal structure."})})]})}const me=Object.freeze(Object.defineProperty({__proto__:null,default:G},Symbol.toStringTag,{value:"Module"}));function H(){const[n,c]=m.useState(8),[a,r]=m.useState(1024),[s,o]=m.useState(75),d=n*Math.log2(a)*s,l=16e3*16/d;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Neural Codec Calculator"}),e.jsxs("div",{className:"flex flex-wrap gap-4 mb-4",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Codebooks: ",n,e.jsx("input",{type:"range",min:1,max:16,step:1,value:n,onChange:i=>c(Number(i.target.value)),className:"w-24 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Size: ",a,e.jsx("input",{type:"range",min:256,max:4096,step:256,value:a,onChange:i=>r(Number(i.target.value)),className:"w-24 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Frame rate: ",s," Hz",e.jsx("input",{type:"range",min:25,max:150,step:25,value:s,onChange:i=>o(Number(i.target.value)),className:"w-24 accent-violet-500"})]})]}),e.jsxs("div",{className:"grid grid-cols-3 gap-3 text-sm",children:[e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3",children:[e.jsx("p",{className:"text-xs text-violet-600 dark:text-violet-400 font-semibold",children:"Bitrate"}),e.jsxs("p",{className:"text-xl font-bold text-violet-600",children:[(d/1e3).toFixed(1)," kbps"]})]}),e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3",children:[e.jsx("p",{className:"text-xs text-violet-600 dark:text-violet-400 font-semibold",children:"Compression"}),e.jsxs("p",{className:"text-xl font-bold text-violet-600",children:[l.toFixed(1),"x"]})]}),e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3",children:[e.jsx("p",{className:"text-xs text-violet-600 dark:text-violet-400 font-semibold",children:"Tokens/second"}),e.jsx("p",{className:"text-xl font-bold text-violet-600",children:n*s})]})]})]})}function O(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Neural audio codecs like Encodec and SoundStream compress audio into discrete tokens using residual vector quantization, enabling language model-based audio generation and extremely low-bitrate compression with high perceptual quality."}),e.jsxs(x,{title:"Residual Vector Quantization (RVQ)",children:[e.jsx("p",{children:"RVQ applies multiple rounds of vector quantization to the residual:"}),e.jsx(t.BlockMath,{math:"r_0 = z, \\quad q_k = \\text{VQ}_k(r_{k-1}), \\quad r_k = r_{k-1} - q_k"}),e.jsxs("p",{className:"mt-2",children:["After ",e.jsx(t.InlineMath,{math:"K"})," codebooks, the reconstructed vector is ",e.jsx(t.InlineMath,{math:"\\hat{z} = \\sum_{k=1}^{K} q_k"}),". Each codebook captures progressively finer details, with the first codebook storing the coarsest approximation."]})]}),e.jsx(H,{}),e.jsxs(b,{title:"Encodec Architecture",id:"encodec-architecture",children:[e.jsx("p",{children:"Encodec uses an encoder-decoder CNN with RVQ in the bottleneck:"}),e.jsx(t.BlockMath,{math:"x \\xrightarrow{\\text{Enc}} z \\xrightarrow{\\text{RVQ}} \\hat{z} \\xrightarrow{\\text{Dec}} \\hat{x}"}),e.jsxs("p",{className:"mt-1",children:["The encoder downsamples by ",e.jsx(t.InlineMath,{math:"320\\times"})," (at 24 kHz input, producing 75 frames/sec). Training combines reconstruction loss, adversarial loss (multi-scale discriminator), and commitment loss for stable quantization:"]}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = \\lambda_r \\mathcal{L}_\\text{recon} + \\lambda_a \\mathcal{L}_\\text{adv} + \\lambda_c \\sum_{k=1}^{K} \\|z - \\text{sg}[q_k]\\|^2"})]}),e.jsxs(g,{title:"Codec Tokens as Language",children:[e.jsxs("p",{children:["With 8 codebooks of size 1024 at 75 Hz, one second of audio becomes a matrix of ",e.jsx(t.InlineMath,{math:"8 \\times 75 = 600"})," discrete tokens. This enables treating audio as a sequence modeling problem:"]}),e.jsxs("ul",{className:"list-disc pl-5 mt-2 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"VALL-E:"})," Autoregressive generation of codec tokens for TTS"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"MusicGen:"})," Interleaved codebook prediction for music"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"AudioPaLM:"})," Unified text and audio tokens in one LM"]})]})]}),e.jsx(u,{title:"Using Encodec for Audio Tokenization",code:`import torch
from transformers import EncodecModel, AutoProcessor

# Load Encodec
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# Encode audio to discrete tokens
audio = torch.randn(1, 1, 24000)  # 1 second at 24kHz
inputs = processor(raw_audio=audio.squeeze().numpy(), sampling_rate=24000, return_tensors="pt")

with torch.no_grad():
    encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])

# Discrete codes: [batch, num_codebooks, time_frames]
codes = encoder_outputs.audio_codes
print(f"Audio codes: {codes.shape}")  # [1, 1, 8, 75]
print(f"Code range: [{codes.min()}, {codes.max()}]")

# Decode back to audio
with torch.no_grad():
    decoded = model.decode(codes, encoder_outputs.audio_scales)
print(f"Reconstructed audio: {decoded.audio_values.shape}")`}),e.jsx(f,{type:"note",title:"Beyond Compression: Audio as Tokens",children:e.jsxs("p",{children:["Neural audio codecs have become the ",e.jsx("strong",{children:"tokenizer for audio"}),", playing the same role as BPE for text. This unification enables multimodal models that seamlessly handle text, speech, music, and sound effects within a single Transformer architecture, opening the door to truly universal audio-language models."]})})]})}const he=Object.freeze(Object.defineProperty({__proto__:null,default:O},Symbol.toStringTag,{value:"Module"}));function $(){const[n,c]=m.useState("cosine"),a=[{name:"Speaker A",x:.8,y:.6,color:"#8b5cf6"},{name:"Speaker B",x:.3,y:.8,color:"#f97316"},{name:"Speaker C",x:.7,y:.2,color:"#06b6d4"}],r=300,s=250,l=n==="cosine"?(i,h)=>{const p=i.x*h.x+i.y*h.y,v=Math.sqrt(i.x*i.x+i.y*i.y),y=Math.sqrt(h.x*h.x+h.y*h.y);return p/(v*y)}:(i,h)=>Math.sqrt((i.x-h.x)**2+(i.y-h.y)**2);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Speaker Embedding Space"}),e.jsx("div",{className:"flex gap-4 mb-3",children:["cosine","euclidean"].map(i=>e.jsx("button",{onClick:()=>c(i),className:`rounded-lg px-3 py-1 text-sm font-medium capitalize ${n===i?"bg-violet-600 text-white":"bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"}`,children:i},i))}),e.jsxs("svg",{width:r,height:s,className:"mx-auto block",children:[e.jsx("rect",{width:r,height:s,fill:"none",stroke:"#e5e7eb",rx:8}),a.map((i,h)=>e.jsxs("g",{children:[e.jsx("circle",{cx:i.x*(r-40)+20,cy:(1-i.y)*(s-40)+20,r:8,fill:i.color,opacity:.8}),e.jsx("text",{x:i.x*(r-40)+20,y:(1-i.y)*(s-40)+25,fontSize:9,fill:i.color,textAnchor:"middle",children:i.name})]},h)),a.map((i,h)=>a.slice(h+1).map((p,v)=>e.jsx("line",{x1:i.x*(r-40)+20,y1:(1-i.y)*(s-40)+20,x2:p.x*(r-40)+20,y2:(1-p.y)*(s-40)+20,stroke:"#d1d5db",strokeWidth:.8,strokeDasharray:"3,3"},`${h}-${v}`)))]}),e.jsxs("div",{className:"mt-2 grid grid-cols-3 gap-2 text-xs text-center",children:[e.jsxs("p",{className:"text-gray-600 dark:text-gray-400",children:["A-B: ",l(a[0],a[1]).toFixed(3)]}),e.jsxs("p",{className:"text-gray-600 dark:text-gray-400",children:["A-C: ",l(a[0],a[2]).toFixed(3)]}),e.jsxs("p",{className:"text-gray-600 dark:text-gray-400",children:["B-C: ",l(a[1],a[2]).toFixed(3)]})]})]})}function U(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Speaker embeddings map variable-length utterances to fixed-dimensional vectors that capture speaker identity, enabling verification, identification, and diarization through simple distance comparisons in embedding space."}),e.jsxs(x,{title:"x-vector Architecture",children:[e.jsx("p",{children:"The x-vector system uses a TDNN (Time-Delay Neural Network) with statistical pooling:"}),e.jsx(t.BlockMath,{math:"e = W_2 \\cdot \\text{ReLU}(W_1 \\cdot [\\mu(h), \\sigma(h)])"}),e.jsxs("p",{className:"mt-2",children:["Frame-level features ",e.jsx(t.InlineMath,{math:"h_t"})," are extracted by TDNN layers, then aggregated via mean ",e.jsx(t.InlineMath,{math:"\\mu"})," and standard deviation ",e.jsx(t.InlineMath,{math:"\\sigma"})," pooling to produce a fixed-size utterance-level embedding."]})]}),e.jsx($,{}),e.jsxs(b,{title:"ECAPA-TDNN",id:"ecapa-tdnn",children:[e.jsx("p",{children:"ECAPA-TDNN improves x-vectors with three key innovations:"}),e.jsx(t.BlockMath,{math:"h_l = \\text{SE}(\\text{Res2Net}(\\text{TDNN}_l(h_{l-1})))"}),e.jsxs("p",{className:"mt-1",children:["(1) ",e.jsx("strong",{children:"Res2Net"})," blocks capture multi-scale features, (2) ",e.jsx("strong",{children:"Squeeze-Excitation"})," (SE) performs channel attention, and (3) ",e.jsx("strong",{children:"Attentive statistical pooling"})," learns frame importance weights:"]}),e.jsx(t.BlockMath,{math:"e = \\sum_t \\alpha_t h_t, \\quad \\alpha_t = \\text{softmax}(w^\\top \\tanh(Vh_t + b))"})]}),e.jsxs(g,{title:"Training with AAM-Softmax",children:[e.jsx("p",{children:"Speaker embeddings are trained with Additive Angular Margin Softmax:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = -\\log \\frac{e^{s \\cos(\\theta_{y} + m)}}{e^{s \\cos(\\theta_{y} + m)} + \\sum_{j \\neq y} e^{s \\cos \\theta_j}}"}),e.jsxs("p",{className:"mt-1",children:["where ",e.jsx(t.InlineMath,{math:"s"})," is a scale factor, ",e.jsx(t.InlineMath,{math:"m"})," is the angular margin, and ",e.jsx(t.InlineMath,{math:"\\theta_y"})," is the angle between the embedding and the class center. This forces inter-class separation in angular space."]})]}),e.jsx(u,{title:"Speaker Embedding Extraction",code:`import torch
import torch.nn as nn

class SimpleSpeakerEncoder(nn.Module):
    def __init__(self, input_dim=80, embed_dim=192):
        super().__init__()
        self.tdnn = nn.Sequential(
            nn.Conv1d(input_dim, 512, 5, padding=2), nn.ReLU(),
            nn.Conv1d(512, 512, 3, dilation=2, padding=2), nn.ReLU(),
            nn.Conv1d(512, 512, 3, dilation=3, padding=3), nn.ReLU(),
        )
        # Attentive statistical pooling
        self.attention = nn.Sequential(
            nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.embed = nn.Linear(1024, embed_dim)  # mean + std

    def forward(self, mel):  # mel: [B, 80, T]
        h = self.tdnn(mel)  # [B, 512, T]
        # Attention weights
        alpha = self.attention(h.transpose(1, 2)).squeeze(-1)  # [B, T]
        alpha = torch.softmax(alpha, dim=-1).unsqueeze(1)  # [B, 1, T]
        # Weighted statistics
        mean = (alpha * h).sum(dim=-1)
        var = (alpha * h**2).sum(dim=-1) - mean**2
        std = torch.sqrt(var.clamp(min=1e-9))
        stats = torch.cat([mean, std], dim=-1)
        return nn.functional.normalize(self.embed(stats), dim=-1)

model = SimpleSpeakerEncoder()
mel = torch.randn(4, 80, 200)
embeddings = model(mel)
print(f"Speaker embeddings: {embeddings.shape}")  # [4, 192]
# Cosine similarity between speakers
sim = torch.mm(embeddings, embeddings.T)
print(f"Similarity matrix:\\n{sim}")`}),e.jsx(f,{type:"note",title:"Self-Supervised Speaker Representations",children:e.jsx("p",{children:"Pre-trained models like wav2vec 2.0 and WavLM produce excellent speaker embeddings when fine-tuned, often outperforming purpose-built speaker encoders. The SUPERB benchmark shows that general audio representations transfer well to speaker tasks, suggesting shared underlying structure in speech representations."})})]})}const pe=Object.freeze(Object.defineProperty({__proto__:null,default:U},Symbol.toStringTag,{value:"Module"}));function Q(){const[n,c]=m.useState(.5),a=[{label:"Same speaker",score:.85,actual:!0},{label:"Same speaker",score:.72,actual:!0},{label:"Different",score:.31,actual:!1},{label:"Same speaker",score:.48,actual:!0},{label:"Different",score:.22,actual:!1},{label:"Different",score:.55,actual:!1}],r=a.filter(l=>l.actual&&l.score>=n).length,s=a.filter(l=>!l.actual&&l.score>=n).length,o=a.filter(l=>l.actual&&l.score<n).length,d=a.filter(l=>!l.actual&&l.score<n).length;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Verification Threshold Explorer"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Threshold: ",n.toFixed(2),e.jsx("input",{type:"range",min:.1,max:.9,step:.05,value:n,onChange:l=>c(Number(l.target.value)),className:"w-40 accent-violet-500"})]}),e.jsx("div",{className:"space-y-1 mb-3",children:a.map((l,i)=>{const p=l.score>=n===l.actual;return e.jsxs("div",{className:"flex items-center gap-2 text-sm",children:[e.jsx("div",{className:"w-32 text-gray-600 dark:text-gray-400",children:l.label}),e.jsxs("div",{className:"flex-1 h-4 bg-gray-100 dark:bg-gray-800 rounded relative",children:[e.jsx("div",{className:"h-full rounded bg-violet-400",style:{width:`${l.score*100}%`}}),e.jsx("div",{className:"absolute top-0 h-full border-l-2 border-red-500",style:{left:`${n*100}%`}})]}),e.jsx("span",{className:`text-xs font-bold ${p?"text-green-600":"text-red-500"}`,children:p?"Correct":"Error"})]},i)})}),e.jsxs("div",{className:"grid grid-cols-4 gap-2 text-sm text-center",children:[e.jsx("div",{className:"bg-green-100 dark:bg-green-900/30 rounded p-2",children:e.jsxs("span",{className:"font-bold text-green-700",children:["TP: ",r]})}),e.jsx("div",{className:"bg-red-100 dark:bg-red-900/30 rounded p-2",children:e.jsxs("span",{className:"font-bold text-red-600",children:["FP: ",s]})}),e.jsx("div",{className:"bg-red-100 dark:bg-red-900/30 rounded p-2",children:e.jsxs("span",{className:"font-bold text-red-600",children:["FN: ",o]})}),e.jsx("div",{className:"bg-green-100 dark:bg-green-900/30 rounded p-2",children:e.jsxs("span",{className:"font-bold text-green-700",children:["TN: ",d]})})]})]})}function Y(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Speaker verification determines whether two utterances belong to the same speaker, while speaker identification classifies an utterance among known speakers. Diarization extends this to segment multi-speaker audio into who spoke when."}),e.jsxs(x,{title:"Speaker Verification",children:[e.jsxs("p",{children:["Given enrollment embedding ",e.jsx(t.InlineMath,{math:"e_\\text{enr}"})," and test embedding ",e.jsx(t.InlineMath,{math:"e_\\text{test}"}),":"]}),e.jsx(t.BlockMath,{math:"\\text{score} = \\cos(e_\\text{enr}, e_\\text{test}) = \\frac{e_\\text{enr}^\\top e_\\text{test}}{\\|e_\\text{enr}\\| \\|e_\\text{test}\\|}"}),e.jsxs("p",{className:"mt-2",children:["Accept if ",e.jsx(t.InlineMath,{math:"\\text{score} \\geq \\tau"}),", reject otherwise. The threshold ",e.jsx(t.InlineMath,{math:"\\tau"})," controls the trade-off between false acceptance rate (FAR) and false rejection rate (FRR)."]})]}),e.jsx(Q,{}),e.jsxs(b,{title:"Equal Error Rate (EER)",id:"eer-metric",children:[e.jsx("p",{children:"The EER is the operating point where FAR equals FRR:"}),e.jsx(t.BlockMath,{math:"\\text{EER} = \\text{FAR}(\\tau^*) = \\text{FRR}(\\tau^*)"}),e.jsx("p",{className:"mt-1",children:"State-of-the-art systems achieve EER below 1% on VoxCeleb1. The minDCF metric weighs false acceptances and rejections differently for real-world applications."})]}),e.jsxs(g,{title:"Speaker Diarization Pipeline",children:[e.jsx("p",{children:"Modern neural diarization systems follow these steps:"}),e.jsxs("ol",{className:"list-decimal pl-5 mt-2 space-y-1",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"VAD:"})," Voice Activity Detection to find speech segments"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Segmentation:"})," Split into overlapping windows (1.5-3s)"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Embedding:"})," Extract speaker embeddings per segment"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Clustering:"})," Agglomerative or spectral clustering of embeddings"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Resegmentation:"})," Refine boundaries with a neural model"]})]})]}),e.jsx(u,{title:"Speaker Verification System",code:`import torch
import torch.nn.functional as F

def verify_speaker(model, audio_enr, audio_test, threshold=0.5):
    """Verify if two audio clips are from the same speaker."""
    with torch.no_grad():
        emb_enr = model(audio_enr)    # [1, 192]
        emb_test = model(audio_test)   # [1, 192]

    # Cosine similarity
    score = F.cosine_similarity(emb_enr, emb_test).item()
    return score >= threshold, score

# Simulate verification with random embeddings
emb_same = F.normalize(torch.randn(1, 192), dim=-1)
emb_similar = F.normalize(emb_same + 0.1 * torch.randn(1, 192), dim=-1)
emb_different = F.normalize(torch.randn(1, 192), dim=-1)

score_same = F.cosine_similarity(emb_same, emb_similar).item()
score_diff = F.cosine_similarity(emb_same, emb_different).item()

print(f"Same speaker similarity: {score_same:.4f}")
print(f"Different speaker similarity: {score_diff:.4f}")

# Compute EER (simplified)
def compute_eer(scores_pos, scores_neg, n_thresholds=100):
    thresholds = torch.linspace(0, 1, n_thresholds)
    for tau in thresholds:
        far = (scores_neg >= tau).float().mean()
        frr = (scores_pos < tau).float().mean()
        if far <= frr:
            return tau.item(), ((far + frr) / 2).item()
    return 0.5, 0.5

scores_pos = torch.tensor([0.85, 0.72, 0.91, 0.68, 0.79])
scores_neg = torch.tensor([0.31, 0.22, 0.15, 0.42, 0.28])
threshold, eer = compute_eer(scores_pos, scores_neg)
print(f"EER: {eer:.4f} at threshold {threshold:.4f}")`}),e.jsx(f,{type:"note",title:"End-to-End Neural Diarization",children:e.jsx("p",{children:"EEND (End-to-End Neural Diarization) replaces the traditional pipeline with a single neural network that directly predicts speaker activity for each frame. It handles overlapping speech naturally and can be combined with encoder-based approaches (EEND-VC) for handling variable numbers of speakers."})})]})}const xe=Object.freeze(Object.defineProperty({__proto__:null,default:Y},Symbol.toStringTag,{value:"Module"}));function K(){const[n,c]=m.useState("disentangle"),a={disentangle:{name:"Disentanglement",desc:"Separate content and speaker representations, swap speaker embedding",pros:"Clean separation, controllable",cons:"May lose prosody nuances",examples:"AutoVC, VQVC+"},cyclegan:{name:"CycleGAN-VC",desc:"Unpaired voice conversion using cycle-consistency loss",pros:"No parallel data needed",cons:"Limited quality, mode collapse risk",examples:"CycleGAN-VC2, CycleGAN-VC3"},diffusion:{name:"Diffusion-based",desc:"Convert via diffusion process conditioned on target speaker",pros:"High quality, flexible",cons:"Slow inference",examples:"DiffVC, CoMoSpeech"},codec:{name:"Codec-based",desc:"Manipulate neural audio codec tokens to change speaker identity",pros:"Real-time possible, high quality",cons:"Requires good codec",examples:"FreeVC, VALL-E X"}},r=a[n];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-3 text-base font-bold text-gray-800 dark:text-gray-200",children:"Voice Conversion Approaches"}),e.jsx("div",{className:"flex flex-wrap gap-2 mb-4",children:Object.entries(a).map(([s,o])=>e.jsx("button",{onClick:()=>c(s),className:`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${n===s?"bg-violet-600 text-white":"bg-gray-100 text-gray-600 hover:bg-violet-100 dark:bg-gray-800 dark:text-gray-400"}`,children:o.name},s))}),e.jsxs("div",{className:"rounded-lg bg-violet-50 dark:bg-violet-900/20 p-4 space-y-2",children:[e.jsx("p",{className:"text-sm text-gray-700 dark:text-gray-300",children:r.desc}),e.jsxs("div",{className:"grid grid-cols-3 gap-2 text-xs",children:[e.jsxs("div",{children:[e.jsx("span",{className:"font-semibold text-green-600",children:"Pros:"})," ",e.jsx("span",{className:"text-gray-600 dark:text-gray-400",children:r.pros})]}),e.jsxs("div",{children:[e.jsx("span",{className:"font-semibold text-red-500",children:"Cons:"})," ",e.jsx("span",{className:"text-gray-600 dark:text-gray-400",children:r.cons})]}),e.jsxs("div",{children:[e.jsx("span",{className:"font-semibold text-violet-600",children:"Examples:"})," ",e.jsx("span",{className:"text-gray-600 dark:text-gray-400",children:r.examples})]})]})]})]})}function X(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Voice conversion transforms the speaker identity of an utterance while preserving its linguistic content. This requires disentangling what is said from who says it, a fundamental challenge in speech representation learning."}),e.jsxs(x,{title:"Voice Conversion Objective",children:[e.jsxs("p",{children:["Given source speech ",e.jsx(t.InlineMath,{math:"x_s"})," from speaker ",e.jsx(t.InlineMath,{math:"s"})," and target speaker ",e.jsx(t.InlineMath,{math:"t"}),":"]}),e.jsx(t.BlockMath,{math:"\\hat{x}_t = G(c(x_s), e_t)"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"c(x_s)"})," extracts content (linguistic information) and ",e.jsx(t.InlineMath,{math:"e_t"})," is the target speaker embedding. The ideal conversion should satisfy: same content as source, same voice as target, natural prosody."]})]}),e.jsx(K,{}),e.jsxs(g,{title:"AutoVC: Information Bottleneck",children:[e.jsx("p",{children:"AutoVC uses a carefully tuned bottleneck to force content-speaker disentanglement:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = \\|x - \\text{Dec}(\\text{Enc}(x)_{:\\text{dim}_c}, e_\\text{spk})\\|^2"}),e.jsxs("p",{className:"mt-1",children:["The encoder output is downsampled to a bottleneck dimension that is large enough to preserve phonetic content but too small to encode speaker identity. The decoder must rely on the separately-provided speaker embedding ",e.jsx(t.InlineMath,{math:"e_\\text{spk}"}),"."]})]}),e.jsx(u,{title:"Simple Voice Conversion with Disentanglement",code:`import torch
import torch.nn as nn

class SimpleVoiceConverter(nn.Module):
    def __init__(self, mel_dim=80, content_dim=32, spk_dim=192):
        super().__init__()
        # Content encoder (bottleneck forces content-only)
        self.content_enc = nn.Sequential(
            nn.Conv1d(mel_dim, 256, 5, padding=2), nn.ReLU(),
            nn.Conv1d(256, 128, 5, padding=2), nn.ReLU(),
            nn.Conv1d(128, content_dim, 1),  # bottleneck
        )
        # Decoder: content + speaker -> mel
        self.decoder = nn.Sequential(
            nn.Conv1d(content_dim + spk_dim, 256, 5, padding=2), nn.ReLU(),
            nn.Conv1d(256, 256, 5, padding=2), nn.ReLU(),
            nn.Conv1d(256, mel_dim, 1),
        )

    def encode_content(self, mel):
        return self.content_enc(mel)

    def decode(self, content, spk_emb):
        # Expand speaker embedding to match time dimension
        spk = spk_emb.unsqueeze(-1).expand(-1, -1, content.size(-1))
        return self.decoder(torch.cat([content, spk], dim=1))

    def convert(self, source_mel, target_spk_emb):
        content = self.encode_content(source_mel)
        return self.decode(content, target_spk_emb)

model = SimpleVoiceConverter()
source_mel = torch.randn(1, 80, 200)  # source utterance
target_spk = torch.randn(1, 192)      # target speaker embedding

# Voice conversion
converted = model.convert(source_mel, target_spk)
print(f"Source mel: {source_mel.shape}")
print(f"Converted mel: {converted.shape}")  # [1, 80, 200]

# Self-reconstruction for training
source_spk = torch.randn(1, 192)
reconstructed = model.convert(source_mel, source_spk)
recon_loss = nn.functional.mse_loss(reconstructed, source_mel)
print(f"Reconstruction loss: {recon_loss.item():.4f}")`}),e.jsx(j,{title:"Ethical Considerations",children:e.jsxs("p",{children:["Voice conversion technology can be misused for voice spoofing, fraud, and deepfakes. Research in ",e.jsx("strong",{children:"anti-spoofing"})," and ",e.jsx("strong",{children:"deepfake detection"})," is critical. The ASVspoof challenge evaluates countermeasures against synthetic speech attacks. Responsible development requires watermarking and detection capabilities."]})}),e.jsx(f,{type:"note",title:"Zero-Shot Voice Conversion",children:e.jsx("p",{children:"Modern systems like FreeVC and kNN-VC enable conversion to any target speaker from just a few seconds of reference audio, without retraining. These leverage pre-trained self-supervised features (from WavLM or HuBERT) which naturally disentangle content from speaker characteristics at different network layers."})})]})}const ge=Object.freeze(Object.defineProperty({__proto__:null,default:X},Symbol.toStringTag,{value:"Module"}));export{se as a,ae as b,ne as c,re as d,ie as e,oe as f,le as g,de as h,ce as i,me as j,he as k,pe as l,xe as m,ge as n,te as s};

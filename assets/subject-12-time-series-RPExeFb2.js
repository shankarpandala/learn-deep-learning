import{j as e,r as v}from"./vendor-DpISuAX6.js";import{r as t}from"./vendor-katex-CbWCYdth.js";import{D as _,T as N,E as k,P as M,N as w,W as A,b as I}from"./subject-01-foundations-D0A1VJsr.js";function L(){const[r,j]=v.useState(.5),[s,n]=v.useState(1),i=420,m=180,o=120,d=Array.from({length:o},(h,u)=>{const f=u/o,p=r*f*3,y=s*Math.sin(2*Math.PI*f*4),b=Math.sin(u*7.3)*.3+Math.cos(u*13.1)*.2;return p+y+b}),x=Math.min(...d)-.3,a=Math.max(...d)+.3,l=(h,u)=>`${h/o*i},${m-(u-x)/(a-x)*m}`,c=d.map((h,u)=>`${u===0?"M":"L"}${l(u,h)}`).join(" ");return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Stationarity Explorer"}),e.jsxs("div",{className:"flex flex-wrap gap-4 mb-3",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Trend: ",r.toFixed(1),e.jsx("input",{type:"range",min:0,max:2,step:.1,value:r,onChange:h=>j(parseFloat(h.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Seasonality: ",s.toFixed(1),e.jsx("input",{type:"range",min:0,max:3,step:.1,value:s,onChange:h=>n(parseFloat(h.target.value)),className:"w-28 accent-violet-500"})]})]}),e.jsx("svg",{width:i,height:m,className:"mx-auto block",children:e.jsx("path",{d:c,fill:"none",stroke:"#8b5cf6",strokeWidth:2})}),e.jsx("p",{className:"mt-2 text-center text-xs text-gray-500 dark:text-gray-400",children:r===0&&s===0?"✓ Approximately stationary (noise only)":"Non-stationary — has trend and/or seasonality"})]})}function z(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Time series data consists of observations recorded sequentially over time. Before applying deep learning, understanding fundamental properties like stationarity, trend, and seasonality is critical for proper modeling and evaluation."}),e.jsxs(_,{title:"Stationarity",children:[e.jsxs("p",{children:["A time series ",e.jsx(t.InlineMath,{math:"X_t"})," is ",e.jsx("strong",{children:"strictly stationary"})," if its joint distribution is invariant to time shifts. In practice we use ",e.jsx("strong",{children:"weak stationarity"}),":"]}),e.jsx(t.BlockMath,{math:"\\mathbb{E}[X_t] = \\mu \\quad \\text{(constant)}, \\qquad \\text{Cov}(X_t, X_{t+h}) = \\gamma(h) \\quad \\text{(depends only on lag } h\\text{)}"})]}),e.jsxs(_,{title:"Autocorrelation Function (ACF)",children:[e.jsx(t.BlockMath,{math:"\\rho(h) = \\frac{\\gamma(h)}{\\gamma(0)} = \\frac{\\text{Cov}(X_t, X_{t+h})}{\\text{Var}(X_t)}"}),e.jsx("p",{className:"mt-2",children:"The ACF reveals repeating patterns, trend persistence, and seasonal cycles in the data."})]}),e.jsx(L,{}),e.jsxs(N,{title:"Classical Decomposition",id:"ts-decomposition",children:[e.jsx("p",{children:"Any time series can be decomposed into three components:"}),e.jsx(t.BlockMath,{math:"X_t = T_t + S_t + R_t"}),e.jsxs("p",{children:["where ",e.jsx(t.InlineMath,{math:"T_t"})," is the trend, ",e.jsx(t.InlineMath,{math:"S_t"})," is the seasonal component, and ",e.jsx(t.InlineMath,{math:"R_t"})," is the residual. A multiplicative variant uses ",e.jsx(t.InlineMath,{math:"X_t = T_t \\cdot S_t \\cdot R_t"}),"."]})]}),e.jsxs(k,{title:"Differencing for Stationarity",children:[e.jsx("p",{children:"First-order differencing removes a linear trend:"}),e.jsx(t.BlockMath,{math:"\\nabla X_t = X_t - X_{t-1}"}),e.jsxs("p",{children:["Seasonal differencing at period ",e.jsx(t.InlineMath,{math:"m"}),": ",e.jsx(t.InlineMath,{math:"\\nabla_m X_t = X_t - X_{t-m}"})]})]}),e.jsx(M,{title:"Stationarity Testing & Decomposition",code:`import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate sample series with trend + seasonality
t = np.arange(200)
series = 0.05 * t + 2 * np.sin(2 * np.pi * t / 12) + np.random.randn(200) * 0.5

# Augmented Dickey-Fuller test for stationarity
result = adfuller(series)
print(f"ADF statistic: {result[0]:.4f}, p-value: {result[1]:.4f}")
print("Stationary" if result[1] < 0.05 else "Non-stationary")

# Decompose (period=12 for monthly seasonality)
decomp = seasonal_decompose(series, model='additive', period=12)
print(f"Trend range: [{decomp.trend[~np.isnan(decomp.trend)].min():.2f}, "
      f"{decomp.trend[~np.isnan(decomp.trend)].max():.2f}]")`}),e.jsx(w,{type:"note",title:"Why Stationarity Matters for DL",children:e.jsxs("p",{children:["While deep learning models can implicitly learn trends and seasonality, making a series stationary before training often improves convergence and generalization. Many state-of-the-art models like N-BEATS apply ",e.jsx("strong",{children:"reversible instance normalization"})," — a learned form of stationarization."]})})]})}const he=Object.freeze(Object.defineProperty({__proto__:null,default:z},Symbol.toStringTag,{value:"Module"}));function D(){const[r,j]=v.useState(4),[s,n]=v.useState(2),i=[2.1,3.5,1.8,4.2,3.9,5.1,2.7,4.8,3.3,5.5,4.1,6],m=34,o=28,d=2,x=[];for(let a=0;a<=i.length-r-s;a++)x.push({start:a,inputEnd:a+r,targetEnd:a+r+s});return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Sliding Window Visualization"}),e.jsxs("div",{className:"flex flex-wrap gap-4 mb-3",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Lookback: ",r,e.jsx("input",{type:"range",min:2,max:6,step:1,value:r,onChange:a=>j(parseInt(a.target.value)),className:"w-24 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Horizon: ",s,e.jsx("input",{type:"range",min:1,max:4,step:1,value:s,onChange:a=>n(parseInt(a.target.value)),className:"w-24 accent-violet-500"})]})]}),e.jsx("div",{className:"overflow-x-auto",children:e.jsxs("svg",{width:i.length*(m+d)+10,height:x.length*(o+d)+o+10,children:[i.map((a,l)=>e.jsx("text",{x:l*(m+d)+m/2,y:14,textAnchor:"middle",className:"text-[10px] fill-gray-500",children:a},`h-${l}`)),x.map((a,l)=>e.jsx("g",{transform:`translate(0, ${l*(o+d)+22})`,children:i.map((c,h)=>{const u=h>=a.start&&h<a.inputEnd,f=h>=a.inputEnd&&h<a.targetEnd;return e.jsx("rect",{x:h*(m+d),y:0,width:m,height:o,rx:3,fill:u?"#8b5cf6":f?"#f97316":"#f3f4f6",opacity:u||f?.85:.25},h)})},l))]})}),e.jsxs("div",{className:"mt-2 flex gap-4 text-xs text-gray-500 dark:text-gray-400",children:[e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-3 rounded bg-violet-500"})," Input window"]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-3 rounded bg-orange-500"})," Forecast horizon"]}),e.jsxs("span",{className:"ml-auto",children:[x.length," samples generated"]})]})]})}function P(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Transforming a raw time series into supervised learning samples requires careful windowing. The choice of lookback length and forecast horizon directly affects what the model can learn and predict."}),e.jsxs(_,{title:"Sliding Window Transformation",children:[e.jsxs("p",{children:["Given a time series ",e.jsx(t.InlineMath,{math:"\\{x_1, \\ldots, x_T\\}"}),", lookback ",e.jsx(t.InlineMath,{math:"L"}),", and horizon ",e.jsx(t.InlineMath,{math:"H"}),":"]}),e.jsx(t.BlockMath,{math:"\\text{Input: } \\mathbf{x}_{t-L:t} = [x_{t-L}, \\ldots, x_{t-1}] \\;\\;\\to\\;\\; \\text{Target: } \\mathbf{y}_{t:t+H} = [x_t, \\ldots, x_{t+H-1}]"}),e.jsxs("p",{className:"mt-2",children:["This produces ",e.jsx(t.InlineMath,{math:"T - L - H + 1"})," training samples from a series of length ",e.jsx(t.InlineMath,{math:"T"}),"."]})]}),e.jsx(D,{}),e.jsxs(k,{title:"Lag Features",children:[e.jsx("p",{children:"Lag features augment each timestep with previous values as additional input dimensions:"}),e.jsx(t.BlockMath,{math:"\\mathbf{f}_t = [x_t, x_{t-1}, x_{t-7}, x_{t-14}, \\ldots]"}),e.jsx("p",{children:"Calendar features (day-of-week, month, holiday flags) provide exogenous context for seasonal patterns."})]}),e.jsx(A,{title:"Data Leakage in Time Series",children:e.jsxs("p",{children:["Never shuffle time series data randomly for train/test splits. Always use a temporal cutoff: train on ",e.jsx(t.InlineMath,{math:"[1, T_{\\text{train}}]"}),", validate on",e.jsx(t.InlineMath,{math:"(T_{\\text{train}}, T_{\\text{val}}]"}),", test on ",e.jsx(t.InlineMath,{math:"(T_{\\text{val}}, T]"}),". Normalization statistics must be computed ",e.jsx("strong",{children:"only from the training set"}),"."]})}),e.jsx(M,{title:"Creating Sliding Windows in PyTorch",code:`import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, series, lookback=24, horizon=6):
        self.X, self.Y = [], []
        for i in range(len(series) - lookback - horizon + 1):
            self.X.append(series[i:i+lookback])
            self.Y.append(series[i+lookback:i+lookback+horizon])
        self.X = torch.stack(self.X)
        self.Y = torch.stack(self.Y)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

# Example: 1000-step series
series = torch.sin(torch.linspace(0, 20*3.14159, 1000)) + torch.randn(1000)*0.1
ds = TimeSeriesDataset(series, lookback=24, horizon=6)
loader = DataLoader(ds, batch_size=32, shuffle=False)  # no shuffle!
print(f"Samples: {len(ds)}, batch X: {next(iter(loader))[0].shape}")`}),e.jsx(w,{type:"note",title:"Stride and Multi-Scale Windows",children:e.jsx("p",{children:"Using a stride greater than 1 reduces overlapping samples, which can speed up training. Multi-scale windowing — combining short and long lookback periods — helps models capture both short-term dynamics and long-range dependencies simultaneously."})})]})}const me=Object.freeze(Object.defineProperty({__proto__:null,default:P},Symbol.toStringTag,{value:"Module"}));function B(){const[r,j]=v.useState(0),s=[3,4.5,2.8,5.1,3.7,4.2,3.9,5],i=[3.2,4.1,3,4.8,3.5,4.5,3.7,5.2].map((c,h)=>h===3?c+r:c),m=s.reduce((c,h,u)=>c+Math.abs(h-i[u]),0)/s.length,o=Math.sqrt(s.reduce((c,h,u)=>c+(h-i[u])**2,0)/s.length),d=s.reduce((c,h,u)=>c+Math.abs((h-i[u])/h),0)/s.length*100,x=360,a=140,l=x/(s.length+1);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Metric Sensitivity to Outliers"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Outlier magnitude: ",r.toFixed(1),e.jsx("input",{type:"range",min:0,max:5,step:.2,value:r,onChange:c=>j(parseFloat(c.target.value)),className:"w-32 accent-violet-500"})]}),e.jsx("svg",{width:x,height:a,className:"mx-auto block",children:s.map((c,h)=>{const u=(h+1)*l,f=a-(c-1)/6*a,p=a-(i[h]-1)/6*a;return e.jsxs("g",{children:[e.jsx("line",{x1:u,y1:f,x2:u,y2:p,stroke:"#e5e7eb",strokeWidth:1,strokeDasharray:"2,2"}),e.jsx("circle",{cx:u,cy:f,r:4,fill:"#8b5cf6"}),e.jsx("circle",{cx:u,cy:p,r:4,fill:"#f97316"})]},h)})}),e.jsxs("div",{className:"mt-2 flex justify-center gap-4 text-xs",children:[e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-3 rounded-full bg-violet-500"})," Actual"]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-3 rounded-full bg-orange-500"})," Predicted"]})]}),e.jsxs("div",{className:"mt-3 grid grid-cols-3 gap-3 text-center text-sm",children:[e.jsxs("div",{className:"rounded-lg bg-violet-50 p-2 dark:bg-violet-900/20",children:[e.jsx("div",{className:"font-bold text-violet-700 dark:text-violet-300",children:"MAE"}),e.jsx("div",{className:"text-gray-700 dark:text-gray-300",children:m.toFixed(3)})]}),e.jsxs("div",{className:"rounded-lg bg-violet-50 p-2 dark:bg-violet-900/20",children:[e.jsx("div",{className:"font-bold text-violet-700 dark:text-violet-300",children:"RMSE"}),e.jsx("div",{className:"text-gray-700 dark:text-gray-300",children:o.toFixed(3)})]}),e.jsxs("div",{className:"rounded-lg bg-violet-50 p-2 dark:bg-violet-900/20",children:[e.jsx("div",{className:"font-bold text-violet-700 dark:text-violet-300",children:"MAPE"}),e.jsxs("div",{className:"text-gray-700 dark:text-gray-300",children:[d.toFixed(1),"%"]})]})]})]})}function q(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Evaluating forecasting models requires metrics that capture different error properties and evaluation protocols that prevent data leakage across time."}),e.jsxs(_,{title:"Core Forecasting Metrics",children:[e.jsx(t.BlockMath,{math:"\\text{MAE} = \\frac{1}{H}\\sum_{h=1}^{H}|y_{t+h} - \\hat{y}_{t+h}|"}),e.jsx(t.BlockMath,{math:"\\text{RMSE} = \\sqrt{\\frac{1}{H}\\sum_{h=1}^{H}(y_{t+h} - \\hat{y}_{t+h})^2}"}),e.jsx(t.BlockMath,{math:"\\text{MAPE} = \\frac{100\\%}{H}\\sum_{h=1}^{H}\\left|\\frac{y_{t+h} - \\hat{y}_{t+h}}{y_{t+h}}\\right|"})]}),e.jsx(B,{}),e.jsxs(N,{title:"MAE vs RMSE Sensitivity",id:"mae-rmse-comparison",children:[e.jsx("p",{children:"RMSE penalizes large errors disproportionately due to squaring:"}),e.jsx(t.BlockMath,{math:"\\text{MAE} \\leq \\text{RMSE} \\leq \\sqrt{H} \\cdot \\text{MAE}"}),e.jsxs("p",{children:["When ",e.jsx(t.InlineMath,{math:"\\text{RMSE} \\gg \\text{MAE}"}),", it signals the presence of occasional large errors (outliers)."]})]}),e.jsx(k,{title:"Rolling Window Evaluation",children:e.jsxs("p",{children:["Instead of a single train-test split, use expanding or sliding window cross-validation: train on ",e.jsx(t.InlineMath,{math:"[1, t]"}),", predict ",e.jsx(t.InlineMath,{math:"[t+1, t+H]"}),", then slide forward by a step size ",e.jsx(t.InlineMath,{math:"s"}),". Average metrics across all folds for a robust estimate."]})}),e.jsx(M,{title:"Forecasting Metrics & Walk-Forward Evaluation",code:`import numpy as np

def mae(y, yhat): return np.mean(np.abs(y - yhat))
def rmse(y, yhat): return np.sqrt(np.mean((y - yhat)**2))
def mape(y, yhat): return 100 * np.mean(np.abs((y - yhat) / y))

# Walk-forward evaluation
def walk_forward_eval(series, model_fn, lookback=24, horizon=6, step=6):
    scores = []
    for t in range(lookback, len(series) - horizon, step):
        X = series[t-lookback:t]
        y_true = series[t:t+horizon]
        y_pred = model_fn(X)  # your model's forecast
        scores.append(mae(y_true, y_pred))
    return np.mean(scores), np.std(scores)

# Example with naive persistence baseline
series = np.sin(np.linspace(0, 20*np.pi, 500)) + np.random.randn(500)*0.1
naive_fn = lambda x: np.full(6, x[-1])  # repeat last value
mean_mae, std_mae = walk_forward_eval(series, naive_fn)
print(f"Naive baseline MAE: {mean_mae:.4f} +/- {std_mae:.4f}")`}),e.jsx(w,{type:"note",title:"Scaled Metrics for Cross-Series Comparison",children:e.jsxs("p",{children:["When comparing across series with different scales, use ",e.jsx("strong",{children:"MASE"})," (Mean Absolute Scaled Error), which normalizes by the in-sample naive forecast error. This makes MASE scale-independent: values below 1.0 indicate the model beats the naive baseline."]})})]})}const xe=Object.freeze(Object.defineProperty({__proto__:null,default:q},Symbol.toStringTag,{value:"Module"}));function F(){const[r,j]=v.useState(1),s=400,n=160,i=40,m=25,o=Array.from({length:i},(f,p)=>2*Math.sin(p*.3)+.5*Math.cos(p*.7)),d=-4,x=4,a=(f,p)=>`${f/i*s},${n-(p-d)/(x-d)*n}`,l=o.slice(0,m).map((f,p)=>`${p===0?"M":"L"}${a(p,f)}`).join(" "),c=o.slice(m-1).map((f,p)=>`${p===0?"M":"L"}${a(m-1+p,f+Math.sin(p*.2)*.2)}`).join(" "),h=o.slice(m-1).map((f,p)=>{const y=(m-1+p)/i*s,b=f+Math.sin(p*.2)*.2,g=r*(.3+p*.08);return{x:y,upper:n-(b+g-d)/(x-d)*n,lower:n-(b-g-d)/(x-d)*n}}),u=h.map((f,p)=>`${p===0?"M":"L"}${f.x},${f.upper}`).join(" ")+h.reverse().map(f=>`L${f.x},${f.lower}`).join(" ")+"Z";return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Probabilistic Forecast"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Uncertainty scale: ",r.toFixed(1),e.jsx("input",{type:"range",min:.2,max:3,step:.1,value:r,onChange:f=>j(parseFloat(f.target.value)),className:"w-32 accent-violet-500"})]}),e.jsxs("svg",{width:s,height:n,className:"mx-auto block",children:[e.jsx("rect",{x:m/i*s,y:0,width:s-m/i*s,height:n,fill:"#8b5cf6",opacity:.05}),e.jsx("path",{d:u,fill:"#8b5cf6",opacity:.2}),e.jsx("path",{d:l,fill:"none",stroke:"#6b7280",strokeWidth:2}),e.jsx("path",{d:c,fill:"none",stroke:"#8b5cf6",strokeWidth:2}),e.jsx("line",{x1:m/i*s,y1:0,x2:m/i*s,y2:n,stroke:"#9ca3af",strokeDasharray:"4,3",strokeWidth:1}),e.jsx("text",{x:m/i*s+4,y:12,className:"text-[10px] fill-gray-400",children:"forecast"})]}),e.jsx("div",{className:"mt-2 flex justify-center gap-4 text-xs text-gray-500",children:e.jsx("span",{children:"Prediction intervals widen with horizon — reflecting growing uncertainty"})})]})}function W(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"DeepAR is an autoregressive RNN model that produces probabilistic forecasts by parameterizing a likelihood function at each time step. It learns across many related time series, sharing patterns via global parameters."}),e.jsxs(_,{title:"DeepAR Model",children:[e.jsxs("p",{children:["At each step ",e.jsx(t.InlineMath,{math:"t"}),", an LSTM computes hidden state ",e.jsx(t.InlineMath,{math:"h_t"})," and outputs distribution parameters:"]}),e.jsx(t.BlockMath,{math:"h_t = \\text{LSTM}(h_{t-1},\\; [x_{t-1},\\; \\mathbf{c}_t])"}),e.jsx(t.BlockMath,{math:"\\mu_t, \\sigma_t = \\text{MLP}(h_t), \\quad z_t \\sim \\mathcal{N}(\\mu_t, \\sigma_t^2)"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"\\mathbf{c}_t"})," contains covariates (time features, static embeddings)."]})]}),e.jsx(F,{}),e.jsxs(N,{title:"Negative Log-Likelihood Loss",id:"deepar-loss",children:[e.jsx("p",{children:"DeepAR is trained by maximizing the log-likelihood of the observed data:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = -\\sum_{t=1}^{T} \\log p(x_t \\mid \\mu_t, \\sigma_t) = \\sum_{t=1}^{T}\\left[\\log \\sigma_t + \\frac{(x_t - \\mu_t)^2}{2\\sigma_t^2}\\right] + C"}),e.jsx("p",{children:"For count data, a negative binomial likelihood replaces the Gaussian."})]}),e.jsxs(k,{title:"Quantile Regression Alternative",children:[e.jsxs("p",{children:["Instead of parametric distributions, predict quantiles directly. The pinball loss for quantile ",e.jsx(t.InlineMath,{math:"q"}),":"]}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_q(y, \\hat{y}) = \\begin{cases} q(y - \\hat{y}) & \\text{if } y \\geq \\hat{y} \\\\ (1-q)(\\hat{y} - y) & \\text{if } y < \\hat{y} \\end{cases}"})]}),e.jsx(M,{title:"DeepAR with GluonTS",code:`from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.dataset.pandas import PandasDataset
import pandas as pd
import numpy as np

# Create sample time series dataset (e.g., 100 daily series)
freq = "D"
prediction_length = 14
dates = pd.date_range("2020-01-01", periods=365, freq=freq)
# Simulate multiple related series (DeepAR learns across all)
dataframes = []
for i in range(100):
    trend = np.linspace(0, 2, 365) + np.random.randn() * 0.5
    seasonal = 2 * np.sin(2 * np.pi * np.arange(365) / 7)
    noise = np.random.randn(365) * 0.3
    df = pd.DataFrame({"target": trend + seasonal + noise}, index=dates)
    df["item_id"] = f"series_{i}"
    dataframes.append(df)
dataset = PandasDataset(pd.concat(dataframes), target="target", item_id="item_id")

# DeepAR: autoregressive RNN with learned likelihood
estimator = DeepAREstimator(
    freq=freq,
    prediction_length=prediction_length,
    num_layers=2,              # LSTM layers
    hidden_size=40,            # Hidden units per layer
    dropout_rate=0.1,
    trainer_kwargs={"max_epochs": 5, "accelerator": "auto"},
)

# Train global model across all 100 series
predictor = estimator.train(dataset)

# Generate probabilistic forecasts (returns sample paths)
forecasts = list(predictor.predict(dataset))
fc = forecasts[0]
print(f"Forecast shape: {fc.samples.shape}")  # (num_samples, 14)
print(f"Mean forecast: {fc.mean[:3]}")
print(f"Quantile 0.9:  {fc.quantile(0.9)[:3]}")`}),e.jsx(w,{type:"note",title:"Multi-Series Training",children:e.jsx("p",{children:"DeepAR's key advantage is training a single global model across thousands of related time series. Each series gets a learned embedding vector, allowing the model to share seasonal and trend patterns while adapting to individual series characteristics."})})]})}const pe=Object.freeze(Object.defineProperty({__proto__:null,default:W},Symbol.toStringTag,{value:"Module"}));function E(){const[r,j]=v.useState(3),s=80,n=50,i=16,m=30,o=r*(s+i+m)+80,d=140;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"N-BEATS Doubly Residual Stack"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Blocks: ",r,e.jsx("input",{type:"range",min:2,max:5,step:1,value:r,onChange:x=>j(parseInt(x.target.value)),className:"w-24 accent-violet-500"})]}),e.jsx("div",{className:"overflow-x-auto",children:e.jsxs("svg",{width:o,height:d,className:"mx-auto block",children:[e.jsx("text",{x:4,y:d/2-15,className:"text-[10px] fill-gray-500",children:"Input"}),e.jsx("text",{x:4,y:d/2+5,className:"text-[10px] fill-violet-500",children:"x"}),Array.from({length:r},(x,a)=>{const l=40+a*(s+i+m),c=(d-n)/2;return e.jsxs("g",{children:[e.jsx("line",{x1:l-m,y1:d/2,x2:l,y2:d/2,stroke:"#8b5cf6",strokeWidth:1.5,markerEnd:"url(#arrow)"}),e.jsx("rect",{x:l,y:c,width:s,height:n,rx:6,fill:"#8b5cf6",opacity:.15,stroke:"#8b5cf6",strokeWidth:1.5}),e.jsxs("text",{x:l+s/2,y:c+20,textAnchor:"middle",className:"text-[10px] fill-violet-700 dark:fill-violet-300 font-semibold",children:["Block ",a+1]}),e.jsx("text",{x:l+s/2,y:c+35,textAnchor:"middle",className:"text-[9px] fill-gray-500",children:a===0?"Trend":a===1?"Season":"Generic"}),e.jsx("line",{x1:l+s/2,y1:c+n,x2:l+s/2,y2:c+n+20,stroke:"#f97316",strokeWidth:1}),e.jsxs("text",{x:l+s/2,y:c+n+30,textAnchor:"middle",className:"text-[9px] fill-orange-500",children:["f",a+1]})]},a)}),e.jsx("defs",{children:e.jsx("marker",{id:"arrow",viewBox:"0 0 10 10",refX:9,refY:5,markerWidth:5,markerHeight:5,orient:"auto-start-auto",children:e.jsx("path",{d:"M 0 0 L 10 5 L 0 10 z",fill:"#8b5cf6"})})}),e.jsx("text",{x:o-30,y:d-10,className:"text-[10px] fill-orange-500",children:"y = sum(fi)"})]})}),e.jsx("p",{className:"mt-2 text-center text-xs text-gray-500 dark:text-gray-400",children:"Each block produces a backcast (residual for next block) and a partial forecast. Final forecast is the sum of all partial forecasts."})]})}function R(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"N-BEATS (Neural Basis Expansion Analysis for Time Series) is a pure deep learning architecture that achieves state-of-the-art results without requiring time series-specific feature engineering. N-HiTS extends it with hierarchical interpolation for long horizons."}),e.jsxs(_,{title:"N-BEATS Block",children:[e.jsxs("p",{children:["Each block receives input ",e.jsx(t.InlineMath,{math:"\\mathbf{x}_\\ell"})," and produces:"]}),e.jsx(t.BlockMath,{math:"\\mathbf{h} = \\text{FC}_4 \\circ \\text{FC}_3 \\circ \\text{FC}_2 \\circ \\text{FC}_1(\\mathbf{x}_\\ell)"}),e.jsx(t.BlockMath,{math:"\\hat{\\mathbf{x}}_\\ell = \\mathbf{V}_b^\\top \\boldsymbol{\\theta}_b(\\mathbf{h}) \\quad \\text{(backcast)}, \\qquad \\hat{\\mathbf{y}}_\\ell = \\mathbf{V}_f^\\top \\boldsymbol{\\theta}_f(\\mathbf{h}) \\quad \\text{(forecast)}"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"\\mathbf{V}_b, \\mathbf{V}_f"})," are basis matrices — either learned or constrained (trend/seasonal)."]})]}),e.jsx(E,{}),e.jsxs(N,{title:"Interpretable Basis Functions",id:"nbeats-basis",children:[e.jsxs("p",{children:["Trend basis uses polynomial coefficients up to degree ",e.jsx(t.InlineMath,{math:"p"}),":"]}),e.jsx(t.BlockMath,{math:"\\mathbf{V}_{\\text{trend}} = \\begin{bmatrix} 1 & t & t^2 & \\cdots & t^p \\end{bmatrix}"}),e.jsxs("p",{children:["Seasonality basis uses Fourier terms with period ",e.jsx(t.InlineMath,{math:"S"}),":"]}),e.jsx(t.BlockMath,{math:"\\mathbf{V}_{\\text{season}} = \\begin{bmatrix} \\cos(2\\pi t/S) & \\sin(2\\pi t/S) & \\cos(4\\pi t/S) & \\cdots \\end{bmatrix}"})]}),e.jsx(k,{title:"N-HiTS: Hierarchical Interpolation",children:e.jsx("p",{children:"N-HiTS adds multi-rate signal sampling — each block operates at a different temporal resolution. Lower blocks capture fine-grained patterns, higher blocks capture long-range structure. The forecast is assembled by interpolating outputs from all levels."})}),e.jsx(M,{title:"N-BEATS / N-HiTS with NeuralForecast",code:`from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
from neuralforecast.losses.pytorch import MAE
import pandas as pd
import numpy as np

# Prepare data in NeuralForecast format: unique_id, ds, y
dates = pd.date_range("2020-01-01", periods=365, freq="D")
series = []
for i in range(50):
    trend = np.linspace(0, 3, 365)
    season = 2 * np.sin(2 * np.pi * np.arange(365) / 7)
    y = trend + season + np.random.randn(365) * 0.5
    df = pd.DataFrame({"unique_id": f"s{i}", "ds": dates, "y": y})
    series.append(df)
data = pd.concat(series).reset_index(drop=True)

horizon = 14

# N-BEATS: interpretable stacks (trend + seasonality)
nbeats = NBEATS(
    h=horizon,
    input_size=2 * horizon,       # lookback = 2x horizon
    stack_types=["trend", "seasonality"],  # interpretable config
    n_blocks=[3, 3],
    mlp_units=[[256, 256]] * 2,
    loss=MAE(),
    max_steps=100,
)

# N-HiTS: hierarchical interpolation for long horizons
nhits = NHITS(
    h=horizon,
    input_size=2 * horizon,
    n_pool_kernel_size=[4, 2, 1],  # multi-rate downsampling
    loss=MAE(),
    max_steps=100,
)

# Train and forecast
nf = NeuralForecast(models=[nbeats, nhits], freq="D")
nf.fit(df=data)
forecasts = nf.predict()
print(forecasts.head())
# Columns: unique_id, ds, NBEATS, NHITS`}),e.jsx(w,{type:"note",title:"N-BEATS vs N-HiTS Trade-offs",children:e.jsx("p",{children:"N-BEATS works best for short-to-medium horizons with its uniform architecture. N-HiTS excels at long horizons by allowing different blocks to focus on different temporal scales, reducing computation by up to 50x while matching or exceeding N-BEATS accuracy."})})]})}const fe=Object.freeze(Object.defineProperty({__proto__:null,default:R},Symbol.toStringTag,{value:"Module"}));function H(){const[r,j]=v.useState(0),n=[1,2,4,8][r],i=400,m=160,o=16,d=i/o,x=30;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Causal Dilated Convolution"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Layer: ",r," (dilation = ",n,")",e.jsx("input",{type:"range",min:0,max:3,step:1,value:r,onChange:a=>j(parseInt(a.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("span",{className:"text-xs text-gray-500",children:["Receptive field: ",1+2*(Math.pow(2,r+1)-1)," steps"]})]}),e.jsxs("svg",{width:i,height:m,className:"mx-auto block",children:[Array.from({length:o},(a,l)=>e.jsxs("g",{children:[e.jsx("rect",{x:l*d+1,y:m-x,width:d-2,height:x-2,rx:3,fill:"#e5e7eb"}),e.jsxs("text",{x:l*d+d/2,y:m-8,textAnchor:"middle",className:"text-[9px] fill-gray-500",children:["t-",o-1-l]}),e.jsx("rect",{x:l*d+1,y:m-2*x-4,width:d-2,height:x-2,rx:3,fill:l>=o-1-2*n&&l<=o-1&&(o-1-l)%n===0?"#8b5cf6":"#f9fafb",stroke:l>=o-1-2*n&&l<=o-1&&(o-1-l)%n===0?"#8b5cf6":"#e5e7eb",strokeWidth:1})]},l)),[0,n,2*n].filter(a=>o-1-a>=0).map(a=>{const l=o-1-a;return e.jsx("line",{x1:l*d+d/2,y1:m-x,x2:(o-1)*d+d/2,y2:m-2*x-4+x,stroke:"#8b5cf6",strokeWidth:1.5,opacity:.6},a)}),e.jsx("text",{x:i-10,y:m-x-10,textAnchor:"end",className:"text-[10px] fill-violet-600 font-semibold",children:"output"}),e.jsx("text",{x:i-10,y:m-6,textAnchor:"end",className:"text-[10px] fill-gray-500",children:"input"})]}),e.jsxs("p",{className:"mt-2 text-center text-xs text-gray-500 dark:text-gray-400",children:["Kernel size 3 with dilation ",n,": reads positions t, t-",n,", t-",2*n," (causal — no future information)"]})]})}function O(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Temporal Convolutional Networks (TCNs) use causal dilated convolutions to model sequences. They achieve large receptive fields with fewer layers than standard CNNs and can be parallelized more efficiently than RNNs during training."}),e.jsxs(_,{title:"Causal Dilated Convolution",children:[e.jsxs("p",{children:["A 1D convolution with dilation ",e.jsx(t.InlineMath,{math:"d"})," and kernel size ",e.jsx(t.InlineMath,{math:"k"})," applied causally:"]}),e.jsx(t.BlockMath,{math:"(x *_d f)(t) = \\sum_{i=0}^{k-1} f(i) \\cdot x_{t - d \\cdot i}"}),e.jsxs("p",{className:"mt-2",children:["Causal constraint: the output at time ",e.jsx(t.InlineMath,{math:"t"})," depends only on ",e.jsx(t.InlineMath,{math:"x_t, x_{t-d}, x_{t-2d}, \\ldots"})," (no future leakage)."]})]}),e.jsx(H,{}),e.jsxs(N,{title:"Receptive Field Growth",id:"tcn-receptive-field",children:[e.jsxs("p",{children:["With ",e.jsx(t.InlineMath,{math:"L"})," layers, kernel size ",e.jsx(t.InlineMath,{math:"k"}),", and exponential dilation ",e.jsx(t.InlineMath,{math:"d_\\ell = 2^\\ell"}),":"]}),e.jsx(t.BlockMath,{math:"R = 1 + (k-1) \\sum_{\\ell=0}^{L-1} 2^\\ell = 1 + (k-1)(2^L - 1)"}),e.jsxs("p",{children:["The receptive field grows ",e.jsx("strong",{children:"exponentially"})," with depth, while parameter count grows only linearly."]})]}),e.jsx(k,{title:"TCN vs LSTM",children:e.jsxs("p",{children:["With ",e.jsx(t.InlineMath,{math:"k=3"})," and ",e.jsx(t.InlineMath,{math:"L=8"})," layers, the TCN has a receptive field of ",e.jsx(t.InlineMath,{math:"1 + 2 \\times 255 = 511"})," time steps — comparable to an LSTM processing 511 steps, but fully parallelizable during training (no sequential dependency)."]})}),e.jsx(M,{title:"TCN Forecasting with Darts",code:`from darts import TimeSeries
from darts.models import TCNModel
from darts.dataprocessing.transformers import Scaler
import numpy as np
import pandas as pd

# Create sample time series
dates = pd.date_range("2020-01-01", periods=500, freq="D")
trend = np.linspace(0, 5, 500)
seasonal = 3 * np.sin(2 * np.pi * np.arange(500) / 7)
noise = np.random.randn(500) * 0.5
values = trend + seasonal + noise

series = TimeSeries.from_times_and_values(dates, values)
train, val = series.split_before(0.8)

# Scale data
scaler = Scaler()
train_scaled = scaler.fit_transform(train)
val_scaled = scaler.transform(val)

# TCN: causal dilated convolutions for sequence modeling
model = TCNModel(
    input_chunk_length=30,        # lookback window
    output_chunk_length=7,        # forecast horizon
    kernel_size=3,                # conv kernel size
    num_filters=64,               # channels per layer
    dilation_base=2,              # exponential dilation
    num_layers=4,                 # -> receptive field = 1 + 2*(2^4 - 1) = 31
    dropout=0.1,
    n_epochs=20,
)

model.fit(train_scaled)
forecast = model.predict(n=len(val_scaled))
forecast = scaler.inverse_transform(forecast)

# Evaluate
from darts.metrics import mape
error = mape(val, forecast)
print(f"TCN MAPE: {error:.2f}%")
print(f"Receptive field: {1 + 2 * (2**4 - 1)} time steps")`}),e.jsx(w,{type:"note",title:"When to Choose TCN over RNN",children:e.jsx("p",{children:"TCNs offer faster training via parallelism and stable gradients (no vanishing gradient problem). They are preferred when the effective context length is known and fixed. RNNs remain competitive when input lengths vary greatly or when online (streaming) inference with minimal memory is required."})})]})}const ue=Object.freeze(Object.defineProperty({__proto__:null,default:O},Symbol.toStringTag,{value:"Module"}));function $(){const[r,j]=v.useState(.7),s=10,n=28,i=2,m=Array.from({length:s*s},(o,d)=>{const x=Math.floor(d/s),a=d%s,l=Math.exp(-.3*Math.abs(x-a))+(Math.sin(x*3.7+a*2.3)*.5+.5)*.4,c=l>r;return{i:x,j:a,active:c,score:l}});return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"ProbSparse Attention Pattern"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Sparsity threshold: ",r.toFixed(2),e.jsx("input",{type:"range",min:.3,max:.95,step:.05,value:r,onChange:o=>j(parseFloat(o.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("span",{className:"text-xs text-gray-500",children:[m.filter(o=>o.active).length,"/",s*s," active (",(m.filter(o=>o.active).length/(s*s)*100).toFixed(0),"%)"]})]}),e.jsx("svg",{width:s*(n+i),height:s*(n+i),className:"mx-auto block",children:m.map((o,d)=>e.jsx("rect",{x:o.j*(n+i),y:o.i*(n+i),width:n,height:n,rx:3,fill:o.active?"#8b5cf6":"#f3f4f6",opacity:o.active?.3+o.score*.7:.3},d))}),e.jsxs("p",{className:"mt-2 text-center text-xs text-gray-500 dark:text-gray-400",children:["Only high-importance query-key pairs are computed, reducing ",e.jsx(t.InlineMath,{math:"O(L^2)"})," to ",e.jsx(t.InlineMath,{math:"O(L \\log L)"})]})]})}function V(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Standard Transformers struggle with long time series due to quadratic attention cost. Informer introduces ProbSparse attention, while Autoformer replaces attention with auto-correlation for efficient long-range forecasting."}),e.jsxs(_,{title:"ProbSparse Self-Attention (Informer)",children:[e.jsxs("p",{children:["Informer selects the top-",e.jsx(t.InlineMath,{math:"u"})," queries with highest KL-divergence from a uniform distribution:"]}),e.jsx(t.BlockMath,{math:"M(q_i, K) = \\max_j \\frac{q_i k_j^\\top}{\\sqrt{d}} - \\frac{1}{L}\\sum_j \\frac{q_i k_j^\\top}{\\sqrt{d}}"}),e.jsxs("p",{className:"mt-2",children:["Only the top-",e.jsx(t.InlineMath,{math:"u = c \\cdot \\ln L"})," queries attend to all keys, achieving ",e.jsx(t.InlineMath,{math:"O(L \\ln L)"})," complexity."]})]}),e.jsx($,{}),e.jsxs(_,{title:"Auto-Correlation Mechanism (Autoformer)",children:[e.jsx("p",{children:"Autoformer replaces dot-product attention with period-based dependencies via autocorrelation:"}),e.jsx(t.BlockMath,{math:"\\mathcal{R}_{XX}(\\tau) = \\frac{1}{L}\\sum_{t=1}^{L} x_t \\cdot x_{t-\\tau}"}),e.jsxs("p",{className:"mt-2",children:["Top-",e.jsx(t.InlineMath,{math:"k"})," periods are selected, and corresponding sub-series are aggregated with ",e.jsx(t.InlineMath,{math:"\\text{Roll}(V, \\tau)"})," alignment."]})]}),e.jsxs(N,{title:"Informer Distilling Operation",id:"informer-distill",children:[e.jsx("p",{children:"Between attention layers, Informer halves the sequence length via 1D convolution + max-pooling:"}),e.jsx(t.BlockMath,{math:"X_{j+1} = \\text{MaxPool}\\left(\\text{ELU}\\left(\\text{Conv1d}(X_j)\\right)\\right)"}),e.jsxs("p",{children:["This creates a pyramidal encoder, reducing the total memory from ",e.jsx(t.InlineMath,{math:"O(L^2)"})," to ",e.jsx(t.InlineMath,{math:"O(L \\log L)"}),"."]})]}),e.jsxs(k,{title:"Autoformer Series Decomposition",children:[e.jsx("p",{children:"Autoformer applies progressive decomposition at every layer. A moving average extracts the trend, and the remainder captures seasonality:"}),e.jsx(t.BlockMath,{math:"\\mathbf{x}_{\\text{trend}} = \\text{AvgPool}(\\text{Pad}(\\mathbf{x})), \\quad \\mathbf{x}_{\\text{season}} = \\mathbf{x} - \\mathbf{x}_{\\text{trend}}"})]}),e.jsx(M,{title:"Informer / Autoformer with HuggingFace",code:`from transformers import (
    InformerConfig, InformerForPrediction,
    AutoformerConfig, AutoformerForPrediction,
)
import torch

# Informer: ProbSparse attention + distilling encoder
config = InformerConfig(
    prediction_length=24,
    context_length=96,
    input_size=7,               # number of variates
    d_model=64,
    encoder_layers=2,
    decoder_layers=1,
    encoder_attention_heads=4,
    lags_sequence=[1, 7, 14],   # autoregressive lags
    num_time_features=2,
)
informer = InformerForPrediction(config)

# Autoformer: auto-correlation + series decomposition
auto_config = AutoformerConfig(
    prediction_length=24,
    context_length=96,
    input_size=7,
    d_model=64,
    encoder_layers=2,
    decoder_layers=1,
    moving_average=25,          # decomposition kernel
    lags_sequence=[1, 7, 14],
    num_time_features=2,
)
autoformer = AutoformerForPrediction(auto_config)

# Simulated input (batch=2, context=96, 7 variates)
past_values = torch.randn(2, 96, 7)
past_time = torch.randn(2, 96, 2)  # time features
future_time = torch.randn(2, 24, 2)

out = informer(
    past_values=past_values,
    past_time_features=past_time,
    future_time_features=future_time,
)
print(f"Informer forecast params: {out.params.shape}")
# SampleTSPredictionOutput contains distribution parameters
print(f"  -> Generates 24-step probabilistic forecast for 7 variates")`}),e.jsx(w,{type:"note",title:"Informer vs Autoformer",children:e.jsx("p",{children:"Informer keeps the attention paradigm but sparsifies it. Autoformer fundamentally changes the mechanism to auto-correlation, which naturally captures periodic patterns. Autoformer generally outperforms Informer on datasets with strong seasonality, while Informer can be more flexible for irregular patterns."})})]})}const ge=Object.freeze(Object.defineProperty({__proto__:null,default:V},Symbol.toStringTag,{value:"Module"}));function X(){const[r,j]=v.useState(8),[s,n]=v.useState(8),i=48,m=420,o=100,d=24,x=[];for(let c=0;c+r<=i;c+=s)x.push({start:c,end:c+r});const a=["#8b5cf6","#f97316","#06b6d4","#ec4899","#10b981","#f59e0b"],l=m/i;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Time Series Patching"}),e.jsxs("div",{className:"flex flex-wrap gap-4 mb-3",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Patch length: ",r,e.jsx("input",{type:"range",min:4,max:16,step:2,value:r,onChange:c=>j(parseInt(c.target.value)),className:"w-24 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Stride: ",s,e.jsx("input",{type:"range",min:2,max:16,step:2,value:s,onChange:c=>n(parseInt(c.target.value)),className:"w-24 accent-violet-500"})]}),e.jsxs("span",{className:"text-xs text-gray-500 self-center",children:[x.length," patches (tokens to Transformer)"]})]}),e.jsxs("svg",{width:m,height:o,className:"mx-auto block",children:[Array.from({length:i},(c,h)=>e.jsx("rect",{x:h*l+.5,y:10,width:l-1,height:d,rx:2,fill:"#e5e7eb"},h)),x.map((c,h)=>e.jsxs("g",{children:[e.jsx("rect",{x:c.start*l,y:50,width:(c.end-c.start)*l-1,height:d,rx:4,fill:a[h%a.length],opacity:.7}),e.jsxs("text",{x:(c.start+(c.end-c.start)/2)*l,y:67,textAnchor:"middle",className:"text-[9px] fill-white font-bold",children:["P",h+1]}),Array.from({length:c.end-c.start},(u,f)=>e.jsx("line",{x1:(c.start+f+.5)*l,y1:10+d,x2:(c.start+f+.5)*l,y2:50,stroke:a[h%a.length],strokeWidth:.5,opacity:.4},f))]},h)),e.jsxs("text",{x:2,y:8,className:"text-[9px] fill-gray-400",children:["Raw time steps (T=",i,")"]}),e.jsx("text",{x:2,y:88,className:"text-[9px] fill-gray-400",children:"Patches as Transformer tokens"})]})]})}function G(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"PatchTST applies Vision Transformer-inspired patching to time series, dramatically reducing the token count fed to the Transformer while preserving local semantic information. Combined with channel independence, it achieves state-of-the-art long-term forecasting results."}),e.jsxs(_,{title:"Time Series Patching",children:[e.jsxs("p",{children:["A univariate series ",e.jsx(t.InlineMath,{math:"\\mathbf{x} \\in \\mathbb{R}^L"})," is segmented into patches of length ",e.jsx(t.InlineMath,{math:"P"})," with stride ",e.jsx(t.InlineMath,{math:"S"}),":"]}),e.jsx(t.BlockMath,{math:"\\mathbf{p}_i = \\mathbf{x}_{iS : iS+P}, \\quad i = 0, \\ldots, \\left\\lfloor\\frac{L-P}{S}\\right\\rfloor"}),e.jsxs("p",{className:"mt-2",children:["Each patch is linearly projected to dimension ",e.jsx(t.InlineMath,{math:"d"}),", producing ",e.jsx(t.InlineMath,{math:"N = \\lfloor(L-P)/S\\rfloor + 1"})," tokens."]})]}),e.jsx(X,{}),e.jsxs(N,{title:"Complexity Reduction via Patching",id:"patch-complexity",children:[e.jsxs("p",{children:["Standard Transformer on raw time steps: ",e.jsx(t.InlineMath,{math:"O(L^2)"}),". With patching:"]}),e.jsx(t.BlockMath,{math:"O(N^2) = O\\!\\left(\\left(\\frac{L}{S}\\right)^2\\right)"}),e.jsxs("p",{children:["For ",e.jsx(t.InlineMath,{math:"L=512, P=S=16"}),": reduces from ",e.jsx(t.InlineMath,{math:"262{,}144"})," to ",e.jsx(t.InlineMath,{math:"1{,}024"})," attention computations (256x speedup)."]})]}),e.jsxs(_,{title:"Channel Independence",children:[e.jsxs("p",{children:["For multivariate series ",e.jsx(t.InlineMath,{math:"\\mathbf{X} \\in \\mathbb{R}^{C \\times L}"}),", channel independence processes each variable separately through the same Transformer:"]}),e.jsx(t.BlockMath,{math:"\\hat{\\mathbf{y}}_c = f_\\theta(\\mathbf{x}_c), \\quad c = 1, \\ldots, C"}),e.jsx("p",{className:"mt-2",children:"Shared weights across channels act as implicit regularization, preventing overfitting to spurious cross-channel correlations."})]}),e.jsx(k,{title:"Channel-Independent vs Channel-Mixing",children:e.jsx("p",{children:"Counterintuitively, channel independence often outperforms models that explicitly model cross-variate dependencies (like full multivariate attention). The shared backbone learns universal temporal patterns while avoiding overfitting to dataset-specific inter-variable relationships."})}),e.jsx(M,{title:"PatchTST with HuggingFace Transformers",code:`from transformers import PatchTSTConfig, PatchTSTForPrediction
import torch

# PatchTST: patching + channel independence + Transformer
config = PatchTSTConfig(
    num_input_channels=7,       # multivariate: 7 channels
    context_length=96,          # lookback window
    prediction_length=24,       # forecast horizon
    patch_length=16,            # each patch covers 16 time steps
    patch_stride=8,             # 50% overlap -> (96-16)//8 + 1 = 11 patches
    d_model=128,
    num_attention_heads=4,
    num_hidden_layers=3,
    feedforward_dim=256,
    dropout=0.1,
    channel_attention=False,    # channel independence (key design choice)
)
model = PatchTSTForPrediction(config)

# Input: (batch, seq_len, num_channels)
past_values = torch.randn(8, 96, 7)
outputs = model(past_values=past_values)

print(f"Input: {past_values.shape}")           # [8, 96, 7]
print(f"Forecast: {outputs.prediction_outputs.shape}")  # [8, 24, 7]
print(f"Patches per channel: {(96 - 16) // 8 + 1}")  # 11 tokens
print(f"Attention cost: 11^2 = 121 vs raw 96^2 = 9216 (76x reduction)")

# Self-supervised pre-training: mask random patches
config_ssl = PatchTSTConfig(
    num_input_channels=7, context_length=96, patch_length=16,
    patch_stride=8, d_model=128, num_attention_heads=4,
    num_hidden_layers=3, mask_type="random", random_mask_ratio=0.4,
)
# Use PatchTSTForPretraining for masked patch reconstruction`}),e.jsx(w,{type:"note",title:"Self-Supervised Pre-Training",children:e.jsx("p",{children:"PatchTST supports masked patch pre-training analogous to BERT: randomly mask patches, reconstruct them, then fine-tune on the forecasting objective. This can improve performance by 2-5% on benchmarks, especially with limited labeled data."})})]})}const ye=Object.freeze(Object.defineProperty({__proto__:null,default:G},Symbol.toStringTag,{value:"Module"}));function K(){const[r,j]=v.useState(3),s=380,n=160,i=Array.from({length:8},(c,h)=>{const u=h+1,f=2.5*Math.pow(u,-.4)+.3;return{x:u,y:f}}),m=9,o=0,d=3,x=(c,h)=>`${c/m*s},${n-(h-o)/(d-o)*n}`,a=i.map((c,h)=>`${h===0?"M":"L"}${x(c.x,c.y)}`).join(" "),l=2.5*Math.pow(r,-.4)+.3;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Foundation Model Scaling"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Training data (log scale): ",r,e.jsx("input",{type:"range",min:1,max:8,step:.5,value:r,onChange:c=>j(parseFloat(c.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("svg",{width:s,height:n,className:"mx-auto block",children:[e.jsx("line",{x1:0,y1:n,x2:s,y2:n,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("line",{x1:0,y1:0,x2:0,y2:n,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("path",{d:a,fill:"none",stroke:"#8b5cf6",strokeWidth:2.5}),e.jsx("circle",{cx:r/m*s,cy:n-(l-o)/(d-o)*n,r:5,fill:"#f97316"}),e.jsxs("text",{x:r/m*s+8,y:n-(l-o)/(d-o)*n+4,className:"text-[10px] fill-orange-500",children:["loss = ",l.toFixed(2)]}),e.jsx("text",{x:s/2,y:n-4,textAnchor:"middle",className:"text-[9px] fill-gray-400",children:"log(training data)"}),e.jsx("text",{x:4,y:12,className:"text-[9px] fill-gray-400",children:"loss"})]}),e.jsx("p",{className:"mt-2 text-center text-xs text-gray-500 dark:text-gray-400",children:"Zero-shot forecasting loss follows power-law scaling with pre-training data volume"})]})}function U(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Foundation models for time series aim to create pre-trained models that generalize across domains without task-specific training. Models like Chronos, TimeGPT, and Moirai demonstrate that LLM-style pre-training can transfer to temporal data."}),e.jsxs(_,{title:"Chronos: Tokenized Time Series Language Model",children:[e.jsx("p",{children:"Chronos maps real-valued time series into discrete tokens via quantization:"}),e.jsx(t.BlockMath,{math:"x_t \\xrightarrow{\\text{scale}} \\tilde{x}_t \\xrightarrow{\\text{bin}} b_t \\in \\{1, \\ldots, B\\}"}),e.jsx("p",{className:"mt-2",children:"A T5-based language model is trained autoregressively on these tokens, then generates probabilistic forecasts by sampling token sequences and de-quantizing."})]}),e.jsx(K,{}),e.jsxs(N,{title:"Zero-Shot Forecasting",id:"zero-shot-ts",children:[e.jsxs("p",{children:["A foundation model ",e.jsx(t.InlineMath,{math:"f_\\theta"})," pre-trained on corpus ",e.jsx(t.InlineMath,{math:"\\mathcal{D}_{\\text{pre}}"})," can forecast unseen series ",e.jsx(t.InlineMath,{math:"\\mathbf{x}_{\\text{new}}"}),":"]}),e.jsx(t.BlockMath,{math:"\\hat{\\mathbf{y}} = f_\\theta(\\mathbf{x}_{\\text{new}}) \\quad \\text{without any gradient updates}"}),e.jsxs("p",{children:["The quality depends on the diversity of ",e.jsx(t.InlineMath,{math:"\\mathcal{D}_{\\text{pre}}"})," and similarity to the target domain."]})]}),e.jsx(k,{title:"Moirai: Any-Variable, Any-Frequency",children:e.jsx("p",{children:"Moirai uses a mixture of parametric distributions and handles varying numbers of variates via a masked attention mechanism. It supports arbitrary prediction lengths and frequencies, making it the most flexible foundation model for time series to date."})}),e.jsx(A,{title:"Limitations of TS Foundation Models",children:e.jsx("p",{children:"Current foundation models can underperform domain-specific models on specialized datasets (e.g., medical or financial data with unique patterns). They also struggle with very long contexts and extreme distribution shifts. Always benchmark against a fine-tuned baseline before deploying zero-shot."})}),e.jsx(M,{title:"Using Chronos for Zero-Shot Forecasting",code:`import torch
import numpy as np
# pip install chronos-forecasting
# from chronos import ChronosPipeline

# Example usage (requires GPU + model download):
# pipeline = ChronosPipeline.from_pretrained(
#     "amazon/chronos-t5-small",
#     device_map="auto",
#     torch_dtype=torch.float32,
# )

# Simulated example of the Chronos workflow
context = torch.tensor(np.sin(np.linspace(0, 8*np.pi, 96)))

# Chronos quantization concept (simplified)
n_bins = 4096
scaled = (context - context.mean()) / (context.std() + 1e-8)
bins = torch.linspace(-3, 3, n_bins)
tokens = torch.bucketize(scaled, bins)
print(f"Context length: {len(context)}, Token range: [{tokens.min()}, {tokens.max()}]")

# In practice: pipeline.predict(context, prediction_length=24, num_samples=20)
# Returns (20, 24) samples for probabilistic forecasting
print("Zero-shot forecast: sample multiple trajectories -> quantile intervals")`}),e.jsx(w,{type:"note",title:"The Debate: Are TS Foundation Models Needed?",children:e.jsx("p",{children:"Unlike NLP where text has universal grammar, time series are highly domain-specific. Recent work shows simple baselines (linear models) can be competitive. Foundation models shine when labeled data is scarce, diverse series must be handled, or rapid prototyping is needed. The field is actively evolving."})})]})}const je=Object.freeze(Object.defineProperty({__proto__:null,default:U},Symbol.toStringTag,{value:"Module"}));function Y(){const[r,j]=v.useState(1.2),s=60,n=420,i=160,m=Array.from({length:s},(p,y)=>Math.sin(y*.3)+.3*Math.cos(y*.7)),o=[18,19,42,43,44],d=m.map((p,y)=>o.includes(y)?p+2.5:p),x=m.map((p,y)=>p+Math.sin(y*11.3)*.1),a=d.map((p,y)=>Math.abs(p-x[y])),l=-2,c=4,h=(p,y)=>`${p/s*n},${i*.6-(y-l)/(c-l)*i*.6}`,u=d.map((p,y)=>`${y===0?"M":"L"}${h(y,p)}`).join(" "),f=x.map((p,y)=>`${y===0?"M":"L"}${h(y,p)}`).join(" ");return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Reconstruction-Based Anomaly Detection"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Threshold: ",r.toFixed(1),e.jsx("input",{type:"range",min:.3,max:3,step:.1,value:r,onChange:p=>j(parseFloat(p.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("svg",{width:n,height:i,className:"mx-auto block",children:[e.jsx("path",{d:u,fill:"none",stroke:"#6b7280",strokeWidth:1.5}),e.jsx("path",{d:f,fill:"none",stroke:"#8b5cf6",strokeWidth:1.5,strokeDasharray:"4,3"}),a.map((p,y)=>{const b=y/s*n,g=p/3*40,T=p>r;return e.jsx("rect",{x:b,y:i-g,width:n/s-1,height:g,fill:T?"#ef4444":"#d1d5db",opacity:.7,rx:1},y)}),e.jsx("line",{x1:0,y1:i-r/3*40,x2:n,y2:i-r/3*40,stroke:"#ef4444",strokeWidth:1,strokeDasharray:"5,3"}),e.jsx("text",{x:n-4,y:i-r/3*40-4,textAnchor:"end",className:"text-[9px] fill-red-500",children:"threshold"})]}),e.jsxs("div",{className:"mt-2 flex justify-center gap-4 text-xs text-gray-500",children:[e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-gray-500"})," Original"]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-violet-500",style:{borderBottom:"1px dashed"}})," Reconstructed"]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-3 bg-red-400 rounded"})," Anomaly"]})]})]})}function Q(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Reconstruction-based anomaly detection trains an autoencoder on normal data. At inference, anomalies produce high reconstruction error because the model has never learned to reproduce abnormal patterns."}),e.jsxs(_,{title:"Reconstruction Error Anomaly Score",children:[e.jsxs("p",{children:["An autoencoder ",e.jsx(t.InlineMath,{math:"f_\\theta"})," maps a window ",e.jsx(t.InlineMath,{math:"\\mathbf{x}"})," to its reconstruction ",e.jsx(t.InlineMath,{math:"\\hat{\\mathbf{x}}"}),". The anomaly score is:"]}),e.jsx(t.BlockMath,{math:"a(\\mathbf{x}) = \\|\\mathbf{x} - f_\\theta(\\mathbf{x})\\|_2^2"}),e.jsxs("p",{className:"mt-2",children:["A point is flagged as anomalous if ",e.jsx(t.InlineMath,{math:"a(\\mathbf{x}) > \\tau"}),", where ",e.jsx(t.InlineMath,{math:"\\tau"})," is a threshold set on validation data."]})]}),e.jsx(Y,{}),e.jsxs(N,{title:"LSTM-Autoencoder Architecture",id:"lstm-ae",children:[e.jsx("p",{children:"The encoder LSTM compresses the input window into a fixed-size latent vector:"}),e.jsx(t.BlockMath,{math:"\\mathbf{z} = \\text{LSTM}_{\\text{enc}}(\\mathbf{x}_1, \\ldots, \\mathbf{x}_T) \\in \\mathbb{R}^d"}),e.jsxs("p",{children:["The decoder LSTM reconstructs the sequence from ",e.jsx(t.InlineMath,{math:"\\mathbf{z}"}),":"]}),e.jsx(t.BlockMath,{math:"\\hat{\\mathbf{x}}_t = \\text{MLP}(\\text{LSTM}_{\\text{dec}}(\\mathbf{z}, \\hat{\\mathbf{x}}_{t-1}))"})]}),e.jsx(k,{title:"Threshold Selection Strategies",children:e.jsxs("p",{children:["Common approaches: (1) fixed percentile (e.g., 99th) of training reconstruction errors, (2) mean + ",e.jsx(t.InlineMath,{math:"k\\sigma"})," of training errors, or (3) learned threshold via a small labeled validation set. Dynamic thresholds that adapt over time handle concept drift better than static ones."]})}),e.jsx(M,{title:"LSTM Autoencoder for Anomaly Detection",code:`import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden=64, latent=32, n_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden, n_layers, batch_first=True)
        self.compress = nn.Linear(hidden, latent)
        self.expand = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, n_layers, batch_first=True)
        self.output = nn.Linear(hidden, input_dim)

    def forward(self, x):  # x: (B, T, 1)
        _, (h_n, _) = self.encoder(x)
        z = self.compress(h_n[-1])               # (B, latent)
        z_exp = self.expand(z).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(z_exp)
        return self.output(dec_out)               # (B, T, 1)

# Train on normal data, detect anomalies via reconstruction error
model = LSTMAutoencoder()
normal_data = torch.sin(torch.linspace(0, 10*3.14, 200)).reshape(1, 200, 1)
recon = model(normal_data)
error = ((normal_data - recon)**2).mean(dim=-1).squeeze()
threshold = error.mean() + 3 * error.std()
print(f"Threshold: {threshold.item():.4f}")
print(f"Anomalous steps: {(error > threshold).sum().item()}")`}),e.jsx(w,{type:"note",title:"VAE for Richer Anomaly Scores",children:e.jsx("p",{children:"Variational autoencoders (VAEs) provide a principled anomaly score via the ELBO: both reconstruction error and KL divergence from the prior contribute. Points that map to unusual latent regions (high KL) are anomalous even if reconstruction appears acceptable."})})]})}const be=Object.freeze(Object.defineProperty({__proto__:null,default:Q},Symbol.toStringTag,{value:"Module"}));function Z(){const[r,j]=v.useState(2),s=50,n=400,i=150,m=Array.from({length:s},(b,g)=>{const T=Math.sin(g*.25)+.4*Math.cos(g*.6),S=g>=28&&g<=31?2.2:g===40?-1.8:0;return T+S}),o=m.map((b,g)=>Math.sin(g*.25)+.4*Math.cos(g*.6)),d=m.map((b,g)=>b-o[g]),x=Math.sqrt(d.reduce((b,g)=>b+g*g,0)/s),a=-3,l=4,c=(b,g)=>`${b/s*n},${i*.65-(g-a)/(l-a)*i*.65}`,h=m.map((b,g)=>`${g===0?"M":"L"}${c(g,b)}`).join(" "),u=o.map((b,g)=>`${g===0?"M":"L"}${c(g,b)}`).join(" "),f=o.map((b,g)=>c(g,b+r*x)),p=o.map((b,g)=>c(g,b-r*x)).reverse(),y=`M${f.join(" L")} L${p.join(" L")} Z`;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Forecast-Based Anomaly Detection"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Sensitivity (k-sigma): ",r.toFixed(1),e.jsx("input",{type:"range",min:1,max:4,step:.2,value:r,onChange:b=>j(parseFloat(b.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("svg",{width:n,height:i,className:"mx-auto block",children:[e.jsx("path",{d:y,fill:"#8b5cf6",opacity:.1}),e.jsx("path",{d:u,fill:"none",stroke:"#8b5cf6",strokeWidth:1.5,strokeDasharray:"4,3"}),e.jsx("path",{d:h,fill:"none",stroke:"#374151",strokeWidth:1.5}),m.map((b,g)=>{if(!(Math.abs(d[g])>r*x))return null;const[S,C]=c(g,b).split(",");return e.jsx("circle",{cx:parseFloat(S),cy:parseFloat(C),r:4,fill:"#ef4444"},g)})]}),e.jsxs("div",{className:"mt-2 flex justify-center gap-4 text-xs text-gray-500",children:[e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-gray-700"})," Actual"]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-violet-500"})," Forecast"]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-3 rounded-full bg-red-500"})," Anomaly"]})]})]})}function J(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Forecasting-based anomaly detection compares model predictions with actual observations. Large prediction errors signal anomalous behavior. This approach naturally handles non-stationary data since the model learns to track normal dynamics."}),e.jsxs(_,{title:"Prediction Error Anomaly Score",children:[e.jsxs("p",{children:["Given a trained forecaster ",e.jsx(t.InlineMath,{math:"\\hat{x}_{t} = f_\\theta(x_{t-L:t-1})"}),", the anomaly score at time ",e.jsx(t.InlineMath,{math:"t"})," is:"]}),e.jsx(t.BlockMath,{math:"a_t = \\frac{|x_t - \\hat{x}_t|}{\\hat{\\sigma}_t}"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"\\hat{\\sigma}_t"})," is the estimated prediction uncertainty. Points with ",e.jsx(t.InlineMath,{math:"a_t > k"})," (e.g., ",e.jsx(t.InlineMath,{math:"k=3"}),") are anomalies."]})]}),e.jsx(Z,{}),e.jsxs(N,{title:"Conformal Prediction Intervals",id:"conformal-anomaly",children:[e.jsx("p",{children:"For calibrated anomaly detection, compute nonconformity scores on a calibration set:"}),e.jsx(t.BlockMath,{math:"s_i = |x_i - \\hat{x}_i|, \\quad i \\in \\mathcal{D}_{\\text{cal}}"}),e.jsxs("p",{children:["The prediction interval at level ",e.jsx(t.InlineMath,{math:"1-\\alpha"})," uses the ",e.jsx(t.InlineMath,{math:"\\lceil(1-\\alpha)(n+1)\\rceil/n"})," quantile of ",e.jsx(t.InlineMath,{math:"\\{s_i\\}"}),":"]}),e.jsx(t.BlockMath,{math:"C_t = [\\hat{x}_t - q_{1-\\alpha},\\; \\hat{x}_t + q_{1-\\alpha}]"}),e.jsxs("p",{children:["This guarantees ",e.jsx(t.InlineMath,{math:"P(x_t \\in C_t) \\geq 1-\\alpha"})," without distributional assumptions."]})]}),e.jsx(k,{title:"Multi-Step Forecast Residuals",children:e.jsxs("p",{children:["For multi-step forecasting, compute anomaly scores across the full horizon. An anomaly at step ",e.jsx(t.InlineMath,{math:"h"})," is weighted by the expected error at that horizon:",e.jsx(t.InlineMath,{math:"a_{t+h} = |x_{t+h} - \\hat{x}_{t+h}| / \\hat{\\sigma}_h"}),", where",e.jsx(t.InlineMath,{math:"\\hat{\\sigma}_h"})," grows with ",e.jsx(t.InlineMath,{math:"h"}),"."]})}),e.jsx(M,{title:"Conformal Anomaly Detection",code:`import torch
import numpy as np

def conformal_anomaly_detector(model, cal_data, test_data, alpha=0.05):
    """Detect anomalies with guaranteed coverage via conformal prediction."""
    # Compute calibration nonconformity scores
    cal_preds = model(cal_data['X'])  # shape: (n_cal, 1)
    cal_scores = torch.abs(cal_data['y'] - cal_preds).squeeze().detach().numpy()

    # Conformal quantile
    n = len(cal_scores)
    q_level = np.ceil((1 - alpha) * (n + 1)) / n
    q_hat = np.quantile(cal_scores, min(q_level, 1.0))

    # Detect anomalies on test set
    test_preds = model(test_data['X'])
    test_scores = torch.abs(test_data['y'] - test_preds).squeeze().detach().numpy()
    anomalies = test_scores > q_hat

    return anomalies, q_hat

# Example with a simple model
n_cal, n_test = 200, 50
cal_scores = np.abs(np.random.randn(n_cal))  # normal residuals
q_hat = np.quantile(cal_scores, 0.95)
test_scores = np.concatenate([np.abs(np.random.randn(45)), np.abs(np.random.randn(5)) + 3])
anomalies = test_scores > q_hat
print(f"Threshold (95%): {q_hat:.3f}")
print(f"Anomalies detected: {anomalies.sum()}/{n_test}")`}),e.jsx(w,{type:"note",title:"Combining Forecast + Reconstruction",children:e.jsx("p",{children:"Ensemble approaches combine forecasting and reconstruction anomaly scores for robustness. Forecasting detects point anomalies well (sudden spikes), while reconstruction catches contextual anomalies (subtle distribution shifts within normal value ranges)."})})]})}const _e=Object.freeze(Object.defineProperty({__proto__:null,default:J},Symbol.toStringTag,{value:"Module"}));function ee(){const[r,j]=v.useState(5),s=10,n=26,i=2,m=Array.from({length:s},(a,l)=>Array.from({length:s},(c,h)=>{const u=Math.abs(l-h);return Math.exp(-u*.5)})),o=Array.from({length:s},(a,l)=>Array.from({length:s},(c,h)=>l===r||h===r?.1+Math.random()*.2:Math.exp(-Math.abs(l-h)*.4)*(.8+Math.random()*.2))),d=s*(n+i),x=d*2+60;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Prior vs Series Association"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Anomaly at position: ",r,e.jsx("input",{type:"range",min:0,max:9,step:1,value:r,onChange:a=>j(parseInt(a.target.value)),className:"w-28 accent-violet-500"})]}),e.jsx("div",{className:"overflow-x-auto",children:e.jsxs("svg",{width:x,height:d+30,className:"mx-auto block",children:[e.jsx("text",{x:d/2,y:12,textAnchor:"middle",className:"text-[10px] fill-violet-600 font-semibold",children:"Prior Association"}),m.map((a,l)=>a.map((c,h)=>e.jsx("rect",{x:h*(n+i),y:l*(n+i)+18,width:n,height:n,rx:2,fill:"#8b5cf6",opacity:c*.8},`p-${l}-${h}`))),e.jsx("text",{x:d+30+d/2,y:12,textAnchor:"middle",className:"text-[10px] fill-orange-600 font-semibold",children:"Series Association"}),o.map((a,l)=>a.map((c,h)=>e.jsx("rect",{x:d+60+h*(n+i),y:l*(n+i)+18,width:n,height:n,rx:2,fill:l===r||h===r?"#ef4444":"#f97316",opacity:c*.8},`s-${l}-${h}`)))]})}),e.jsx("p",{className:"mt-2 text-center text-xs text-gray-500 dark:text-gray-400",children:"Anomalous points show high discrepancy between prior (learned) and series (observed) associations"})]})}function te(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"The Anomaly Transformer introduces a novel association discrepancy framework that leverages the difference between prior and series associations in attention maps to detect anomalies without requiring labeled anomaly data."}),e.jsxs(_,{title:"Association Discrepancy",children:[e.jsx("p",{children:"For each time point, compute two association distributions:"}),e.jsx(t.BlockMath,{math:"\\text{Prior: } P_t \\sim \\mathcal{N}(t, \\sigma^2) \\quad \\text{(learned Gaussian kernel)}"}),e.jsx(t.BlockMath,{math:"\\text{Series: } S_t = \\text{Softmax}(Q_t K^\\top / \\sqrt{d})"}),e.jsx("p",{className:"mt-2",children:"The anomaly score uses the KL-divergence between them:"}),e.jsx(t.BlockMath,{math:"\\text{AssDis}(t) = \\text{KL}(P_t \\| S_t) + \\text{KL}(S_t \\| P_t)"})]}),e.jsx(ee,{}),e.jsxs(N,{title:"Minimax Association Learning",id:"anomaly-transformer-loss",children:[e.jsx("p",{children:"The Anomaly Transformer training objective uses a minimax strategy:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L} = \\|\\mathbf{x} - \\hat{\\mathbf{x}}\\|_2^2 - \\lambda \\cdot \\text{AssDis}(\\text{stop\\_grad}(P), S) + \\lambda \\cdot \\text{AssDis}(P, \\text{stop\\_grad}(S))"}),e.jsx("p",{children:"The prior association is encouraged to approach the series association (minimizing discrepancy), while the series association is pushed away (maximizing discrepancy). This amplifies the difference for anomalous points."})]}),e.jsx(k,{title:"Why Attention Reveals Anomalies",children:e.jsx("p",{children:"Normal points form strong associations with their temporal neighbors — attention concentrates on nearby, similar patterns. Anomalous points cannot find similar patterns, so their attention becomes diffuse (high entropy), creating a measurable discrepancy from the expected prior association."})}),e.jsx(M,{title:"Anomaly Transformer Scoring",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class AnomalyAttentionLayer(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        # Learnable prior: sigma parameter for each head
        self.sigma = nn.Parameter(torch.ones(n_heads) * 1.0)

    def forward(self, x):  # x: (B, L, D)
        B, L, D = x.shape
        qkv = self.W_qkv(x).reshape(B, L, 3, self.n_heads, self.d_k)
        Q, K, V = qkv.unbind(dim=2)  # each: (B, L, H, d_k)
        Q, K, V = [t.transpose(1, 2) for t in (Q, K, V)]

        # Series association (standard attention)
        series_assoc = F.softmax(Q @ K.transpose(-2, -1) / self.d_k**0.5, dim=-1)

        # Prior association (Gaussian kernel)
        positions = torch.arange(L, device=x.device).float()
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1))**2
        prior_assoc = F.softmax(-dist / (2 * self.sigma.view(1, -1, 1, 1)**2 + 1e-8), dim=-1)
        prior_assoc = prior_assoc.expand(B, -1, -1, -1)

        # Association discrepancy per time point
        kl_ps = (prior_assoc * (prior_assoc.log() - series_assoc.log() + 1e-8)).sum(-1)
        kl_sp = (series_assoc * (series_assoc.log() - prior_assoc.log() + 1e-8)).sum(-1)
        discrepancy = (kl_ps + kl_sp).mean(dim=1)  # average over heads: (B, L)

        return (series_assoc @ V).transpose(1, 2).reshape(B, L, D), discrepancy

layer = AnomalyAttentionLayer()
x = torch.randn(4, 32, 64)
out, disc = layer(x)
print(f"Output: {out.shape}, Discrepancy: {disc.shape}")
print(f"Top anomaly scores: {disc[0].topk(3).values.tolist()}")`}),e.jsx(w,{type:"note",title:"Beyond Anomaly Transformer",children:e.jsx("p",{children:"Other Transformer-based approaches include TranAD (adversarial training with attention), and GDN (graph deviation network for multivariate data). The key insight shared across methods: attention patterns contain rich information about temporal relationships that anomalies disrupt."})})]})}const ve=Object.freeze(Object.defineProperty({__proto__:null,default:te},Symbol.toStringTag,{value:"Module"}));function ae(){const[r,j]=v.useState("cnn"),s={cnn:{name:"CNN (InceptionTime)",layers:["Conv 1x10","Conv 1x20","Conv 1x40","MaxPool","GAP","FC"],color:"#8b5cf6"},rnn:{name:"LSTM Classifier",layers:["LSTM-1","LSTM-2","Last h_T","FC","Softmax",""],color:"#f97316"},transformer:{name:"TST Classifier",layers:["Patch","Pos Enc","Encoder x3","CLS Token","FC",""],color:"#06b6d4"}},n=s[r],i=60,m=36,o=10,d=n.layers.filter(x=>x).length*(i+o)+20;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Classification Architecture Comparison"}),e.jsx("div",{className:"flex gap-2 mb-3",children:Object.entries(s).map(([x,a])=>e.jsx("button",{onClick:()=>j(x),className:`px-3 py-1 rounded-full text-xs font-medium transition-colors ${r===x?"bg-violet-500 text-white":"bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"}`,children:a.name},x))}),e.jsx("div",{className:"overflow-x-auto",children:e.jsxs("svg",{width:d,height:80,className:"mx-auto block",children:[n.layers.filter(x=>x).map((x,a)=>e.jsxs("g",{children:[a>0&&e.jsx("line",{x1:a*(i+o)-o+5,y1:40,x2:a*(i+o)+5,y2:40,stroke:"#d1d5db",strokeWidth:1.5,markerEnd:"url(#arrowC)"}),e.jsx("rect",{x:a*(i+o)+5,y:22,width:i,height:m,rx:6,fill:n.color,opacity:.15+a/n.layers.length*.3,stroke:n.color,strokeWidth:1.5}),e.jsx("text",{x:a*(i+o)+5+i/2,y:44,textAnchor:"middle",className:"text-[9px] fill-gray-700 dark:fill-gray-300",children:x})]},a)),e.jsx("defs",{children:e.jsx("marker",{id:"arrowC",viewBox:"0 0 10 10",refX:9,refY:5,markerWidth:4,markerHeight:4,orient:"auto",children:e.jsx("path",{d:"M 0 0 L 10 5 L 0 10 z",fill:"#d1d5db"})})})]})}),e.jsxs("p",{className:"mt-2 text-center text-xs text-gray-500 dark:text-gray-400",children:[r==="cnn"&&"Multi-scale convolutions capture patterns at different temporal resolutions",r==="rnn"&&"Sequential processing captures order-dependent features, uses final hidden state",r==="transformer"&&"Self-attention over patches with a learnable classification token"]})]})}function se(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Time series classification assigns a label to an entire sequence based on its temporal patterns. Applications include ECG diagnosis, activity recognition, and industrial fault detection. Deep learning approaches have largely replaced traditional distance-based methods on standard benchmarks."}),e.jsxs(_,{title:"Time Series Classification Task",children:[e.jsxs("p",{children:["Given a labeled dataset ",e.jsx(t.InlineMath,{math:"\\{(\\mathbf{x}_i, y_i)\\}_{i=1}^N"})," where ",e.jsx(t.InlineMath,{math:"\\mathbf{x}_i \\in \\mathbb{R}^T"})," and ",e.jsx(t.InlineMath,{math:"y_i \\in \\{1, \\ldots, K\\}"}),":"]}),e.jsx(t.BlockMath,{math:"f_\\theta : \\mathbb{R}^T \\to \\Delta^K, \\qquad \\hat{y} = \\arg\\max_k f_\\theta(\\mathbf{x})_k"}),e.jsxs("p",{className:"mt-2",children:["The model maps a variable-length time series to a probability distribution over ",e.jsx(t.InlineMath,{math:"K"})," classes."]})]}),e.jsx(ae,{}),e.jsxs(N,{title:"InceptionTime Architecture",id:"inceptiontime",children:[e.jsx("p",{children:"InceptionTime applies multiple parallel convolutions with different kernel sizes at each layer:"}),e.jsx(t.BlockMath,{math:"\\mathbf{h}_\\ell = \\text{BN}\\left(\\sum_{k \\in \\{10,20,40\\}} \\text{Conv}_{1 \\times k}(\\mathbf{h}_{\\ell-1}) + \\text{MaxPool}_{3}(\\mathbf{h}_{\\ell-1})\\right)"}),e.jsxs("p",{children:["Global Average Pooling (GAP) aggregates temporal features: ",e.jsx(t.InlineMath,{math:"\\bar{\\mathbf{h}} = \\frac{1}{T}\\sum_t \\mathbf{h}_t"}),", followed by a linear classifier."]})]}),e.jsx(k,{title:"ResNet Baseline",children:e.jsx("p",{children:"A simple 1D ResNet with 3 residual blocks of (Conv-BN-ReLU) x 3 and GAP achieves competitive results on the UCR archive (128 datasets). It serves as the standard deep learning baseline, outperforming most non-DL methods with minimal tuning."})}),e.jsx(M,{title:"InceptionTime Module in PyTorch",code:`import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch=32):
        super().__init__()
        self.conv10 = nn.Conv1d(in_ch, out_ch, kernel_size=10, padding=4)
        self.conv20 = nn.Conv1d(in_ch, out_ch, kernel_size=20, padding=9)
        self.conv40 = nn.Conv1d(in_ch, out_ch, kernel_size=40, padding=19)
        self.mp_conv = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
        )
        self.bn = nn.BatchNorm1d(out_ch * 4)

    def forward(self, x):  # x: (B, C, T)
        c10 = self.conv10(x)[:, :, :x.size(2)]
        c20 = self.conv20(x)[:, :, :x.size(2)]
        c40 = self.conv40(x)[:, :, :x.size(2)]
        mp = self.mp_conv(x)
        return torch.relu(self.bn(torch.cat([c10, c20, c40, mp], dim=1)))

class InceptionTime(nn.Module):
    def __init__(self, in_ch=1, n_classes=5, depth=6):
        super().__init__()
        ch = 32 * 4  # output channels per inception block
        self.blocks = nn.ModuleList([InceptionBlock(in_ch if i == 0 else ch) for i in range(depth)])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(ch, n_classes)

    def forward(self, x):  # x: (B, 1, T)
        for block in self.blocks:
            x = block(x)
        return self.fc(self.gap(x).squeeze(-1))

model = InceptionTime(in_ch=1, n_classes=5)
x = torch.randn(8, 1, 128)
print(f"Predictions: {model(x).shape}")  # (8, 5)`}),e.jsx(w,{type:"note",title:"Ensembling for Robustness",children:e.jsx("p",{children:"The original InceptionTime paper uses an ensemble of 5 models with different random initializations. This reduces variance significantly and is a common practice in time series classification where datasets are often small (tens to hundreds of samples)."})})]})}const ke=Object.freeze(Object.defineProperty({__proto__:null,default:se},Symbol.toStringTag,{value:"Module"}));function ne(){const[r,j]=v.useState(1.5),s=12,n=Array.from({length:s},(u,f)=>Math.sin(f*.5)*2),i=Array.from({length:s},(u,f)=>Math.sin(f*.5/r)*2),m=380,o=140,d=50,x=90,a=m/(s+1),l=(u,f)=>({x:(u+1)*a,y:d/2-f*10+10}),c=(u,f)=>({x:(u+1)*a,y:x+d/2-f*10}),h=n.map((u,f)=>{const p=Math.min(s-1,Math.round(f/r));return{i:f,j:p}});return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Dynamic Time Warping Alignment"}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3",children:["Time warp: ",r.toFixed(1),"x",e.jsx("input",{type:"range",min:.6,max:2.5,step:.1,value:r,onChange:u=>j(parseFloat(u.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("svg",{width:m,height:o,className:"mx-auto block",children:[h.map((u,f)=>{const p=l(u.i,n[u.i]),y=c(u.j,i[u.j]);return e.jsx("line",{x1:p.x,y1:p.y,x2:y.x,y2:y.y,stroke:"#8b5cf6",strokeWidth:.8,opacity:.3},f)}),n.map((u,f)=>{const p=l(f,u);return e.jsx("circle",{cx:p.x,cy:p.y,r:3,fill:"#8b5cf6"},`a-${f}`)}),i.map((u,f)=>{const p=c(f,u);return e.jsx("circle",{cx:p.x,cy:p.y,r:3,fill:"#f97316"},`b-${f}`)}),e.jsx("text",{x:8,y:d/2+10,className:"text-[9px] fill-violet-500",children:"A"}),e.jsx("text",{x:8,y:x+d/2,className:"text-[9px] fill-orange-500",children:"B"})]}),e.jsx("p",{className:"mt-1 text-center text-xs text-gray-500 dark:text-gray-400",children:"DTW finds the optimal alignment between time-warped versions of the same pattern"})]})}function re(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Dynamic Time Warping (DTW) and shapelets are classical time series methods that deep learning has begun to incorporate. DTW provides a time-invariant distance measure, while shapelets identify discriminative local patterns."}),e.jsxs(_,{title:"Dynamic Time Warping (DTW)",children:[e.jsxs("p",{children:["DTW finds the minimum-cost alignment between two sequences ",e.jsx(t.InlineMath,{math:"\\mathbf{a}"})," and ",e.jsx(t.InlineMath,{math:"\\mathbf{b}"})," via dynamic programming:"]}),e.jsx(t.BlockMath,{math:"D(i, j) = d(a_i, b_j) + \\min\\{D(i-1, j),\\; D(i, j-1),\\; D(i-1, j-1)\\}"}),e.jsxs("p",{className:"mt-2",children:["The DTW distance is ",e.jsx(t.InlineMath,{math:"D(M, N)"}),", allowing one-to-many point matchings to handle temporal distortions."]})]}),e.jsx(ne,{}),e.jsxs(N,{title:"Soft-DTW: Differentiable DTW",id:"soft-dtw",children:[e.jsxs("p",{children:["Soft-DTW replaces the hard ",e.jsx(t.InlineMath,{math:"\\min"})," with a smooth minimum for gradient-based learning:"]}),e.jsx(t.BlockMath,{math:"\\text{min}^{\\gamma}(a_1, \\ldots, a_n) = -\\gamma \\log \\sum_i e^{-a_i/\\gamma}"}),e.jsx(t.BlockMath,{math:"D^\\gamma(i, j) = d(a_i, b_j) + \\text{min}^{\\gamma}\\{D^\\gamma(i-1,j),\\; D^\\gamma(i,j-1),\\; D^\\gamma(i-1,j-1)\\}"}),e.jsxs("p",{children:["As ",e.jsx(t.InlineMath,{math:"\\gamma \\to 0"}),", soft-DTW recovers exact DTW."]})]}),e.jsxs(_,{title:"Shapelets",children:[e.jsxs("p",{children:["A shapelet ",e.jsx(t.InlineMath,{math:"\\mathbf{s} \\in \\mathbb{R}^l"})," (",e.jsx(t.InlineMath,{math:"l \\ll T"}),") is a subsequence pattern that is maximally discriminative between classes:"]}),e.jsx(t.BlockMath,{math:"d_{\\text{shapelet}}(\\mathbf{x}, \\mathbf{s}) = \\min_{t} \\|\\mathbf{x}_{t:t+l} - \\mathbf{s}\\|_2"}),e.jsx("p",{className:"mt-2",children:"Learned shapelets are initialized randomly and optimized end-to-end via gradient descent."})]}),e.jsx(k,{title:"Learned Shapelets as Soft Convolutions",children:e.jsx("p",{children:"A learned shapelet layer computes soft minimum distances using a smooth approximation. This is equivalent to a special type of convolutional layer where the kernel represents a prototype pattern and the output measures similarity rather than linear correlation."})}),e.jsx(M,{title:"Soft-DTW & Learned Shapelets",code:`import torch
import torch.nn as nn

def soft_dtw(x, y, gamma=0.1):
    """Differentiable DTW distance (simplified)."""
    M, N = x.size(0), y.size(0)
    D = torch.full((M + 1, N + 1), float('inf'))
    D[0, 0] = 0
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            cost = (x[i-1] - y[j-1])**2
            neighbors = torch.stack([D[i-1,j], D[i,j-1], D[i-1,j-1]])
            D[i, j] = cost + (-gamma * torch.logsumexp(-neighbors / gamma, dim=0))
    return D[M, N]

class LearnedShapelets(nn.Module):
    def __init__(self, n_shapelets=10, shapelet_len=15, n_classes=5):
        super().__init__()
        self.shapelets = nn.Parameter(torch.randn(n_shapelets, shapelet_len))
        self.fc = nn.Linear(n_shapelets, n_classes)

    def forward(self, x):  # x: (B, T)
        B, T = x.shape
        L = self.shapelets.size(1)
        dists = []
        for s in self.shapelets:
            # Sliding distance to each shapelet
            d = torch.stack([((x[:, t:t+L] - s)**2).sum(-1) for t in range(T - L + 1)], dim=1)
            dists.append(d.min(dim=1).values)  # min distance
        features = torch.stack(dists, dim=1)  # (B, n_shapelets)
        return self.fc(features)

model = LearnedShapelets()
x = torch.randn(8, 128)
print(f"Predictions: {model(x).shape}")  # (8, 5)`}),e.jsx(I,{title:"Exercise: DTW Complexity",children:e.jsxs("p",{children:["Standard DTW has ",e.jsx(t.InlineMath,{math:"O(MN)"})," complexity. For a dataset of ",e.jsx(t.InlineMath,{math:"n"})," training series and 1-NN classification, the total cost is ",e.jsx(t.InlineMath,{math:"O(n \\cdot M \\cdot N)"})," per test query. How does the Sakoe-Chiba band constraint with width ",e.jsx(t.InlineMath,{math:"w"})," reduce this? What is the new complexity?"]})})]})}const Me=Object.freeze(Object.defineProperty({__proto__:null,default:re},Symbol.toStringTag,{value:"Module"}));function ie(){const[r,j]=v.useState("fused"),s=4,n=30,i=400,m=160,o=m/s,d=Array.from({length:s},(a,l)=>Array.from({length:n},(c,h)=>Math.sin(h*.3*(l+1))+Math.cos(h*.15)*(l*.3))),x=["#8b5cf6","#f97316","#06b6d4","#10b981"];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Multivariate Strategies"}),e.jsx("div",{className:"flex gap-2 mb-3",children:[{key:"fused",label:"Early Fusion"},{key:"independent",label:"Channel Independent"},{key:"attention",label:"Cross-Channel Attention"}].map(a=>e.jsx("button",{onClick:()=>j(a.key),className:`px-3 py-1 rounded-full text-xs font-medium transition-colors ${r===a.key?"bg-violet-500 text-white":"bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"}`,children:a.label},a.key))}),e.jsxs("svg",{width:i,height:m,className:"mx-auto block",children:[d.map((a,l)=>{const c=l*o,h=a.map((f,p)=>{const y=p/n*i,b=c+o/2-f*6;return`${p===0?"M":"L"}${y},${b}`}).join(" "),u=r==="independent"?l===0?1:.3:1;return e.jsx("path",{d:h,fill:"none",stroke:x[l],strokeWidth:1.5,opacity:u},l)}),r==="attention"&&e.jsxs(e.Fragment,{children:[e.jsx("line",{x1:i*.5,y1:o*.5,x2:i*.5,y2:o*1.5,stroke:"#9ca3af",strokeWidth:1,strokeDasharray:"3,2"}),e.jsx("line",{x1:i*.5,y1:o*1.5,x2:i*.5,y2:o*2.5,stroke:"#9ca3af",strokeWidth:1,strokeDasharray:"3,2"}),e.jsx("line",{x1:i*.5,y1:o*2.5,x2:i*.5,y2:o*3.5,stroke:"#9ca3af",strokeWidth:1,strokeDasharray:"3,2"}),e.jsx("text",{x:i*.5+4,y:o*2,className:"text-[8px] fill-gray-400",children:"attn"})]}),r==="fused"&&e.jsx("rect",{x:0,y:0,width:i,height:m,fill:"#8b5cf6",opacity:.04,rx:4})]}),e.jsxs("p",{className:"mt-2 text-center text-xs text-gray-500 dark:text-gray-400",children:[r==="fused"&&"Stack all channels as input features — captures cross-channel patterns from the start",r==="independent"&&"Process each channel separately, merge later — avoids spurious correlations",r==="attention"&&"Cross-channel attention learns which inter-variable dependencies matter"]})]})}function oe(){return e.jsxs("div",{className:"space-y-6",children:[e.jsxs("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:["Multivariate time series classification (MTSC) operates on ",e.jsx(t.InlineMath,{math:"C"})," correlated channels simultaneously. The key challenge is modeling both temporal dynamics within each channel and dependencies across channels effectively."]}),e.jsxs(_,{title:"Multivariate TS Classification",children:[e.jsxs("p",{children:["Given input ",e.jsx(t.InlineMath,{math:"\\mathbf{X} \\in \\mathbb{R}^{C \\times T}"})," with ",e.jsx(t.InlineMath,{math:"C"})," variables and ",e.jsx(t.InlineMath,{math:"T"})," time steps:"]}),e.jsx(t.BlockMath,{math:"f_\\theta : \\mathbb{R}^{C \\times T} \\to \\Delta^K"}),e.jsx("p",{className:"mt-2",children:"Three strategies for handling multiple channels: early fusion, channel independence, and cross-channel attention."})]}),e.jsx(ie,{}),e.jsxs(N,{title:"Cross-Variable Attention",id:"cross-var-attention",children:[e.jsxs("p",{children:["Given per-channel representations ",e.jsx(t.InlineMath,{math:"\\mathbf{H} \\in \\mathbb{R}^{C \\times d}"}),", cross-variable attention computes:"]}),e.jsx(t.BlockMath,{math:"\\text{CVAttn}(\\mathbf{H}) = \\text{Softmax}\\!\\left(\\frac{\\mathbf{H}\\mathbf{H}^\\top}{\\sqrt{d}}\\right)\\mathbf{H}"}),e.jsxs("p",{children:["This ",e.jsx(t.InlineMath,{math:"C \\times C"})," attention matrix captures pairwise channel dependencies, scaling to hundreds of variables when ",e.jsx(t.InlineMath,{math:"C \\ll T"}),"."]})]}),e.jsx(k,{title:"ROCKET: Random Convolutional Kernels",children:e.jsx("p",{children:"ROCKET generates thousands of random 1D convolutional kernels (varying lengths, dilations, biases) and extracts two features per kernel: max value and proportion of positive values. These features are fed to a simple linear classifier, achieving near state-of-the-art accuracy at a fraction of the training cost."})}),e.jsx(A,{title:"Curse of Dimensionality in MTSC",children:e.jsx("p",{children:"Adding more variables does not always improve classification. Irrelevant channels add noise that can degrade performance. Use channel selection (dropout, learned gating, or mutual information) to identify which variables contain discriminative information for each class."})}),e.jsx(M,{title:"Multivariate TS Classifier with Cross-Channel Attention",code:`import torch
import torch.nn as nn

class MultivariateTSClassifier(nn.Module):
    def __init__(self, n_vars=6, seq_len=128, d_model=64, n_classes=4, n_heads=4):
        super().__init__()
        # Per-channel temporal encoder (shared weights)
        self.temporal_enc = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=8, padding=3),
            nn.ReLU(), nn.BatchNorm1d(d_model),
            nn.AdaptiveAvgPool1d(1)  # global average pooling
        )
        # Cross-variable attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(n_vars * d_model, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):  # x: (B, C, T)
        B, C, T = x.shape
        # Encode each channel independently
        h = torch.stack([self.temporal_enc(x[:, c:c+1, :]).squeeze(-1) for c in range(C)], dim=1)
        # Cross-channel attention: (B, C, d_model)
        h_attn, _ = self.cross_attn(h, h, h)
        h = self.norm(h + h_attn)
        # Flatten and classify
        return self.classifier(h.reshape(B, -1))

model = MultivariateTSClassifier(n_vars=6, seq_len=128, n_classes=4)
x = torch.randn(16, 6, 128)
print(f"Output: {model(x).shape}")  # (16, 4)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")`}),e.jsx(w,{type:"note",title:"Benchmarking: UEA Archive",children:e.jsx("p",{children:"The UEA Multivariate Time Series Archive contains 30 benchmark datasets spanning domains like motion capture, medical sensors, and audio. When evaluating MTSC models, report critical difference diagrams across datasets rather than cherry-picking individual results. No single method dominates all datasets."})})]})}const Ne=Object.freeze(Object.defineProperty({__proto__:null,default:oe},Symbol.toStringTag,{value:"Module"}));export{me as a,xe as b,pe as c,fe as d,ue as e,ge as f,ye as g,je as h,be as i,_e as j,ve as k,ke as l,Me as m,Ne as n,he as s};

import{j as e,r as p}from"./vendor-DpISuAX6.js";import{r as t}from"./vendor-katex-CbWCYdth.js";import{D as m,E as u,P as j,N as f,T as b,W as P,a as T}from"./subject-01-foundations-D0A1VJsr.js";function I(){const[a,c]=p.useState(0),i=["S0","S1","S2","S3"],l=[0,1,-1,5],s=[60,150,240,330];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"MDP Trajectory Visualizer"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Step: ",a,e.jsx("input",{type:"range",min:0,max:3,step:1,value:a,onChange:n=>c(parseInt(n.target.value)),className:"w-32 accent-violet-500"})]}),e.jsxs("span",{className:"text-sm text-violet-600 dark:text-violet-400 font-medium",children:["Reward: ",l[a]," | Cumulative: ",l.slice(0,a+1).reduce((n,r)=>n+r,0)]})]}),e.jsxs("svg",{width:400,height:100,className:"mx-auto block",children:[i.map((n,r)=>e.jsxs("g",{children:[e.jsx("circle",{cx:s[r],cy:50,r:22,fill:r===a?"#7c3aed":"#e5e7eb",stroke:"#7c3aed",strokeWidth:2}),e.jsx("text",{x:s[r],y:55,textAnchor:"middle",fill:r===a?"white":"#374151",fontSize:13,fontWeight:"bold",children:n}),r<3&&e.jsx("line",{x1:s[r]+24,y1:50,x2:s[r+1]-24,y2:50,stroke:"#9ca3af",strokeWidth:1.5,markerEnd:"url(#arrow)"}),e.jsxs("text",{x:s[r],y:88,textAnchor:"middle",fill:"#7c3aed",fontSize:11,children:["r=",l[r]]})]},n)),e.jsx("defs",{children:e.jsx("marker",{id:"arrow",markerWidth:"8",markerHeight:"6",refX:"8",refY:"3",orient:"auto",children:e.jsx("path",{d:"M0,0 L8,3 L0,6",fill:"#9ca3af"})})})]})]})}function z(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"A Markov Decision Process (MDP) provides the mathematical framework for sequential decision-making under uncertainty. Nearly all of reinforcement learning builds on this foundation."}),e.jsxs(m,{title:"Markov Decision Process",children:[e.jsxs("p",{children:["An MDP is a tuple ",e.jsx(t.InlineMath,{math:"(\\mathcal{S}, \\mathcal{A}, P, R, \\gamma)"})," where:"]}),e.jsx(t.BlockMath,{math:"\\mathcal{S} \\text{ (states)}, \\quad \\mathcal{A} \\text{ (actions)}, \\quad P(s'|s,a) \\text{ (transitions)}, \\quad R(s,a) \\text{ (reward)}, \\quad \\gamma \\in [0,1) \\text{ (discount)}"}),e.jsxs("p",{className:"mt-2",children:["The ",e.jsx("strong",{children:"Markov property"}),": ",e.jsx(t.InlineMath,{math:"P(s_{t+1}|s_t, a_t) = P(s_{t+1}|s_0,...,s_t, a_0,...,a_t)"})]})]}),e.jsxs(m,{title:"Policy",children:[e.jsxs("p",{children:["A policy ",e.jsx(t.InlineMath,{math:"\\pi"})," maps states to action distributions:"]}),e.jsx(t.BlockMath,{math:"\\pi(a|s) = P(A_t = a \\mid S_t = s)"}),e.jsxs("p",{className:"mt-2",children:["The goal is to find ",e.jsx(t.InlineMath,{math:"\\pi^*"})," that maximizes the expected discounted return:"]}),e.jsx(t.BlockMath,{math:"G_t = \\sum_{k=0}^{\\infty} \\gamma^k R_{t+k+1}"})]}),e.jsx(I,{}),e.jsxs(u,{title:"GridWorld MDP",children:[e.jsx("p",{children:"Consider a 4x4 grid. The agent starts at (0,0) and the goal is (3,3)."}),e.jsx(t.BlockMath,{math:"\\mathcal{S} = \\{(i,j) : 0 \\le i,j \\le 3\\}, \\quad \\mathcal{A} = \\{\\uparrow, \\downarrow, \\leftarrow, \\rightarrow\\}"}),e.jsxs("p",{children:["With ",e.jsx(t.InlineMath,{math:"R = -1"})," per step and ",e.jsx(t.InlineMath,{math:"R = 0"})," at the goal, the agent learns to take the shortest path."]})]}),e.jsx(j,{title:"Simple MDP Environment in Python",code:`import numpy as np

class GridWorldMDP:
    def __init__(self, size=4, gamma=0.99):
        self.size = size
        self.gamma = gamma
        self.state = (0, 0)
        self.goal = (size - 1, size - 1)
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R, L, D, U

    def step(self, action_idx):
        dr, dc = self.actions[action_idx]
        r = max(0, min(self.size - 1, self.state[0] + dr))
        c = max(0, min(self.size - 1, self.state[1] + dc))
        self.state = (r, c)
        done = self.state == self.goal
        reward = 0.0 if done else -1.0
        return self.state, reward, done

    def reset(self):
        self.state = (0, 0)
        return self.state

env = GridWorldMDP()
state = env.reset()
total_reward = 0
for _ in range(20):
    action = np.random.randint(4)  # random policy
    state, reward, done = env.step(action)
    total_reward += reward
    if done:
        break
print(f"Final state: {state}, Total reward: {total_reward}")`}),e.jsx(f,{type:"note",title:"Why Discount?",children:e.jsxs("p",{children:["The discount factor ",e.jsx(t.InlineMath,{math:"\\gamma"})," serves two purposes: it ensures the infinite sum converges and encodes a preference for sooner rewards. With ",e.jsx(t.InlineMath,{math:"\\gamma = 0"}),", the agent is myopic; with ",e.jsx(t.InlineMath,{math:"\\gamma \\to 1"}),", it plans far ahead."]})})]})}const ce=Object.freeze(Object.defineProperty({__proto__:null,default:z},Symbol.toStringTag,{value:"Module"}));function A(){const[a,c]=p.useState(.9),[i,l]=p.useState(1),[s,n]=p.useState(5),r=i+a*s;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Bellman Backup Calculator"}),e.jsxs("div",{className:"flex flex-wrap items-center gap-4 mb-3",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["gamma = ",a.toFixed(2),e.jsx("input",{type:"range",min:0,max:.99,step:.01,value:a,onChange:o=>c(parseFloat(o.target.value)),className:"w-24 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["r = ",i.toFixed(1),e.jsx("input",{type:"range",min:-5,max:5,step:.1,value:i,onChange:o=>l(parseFloat(o.target.value)),className:"w-24 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["V(s') = ",s.toFixed(1),e.jsx("input",{type:"range",min:0,max:10,step:.1,value:s,onChange:o=>n(parseFloat(o.target.value)),className:"w-24 accent-violet-500"})]})]}),e.jsxs("p",{className:"text-center text-violet-700 dark:text-violet-400 font-mono text-lg",children:["V(s) = ",i.toFixed(1)," + ",a.toFixed(2)," x ",s.toFixed(1)," = ",e.jsx("strong",{children:r.toFixed(2)})]})]})}function R(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"The Bellman equations express a recursive relationship between the value of a state and the values of its successor states. They are the backbone of almost every RL algorithm."}),e.jsxs(m,{title:"State-Value Function",children:[e.jsxs("p",{children:["The value of state ",e.jsx(t.InlineMath,{math:"s"})," under policy ",e.jsx(t.InlineMath,{math:"\\pi"}),":"]}),e.jsx(t.BlockMath,{math:"V^\\pi(s) = \\mathbb{E}_\\pi\\left[\\sum_{k=0}^{\\infty} \\gamma^k R_{t+k+1} \\mid S_t = s\\right]"})]}),e.jsxs(m,{title:"Action-Value Function",children:[e.jsxs("p",{children:["The value of taking action ",e.jsx(t.InlineMath,{math:"a"})," in state ",e.jsx(t.InlineMath,{math:"s"})," under policy ",e.jsx(t.InlineMath,{math:"\\pi"}),":"]}),e.jsx(t.BlockMath,{math:"Q^\\pi(s,a) = \\mathbb{E}_\\pi\\left[\\sum_{k=0}^{\\infty} \\gamma^k R_{t+k+1} \\mid S_t = s, A_t = a\\right]"})]}),e.jsxs(b,{title:"Bellman Expectation Equation",id:"bellman-expectation",children:[e.jsx(t.BlockMath,{math:"V^\\pi(s) = \\sum_a \\pi(a|s) \\sum_{s'} P(s'|s,a)\\left[R(s,a,s') + \\gamma V^\\pi(s')\\right]"}),e.jsx("p",{className:"mt-2",children:"This decomposes the value into an immediate reward plus the discounted future value."})]}),e.jsxs(b,{title:"Bellman Optimality Equation",id:"bellman-optimality",children:[e.jsx(t.BlockMath,{math:"V^*(s) = \\max_a \\sum_{s'} P(s'|s,a)\\left[R(s,a,s') + \\gamma V^*(s')\\right]"}),e.jsx("p",{className:"mt-2",children:"And for the optimal action-value function:"}),e.jsx(t.BlockMath,{math:"Q^*(s,a) = \\sum_{s'} P(s'|s,a)\\left[R(s,a,s') + \\gamma \\max_{a'} Q^*(s',a')\\right]"})]}),e.jsx(A,{}),e.jsxs(u,{title:"Two-State MDP",children:[e.jsx("p",{children:"State A: action 'stay' gives r=2 and stays in A. Action 'go' gives r=0 and moves to B (terminal)."}),e.jsx(t.BlockMath,{math:"V^*(A) = \\max\\{2 + \\gamma V^*(A),\\; 0\\} = \\frac{2}{1-\\gamma}"}),e.jsxs("p",{children:["With ",e.jsx(t.InlineMath,{math:"\\gamma=0.9"}),": ",e.jsx(t.InlineMath,{math:"V^*(A)=20"}),"."]})]}),e.jsx(j,{title:"Solving Bellman Equations with Linear Algebra",code:`import numpy as np

# 3-state MDP: transition matrix under a fixed policy
P = np.array([
    [0.7, 0.3, 0.0],  # state 0 transitions
    [0.0, 0.6, 0.4],  # state 1 transitions
    [0.0, 0.0, 1.0],  # state 2 (terminal)
])
R = np.array([1.0, 0.5, 0.0])  # expected rewards
gamma = 0.9

# Bellman equation: V = R + gamma * P @ V
# => (I - gamma * P) @ V = R
V = np.linalg.solve(np.eye(3) - gamma * P, R)
print("State values:", V.round(3))
# V[0] should be highest since it collects reward longest`}),e.jsx(f,{type:"note",title:"Bellman Equations in Practice",children:e.jsxs("p",{children:["Direct matrix solution has ",e.jsx(t.InlineMath,{math:"O(|\\mathcal{S}|^3)"})," complexity, making it infeasible for large state spaces. In practice, iterative methods (value iteration, TD learning) or function approximation (deep RL) are used instead."]})})]})}const de=Object.freeze(Object.defineProperty({__proto__:null,default:R},Symbol.toStringTag,{value:"Module"}));function C(){const[a,c]=p.useState(0),i=[[0,0,0,0],[-1,-1,-1,0],[-1.9,-1.9,-1,0],[-2.71,-1.9,-1,0],[-2.71,-2.71,-1,0]],l=i[Math.min(a,i.length-1)];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Value Iteration (1D Grid)"}),e.jsx("div",{className:"flex items-center gap-4 mb-3",children:e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Iteration: ",a,e.jsx("input",{type:"range",min:0,max:4,step:1,value:a,onChange:s=>c(parseInt(s.target.value)),className:"w-32 accent-violet-500"})]})}),e.jsx("div",{className:"flex justify-center gap-2",children:l.map((s,n)=>e.jsxs("div",{className:`w-20 h-16 rounded-lg flex flex-col items-center justify-center text-sm font-mono ${n===3?"bg-violet-100 dark:bg-violet-900/40 border-2 border-violet-500":"bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-600"}`,children:[e.jsxs("span",{className:"text-xs text-gray-500",children:["S",n]}),e.jsx("span",{className:"font-bold text-violet-700 dark:text-violet-400",children:s.toFixed(2)})]},n))}),e.jsx("p",{className:"text-center text-xs text-gray-500 mt-2",children:"S3 is the goal (V=0). Each step costs -1."})]})}function O(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"When the MDP model is fully known, dynamic programming methods can compute optimal policies exactly. Value iteration and policy iteration are the two classical algorithms."}),e.jsxs(m,{title:"Value Iteration",children:[e.jsx("p",{children:"Repeatedly apply the Bellman optimality operator until convergence:"}),e.jsx(t.BlockMath,{math:"V_{k+1}(s) = \\max_a \\sum_{s'} P(s'|s,a)\\left[R(s,a,s') + \\gamma V_k(s')\\right]"}),e.jsxs("p",{className:"mt-2",children:["Converges to ",e.jsx(t.InlineMath,{math:"V^*"})," as ",e.jsx(t.InlineMath,{math:"k \\to \\infty"})," due to the contraction mapping theorem."]})]}),e.jsxs(b,{title:"Contraction Mapping",id:"contraction",children:[e.jsxs("p",{children:["The Bellman optimality operator ",e.jsx(t.InlineMath,{math:"\\mathcal{T}"})," is a ",e.jsx(t.InlineMath,{math:"\\gamma"}),"-contraction in the sup-norm:"]}),e.jsx(t.BlockMath,{math:"\\|\\mathcal{T}V_1 - \\mathcal{T}V_2\\|_\\infty \\le \\gamma \\|V_1 - V_2\\|_\\infty"}),e.jsxs("p",{className:"mt-2",children:["By the Banach fixed-point theorem, iteration converges at rate ",e.jsx(t.InlineMath,{math:"O(\\gamma^k)"}),"."]})]}),e.jsx(C,{}),e.jsxs(m,{title:"Policy Iteration",children:[e.jsx("p",{children:"Alternates between two steps:"}),e.jsx(t.BlockMath,{math:"\\text{1. Evaluate: } V^{\\pi_k}(s) = \\sum_a \\pi_k(a|s)\\sum_{s'}P(s'|s,a)[R + \\gamma V^{\\pi_k}(s')]"}),e.jsx(t.BlockMath,{math:"\\text{2. Improve: } \\pi_{k+1}(s) = \\arg\\max_a \\sum_{s'}P(s'|s,a)[R + \\gamma V^{\\pi_k}(s')]"})]}),e.jsxs(u,{title:"Convergence Comparison",children:[e.jsxs("p",{children:["For a 100-state MDP with ",e.jsx(t.InlineMath,{math:"\\gamma=0.99"}),":"]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Value iteration"}),": ~500 iterations to converge (each iteration is cheap)."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Policy iteration"}),": ~10 iterations (each requires solving a linear system)."]}),e.jsx("p",{children:"Policy iteration often converges in fewer outer loops but each step is more expensive."})]}),e.jsx(j,{title:"Value Iteration Implementation",code:`import numpy as np

def value_iteration(P, R, gamma=0.99, theta=1e-8):
    """P: (S,A,S') transition probs, R: (S,A) rewards"""
    n_states, n_actions, _ = P.shape
    V = np.zeros(n_states)
    for i in range(10000):
        V_new = np.max(
            np.sum(P * (R[:, :, None] + gamma * V[None, None, :]), axis=2),
            axis=1
        )
        if np.max(np.abs(V_new - V)) < theta:
            print(f"Converged in {i+1} iterations")
            break
        V = V_new
    policy = np.argmax(
        np.sum(P * (R[:, :, None] + gamma * V[None, None, :]), axis=2),
        axis=1
    )
    return V, policy

# Small 3-state, 2-action MDP
P = np.array([
    [[0.7, 0.3, 0.0], [0.0, 0.5, 0.5]],
    [[0.0, 0.9, 0.1], [0.0, 0.0, 1.0]],
    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
])
R = np.array([[1.0, 0.5], [0.5, -1.0], [0.0, 0.0]])
V, pi = value_iteration(P, R)
print("Optimal values:", V.round(3))
print("Optimal policy:", pi)`}),e.jsx(f,{type:"note",title:"From DP to Deep RL",children:e.jsxs("p",{children:["Dynamic programming requires a complete model ",e.jsx(t.InlineMath,{math:"P(s'|s,a)"}),". In practice, this is rarely available. Modern RL methods replace exact computation with sampling (Monte Carlo, TD learning) and function approximation (neural networks), connecting these classical ideas to deep learning."]})})]})}const he=Object.freeze(Object.defineProperty({__proto__:null,default:O},Symbol.toStringTag,{value:"Module"}));function D(){const[a,c]=p.useState(.1),[i,l]=p.useState(.1),s=[[0,0],[0,0],[0,0],[0,0]],[n,r]=p.useState(s),[o,h]=p.useState(0),x=()=>{const d=Math.floor(Math.random()*3),y=Math.random()<i?Math.floor(Math.random()*2):n[d][0]>=n[d][1]?0:1,w=d===2&&y===0?1:-.1,_=Math.min(d+1,3),N=Math.max(n[_][0],n[_][1]),k=n.map(M=>[...M]);k[d][y]=k[d][y]+a*(w+.9*N-k[d][y]),r(k),h(o+1)};return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Q-Table Update Simulator"}),e.jsxs("div",{className:"flex flex-wrap items-center gap-4 mb-3",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["alpha = ",a.toFixed(2)," ",e.jsx("input",{type:"range",min:.01,max:1,step:.01,value:a,onChange:d=>c(parseFloat(d.target.value)),className:"w-20 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["epsilon = ",i.toFixed(2)," ",e.jsx("input",{type:"range",min:0,max:1,step:.01,value:i,onChange:d=>l(parseFloat(d.target.value)),className:"w-20 accent-violet-500"})]}),e.jsx("button",{onClick:x,className:"px-3 py-1 rounded bg-violet-500 text-white text-sm hover:bg-violet-600",children:"Step"}),e.jsx("button",{onClick:()=>{r(s.map(d=>[...d])),h(0)},className:"px-3 py-1 rounded bg-gray-300 text-gray-700 text-sm hover:bg-gray-400",children:"Reset"}),e.jsxs("span",{className:"text-sm text-gray-500",children:["Steps: ",o]})]}),e.jsxs("table",{className:"mx-auto text-sm border-collapse",children:[e.jsx("thead",{children:e.jsxs("tr",{children:[e.jsx("th",{className:"px-3 py-1 text-gray-600",children:"State"}),e.jsx("th",{className:"px-3 py-1 text-violet-600",children:"Q(s,a0)"}),e.jsx("th",{className:"px-3 py-1 text-violet-600",children:"Q(s,a1)"})]})}),e.jsx("tbody",{children:n.map((d,y)=>e.jsxs("tr",{children:[e.jsxs("td",{className:"px-3 py-1 text-center font-mono",children:["S",y]}),d.map((w,_)=>e.jsx("td",{className:"px-3 py-1 text-center font-mono text-violet-700 dark:text-violet-400",children:w.toFixed(3)},_))]},y))})]})]})}function F(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Q-learning is an off-policy temporal difference method that learns the optimal action-value function directly, without requiring a model of the environment."}),e.jsxs(m,{title:"Q-Learning Update Rule",children:[e.jsx(t.BlockMath,{math:"Q(s,a) \\leftarrow Q(s,a) + \\alpha \\left[r + \\gamma \\max_{a'} Q(s',a') - Q(s,a)\\right]"}),e.jsxs("p",{className:"mt-2",children:["The term ",e.jsx(t.InlineMath,{math:"r + \\gamma \\max_{a'} Q(s',a') - Q(s,a)"})," is the ",e.jsx("strong",{children:"TD error"}),". The key insight: we use ",e.jsx(t.InlineMath,{math:"\\max"})," over next actions regardless of what action was actually taken (off-policy)."]})]}),e.jsx(m,{title:"Epsilon-Greedy Exploration",children:e.jsx(t.BlockMath,{math:"a = \\begin{cases} \\arg\\max_a Q(s,a) & \\text{with probability } 1-\\varepsilon \\\\ \\text{random action} & \\text{with probability } \\varepsilon \\end{cases}"})}),e.jsx(D,{}),e.jsx(u,{title:"Cliff Walking",children:e.jsx("p",{children:"In the cliff walking problem, Q-learning learns the optimal path along the cliff edge (shortest but risky), while SARSA learns a safer path further from the cliff because it accounts for its own exploratory behavior."})}),e.jsx(j,{title:"Tabular Q-Learning",code:`import numpy as np

def q_learning(env, n_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = np.zeros((env.n_states, env.n_actions))
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                a = np.random.randint(env.n_actions)
            else:
                a = np.argmax(Q[s])
            s_next, reward, done = env.step(a)
            # Q-learning update (off-policy: uses max over next actions)
            td_error = reward + gamma * np.max(Q[s_next]) - Q[s]
            Q[s, a] += alpha * td_error
            s = s_next
    return Q

# The key difference from SARSA:
# SARSA: Q[s,a] += alpha * (r + gamma * Q[s',a'] - Q[s,a])   (on-policy)
# Q-learning: Q[s,a] += alpha * (r + gamma * max Q[s',:] - Q[s,a])  (off-policy)`}),e.jsx(P,{title:"Maximization Bias",children:e.jsxs("p",{children:["Q-learning's ",e.jsx(t.InlineMath,{math:"\\max"})," operator introduces an upward bias in value estimates because ",e.jsx(t.InlineMath,{math:"\\mathbb{E}[\\max Q] \\ge \\max \\mathbb{E}[Q]"}),". This is the motivation for Double Q-learning and later Double DQN."]})}),e.jsx(f,{type:"note",title:"Tabular to Deep",children:e.jsxs("p",{children:["Tabular Q-learning stores one value per state-action pair, limiting it to small discrete spaces. Deep Q-Networks replace the table with a neural network ",e.jsx(t.InlineMath,{math:"Q_\\theta(s,a)"}),", enabling Q-learning on high-dimensional inputs like images."]})})]})}const me=Object.freeze(Object.defineProperty({__proto__:null,default:F},Symbol.toStringTag,{value:"Module"}));function B(){const[a,c]=p.useState(5),[i,l]=p.useState(3),s=Array.from({length:a},(r,o)=>({id:o,s:`s${o}`,a:`a${o%2}`,r:(Math.random()*2-1).toFixed(1),sn:`s${o+1}`})),n=new Set;for(;n.size<Math.min(i,a);)n.add(Math.floor(Math.random()*a));return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Experience Replay Buffer"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Buffer: ",a," ",e.jsx("input",{type:"range",min:3,max:8,step:1,value:a,onChange:r=>c(parseInt(r.target.value)),className:"w-20 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Batch: ",i," ",e.jsx("input",{type:"range",min:1,max:a,step:1,value:Math.min(i,a),onChange:r=>l(parseInt(r.target.value)),className:"w-20 accent-violet-500"})]})]}),e.jsx("div",{className:"flex flex-wrap gap-2 justify-center",children:s.map((r,o)=>e.jsxs("div",{className:`px-3 py-2 rounded-lg text-xs font-mono ${n.has(o)?"bg-violet-100 dark:bg-violet-900/40 border-2 border-violet-500":"bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-600"}`,children:["(",r.s,", ",r.a,", ",r.r,", ",r.sn,")"]},o))}),e.jsx("p",{className:"text-center text-xs text-gray-500 mt-2",children:"Violet = sampled for training batch"})]})}function Q(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Deep Q-Networks (DQN) combined neural network function approximation with Q-learning, achieving human-level play on Atari games. Two key innovations made this stable: experience replay and target networks."}),e.jsxs(m,{title:"DQN Loss Function",children:[e.jsxs("p",{children:["The network ",e.jsx(t.InlineMath,{math:"Q_\\theta(s,a)"})," is trained to minimize:"]}),e.jsx(t.BlockMath,{math:"\\mathcal{L}(\\theta) = \\mathbb{E}_{(s,a,r,s') \\sim \\mathcal{D}}\\left[\\left(r + \\gamma \\max_{a'} Q_{\\theta^-}(s',a') - Q_\\theta(s,a)\\right)^2\\right]"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"\\theta^-"})," are the ",e.jsx("strong",{children:"target network"})," parameters, updated periodically: ",e.jsx(t.InlineMath,{math:"\\theta^- \\leftarrow \\theta"})," every ",e.jsx(t.InlineMath,{math:"C"})," steps."]})]}),e.jsx(m,{title:"Experience Replay",children:e.jsxs("p",{children:["Store transitions ",e.jsx(t.InlineMath,{math:"(s, a, r, s')"})," in a buffer ",e.jsx(t.InlineMath,{math:"\\mathcal{D}"})," and sample random mini-batches for training. This breaks temporal correlations and reuses data efficiently."]})}),e.jsx(B,{}),e.jsx(b,{title:"Why Target Networks Stabilize Training",id:"target-stability",children:e.jsxs("p",{children:["Without a target network, the TD target ",e.jsx(t.InlineMath,{math:"r + \\gamma \\max Q_\\theta(s', a')"})," changes with every gradient step, creating a moving target. The target network provides a stable objective for ",e.jsx(t.InlineMath,{math:"C"})," steps, reducing oscillations and divergence."]})}),e.jsx(u,{title:"DQN on Atari",children:e.jsx("p",{children:"The original DQN (Mnih et al., 2015) processed 84x84 grayscale frames through 3 conv layers and 2 FC layers. With a replay buffer of 1M transitions and target update every 10K steps, it surpassed human performance on 29 of 49 Atari games."})}),e.jsx(j,{title:"DQN in PyTorch",code:`import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(torch.stack, zip(*[(torch.tensor(s), torch.tensor([a]),
            torch.tensor([r]), torch.tensor(sn), torch.tensor([d]))
            for s, a, r, sn, d in batch]))

def train_step(q_net, target_net, buffer, optimizer, gamma=0.99, batch=32):
    s, a, r, s_next, done = buffer.sample(batch)
    q_values = q_net(s).gather(1, a.long())
    with torch.no_grad():
        max_next_q = target_net(s_next).max(1, keepdim=True)[0]
        target = r + gamma * max_next_q * (1 - done)
    loss = nn.functional.mse_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()`}),e.jsx(f,{type:"note",title:"Soft Target Updates",children:e.jsxs("p",{children:["Instead of hard copies every ",e.jsx(t.InlineMath,{math:"C"})," steps, many implementations use Polyak averaging: ",e.jsx(t.InlineMath,{math:"\\theta^- \\leftarrow \\tau \\theta + (1-\\tau)\\theta^-"})," with",e.jsx(t.InlineMath,{math:"\\tau \\approx 0.005"}),", providing smoother target network updates."]})})]})}const pe=Object.freeze(Object.defineProperty({__proto__:null,default:Q},Symbol.toStringTag,{value:"Module"}));function V(){const[a,c]=p.useState(5),i=[-1.2,0,.8,-.3],l=["Left","Stay","Right","Jump"],s=i.reduce((r,o)=>r+o,0)/i.length,n=i.map(r=>a+(r-s));return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Dueling DQN Decomposition"}),e.jsx("div",{className:"flex items-center gap-4 mb-3",children:e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["V(s) = ",a.toFixed(1),e.jsx("input",{type:"range",min:0,max:10,step:.1,value:a,onChange:r=>c(parseFloat(r.target.value)),className:"w-28 accent-violet-500"})]})}),e.jsx("div",{className:"flex flex-wrap gap-2 justify-center",children:l.map((r,o)=>e.jsxs("div",{className:"px-3 py-2 rounded-lg bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 text-center",children:[e.jsx("div",{className:"text-xs text-gray-500",children:r}),e.jsxs("div",{className:"text-xs text-violet-500",children:["A=",i[o].toFixed(1)]}),e.jsxs("div",{className:"font-bold text-violet-700 dark:text-violet-400 text-sm",children:["Q=",n[o].toFixed(2)]})]},o))}),e.jsx("p",{className:"text-center text-xs text-gray-500 mt-2",children:"Q(s,a) = V(s) + A(s,a) - mean(A)"})]})}function L(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Several improvements to the basic DQN address its limitations: overestimation bias, inefficient replay sampling, and entangled value-advantage estimation. Rainbow combines six of these improvements into a single agent."}),e.jsxs(m,{title:"Double DQN",children:[e.jsx("p",{children:"Decouples action selection from evaluation to reduce overestimation:"}),e.jsx(t.BlockMath,{math:"y = r + \\gamma Q_{\\theta^-}\\!\\left(s',\\; \\arg\\max_{a'} Q_\\theta(s', a')\\right)"}),e.jsxs("p",{className:"mt-2",children:["The online network ",e.jsx(t.InlineMath,{math:"\\theta"})," selects the action; the target network ",e.jsx(t.InlineMath,{math:"\\theta^-"})," evaluates it. This simple change significantly reduces the maximization bias of standard DQN."]})]}),e.jsxs(m,{title:"Prioritized Experience Replay",children:[e.jsx("p",{children:"Sample transitions proportional to their TD error magnitude:"}),e.jsx(t.BlockMath,{math:"P(i) = \\frac{p_i^\\alpha}{\\sum_k p_k^\\alpha}, \\quad p_i = |\\delta_i| + \\varepsilon"}),e.jsxs("p",{className:"mt-2",children:["Importance sampling weights correct the bias: ",e.jsx(t.InlineMath,{math:"w_i = (N \\cdot P(i))^{-\\beta}"})]})]}),e.jsxs(m,{title:"Dueling Architecture",children:[e.jsx("p",{children:"Decomposes Q into state value and advantage streams:"}),e.jsx(t.BlockMath,{math:"Q(s,a;\\theta) = V(s;\\theta_v) + A(s,a;\\theta_a) - \\frac{1}{|\\mathcal{A}|}\\sum_{a'} A(s,a';\\theta_a)"})]}),e.jsx(V,{}),e.jsxs(u,{title:"Rainbow DQN Components",children:[e.jsx("p",{children:"Rainbow (Hessel et al., 2018) combines six improvements:"}),e.jsxs("ol",{className:"list-decimal ml-5 mt-2 space-y-1",children:[e.jsx("li",{children:"Double DQN (reduce overestimation)"}),e.jsx("li",{children:"Prioritized replay (focus on surprising transitions)"}),e.jsx("li",{children:"Dueling networks (separate value and advantage)"}),e.jsx("li",{children:"Multi-step returns (faster credit assignment)"}),e.jsx("li",{children:"Distributional RL (model return distribution)"}),e.jsx("li",{children:"Noisy networks (parameter-space exploration)"})]})]}),e.jsx(j,{title:"Double DQN Update",code:`import torch
import torch.nn as nn

def double_dqn_loss(q_net, target_net, batch, gamma=0.99):
    states, actions, rewards, next_states, dones = batch

    # Current Q-values
    q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Double DQN: select action with online net, evaluate with target
        best_actions = q_net(next_states).argmax(dim=1, keepdim=True)
        next_q = target_net(next_states).gather(1, best_actions).squeeze(1)
        targets = rewards + gamma * next_q * (1 - dones.float())

    loss = nn.functional.smooth_l1_loss(q_values, targets)  # Huber loss
    return loss

# Comparison: standard DQN uses target_net for BOTH selection and evaluation
# next_q = target_net(next_states).max(dim=1)[0]  # overestimates!`}),e.jsx(P,{title:"Hyperparameter Sensitivity",children:e.jsxs("p",{children:["DQN variants are notoriously sensitive to hyperparameters. Buffer size, target update frequency, learning rate, and epsilon schedule all interact. Prioritized replay adds",e.jsx(t.InlineMath,{math:"\\alpha"})," and ",e.jsx(t.InlineMath,{math:"\\beta"})," annealing. Always start with published defaults before tuning."]})}),e.jsx(f,{type:"note",title:"Beyond Discrete Actions",children:e.jsxs("p",{children:["DQN methods require a discrete action space to compute ",e.jsx(t.InlineMath,{math:"\\max_a Q(s,a)"}),". For continuous actions, policy gradient and actor-critic methods (covered next) are needed."]})})]})}const xe=Object.freeze(Object.defineProperty({__proto__:null,default:L},Symbol.toStringTag,{value:"Module"}));function q(){const[a,c]=p.useState(10),[i,l]=p.useState(0),s=Array.from({length:a},(o,h)=>Math.sin((h+i)*7.3)*5+Math.cos((h+i)*3.1)*3),n=s.reduce((o,h)=>o+h,0)/a,r=s.reduce((o,h)=>o+(h-n)**2,0)/a;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Gradient Estimate Variance"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Trajectories: ",a,e.jsx("input",{type:"range",min:1,max:50,step:1,value:a,onChange:o=>c(parseInt(o.target.value)),className:"w-28 accent-violet-500"})]}),e.jsx("button",{onClick:()=>l(i+1),className:"px-3 py-1 rounded bg-violet-500 text-white text-sm hover:bg-violet-600",children:"Resample"}),e.jsxs("span",{className:"text-sm text-violet-600 dark:text-violet-400",children:["Var: ",r.toFixed(2)]})]}),e.jsxs("svg",{width:360,height:80,className:"mx-auto block",children:[e.jsx("line",{x1:20,y1:40,x2:340,y2:40,stroke:"#d1d5db",strokeWidth:1}),s.map((o,h)=>e.jsx("circle",{cx:180+o*12,cy:40,r:3,fill:"#7c3aed",opacity:.5},h)),e.jsx("line",{x1:180+n*12,y1:20,x2:180+n*12,y2:60,stroke:"#f97316",strokeWidth:2}),e.jsx("text",{x:180+n*12,y:15,textAnchor:"middle",fill:"#f97316",fontSize:10,children:"mean"})]})]})}function E(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"REINFORCE is the simplest policy gradient algorithm: it directly differentiates the expected return with respect to policy parameters using the log-derivative trick."}),e.jsxs(b,{title:"Policy Gradient Theorem",id:"policy-gradient-theorem",children:[e.jsx(t.BlockMath,{math:"\\nabla_\\theta J(\\theta) = \\mathbb{E}_{\\pi_\\theta}\\left[\\sum_{t=0}^T \\nabla_\\theta \\log \\pi_\\theta(a_t|s_t) \\cdot G_t\\right]"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"G_t = \\sum_{k=t}^T \\gamma^{k-t} r_k"})," is the return from time step ",e.jsx(t.InlineMath,{math:"t"}),"."]})]}),e.jsxs(m,{title:"REINFORCE Algorithm",children:[e.jsx("p",{children:"For each episode:"}),e.jsx(t.BlockMath,{math:"\\theta \\leftarrow \\theta + \\alpha \\sum_{t=0}^T \\nabla_\\theta \\log \\pi_\\theta(a_t|s_t) \\cdot G_t"}),e.jsxs("p",{className:"mt-2",children:["This is a Monte Carlo method: it requires complete episodes and uses the actual return ",e.jsx(t.InlineMath,{math:"G_t"})," rather than a bootstrapped estimate."]})]}),e.jsx(q,{}),e.jsxs(u,{title:"Intuition Behind the Gradient",children:[e.jsxs("p",{children:["The gradient ",e.jsx(t.InlineMath,{math:"\\nabla \\log \\pi(a|s) \\cdot G"})," does two things:"]}),e.jsxs("p",{children:["If ",e.jsx(t.InlineMath,{math:"G > 0"}),": increase the probability of action ",e.jsx(t.InlineMath,{math:"a"})," (it led to good returns)."]}),e.jsxs("p",{children:["If ",e.jsx(t.InlineMath,{math:"G < 0"}),": decrease the probability (it led to bad returns)."]}),e.jsx("p",{children:"The magnitude of the update scales with how good or bad the outcome was."})]}),e.jsx(j,{title:"REINFORCE in PyTorch",code:`import torch
import torch.nn as nn
from torch.distributions import Categorical

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim), nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

def reinforce_episode(policy, optimizer, env, gamma=0.99):
    log_probs, rewards = [], []
    state = env.reset()
    done = False
    while not done:
        probs = policy(torch.FloatTensor(state))
        dist = Categorical(probs)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        state, reward, done, _ = env.step(action.item())
        rewards.append(reward)

    # Compute discounted returns
    G, returns = 0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Policy gradient update
    loss = sum(-lp * G for lp, G in zip(log_probs, returns))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()`}),e.jsx(f,{type:"note",title:"High Variance Problem",children:e.jsxs("p",{children:["REINFORCE suffers from high variance because it uses full Monte Carlo returns. Subtracting a baseline ",e.jsx(t.InlineMath,{math:"b(s)"})," from ",e.jsx(t.InlineMath,{math:"G_t"})," reduces variance without introducing bias. The natural choice ",e.jsx(t.InlineMath,{math:"b(s) = V(s)"})," leads to the advantage function, covered in the next section."]})})]})}const ge=Object.freeze(Object.defineProperty({__proto__:null,default:E},Symbol.toStringTag,{value:"Module"}));function G(){const[a,c]=p.useState(3),i=[1.5,3,4.5,2],l=["Left","Stay","Right","Jump"],s=i.map(n=>n-a);return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Advantage = Q - V"}),e.jsx("div",{className:"flex items-center gap-4 mb-3",children:e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["V(s) = ",a.toFixed(1),e.jsx("input",{type:"range",min:0,max:6,step:.1,value:a,onChange:n=>c(parseFloat(n.target.value)),className:"w-28 accent-violet-500"})]})}),e.jsx("div",{className:"flex flex-wrap gap-2 justify-center",children:l.map((n,r)=>e.jsxs("div",{className:`px-4 py-2 rounded-lg text-center ${s[r]>=0?"bg-violet-100 dark:bg-violet-900/40 border border-violet-400":"bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-600"}`,children:[e.jsx("div",{className:"text-xs text-gray-500",children:n}),e.jsxs("div",{className:"text-xs",children:["Q=",i[r].toFixed(1)]}),e.jsxs("div",{className:`font-bold text-sm ${s[r]>=0?"text-violet-700 dark:text-violet-400":"text-gray-500"}`,children:["A=",s[r].toFixed(1)]})]},r))}),e.jsx("p",{className:"text-center text-xs text-gray-500 mt-2",children:"Positive advantage = better than average for this state"})]})}function W(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Subtracting a baseline from the return reduces gradient variance without changing the expected gradient. The advantage function is the canonical baseline choice."}),e.jsxs(m,{title:"Advantage Function",children:[e.jsx(t.BlockMath,{math:"A^\\pi(s,a) = Q^\\pi(s,a) - V^\\pi(s)"}),e.jsxs("p",{className:"mt-2",children:["The advantage measures how much better action ",e.jsx(t.InlineMath,{math:"a"})," is compared to the average action under ",e.jsx(t.InlineMath,{math:"\\pi"}),". By construction, ",e.jsx(t.InlineMath,{math:"\\mathbb{E}_{a \\sim \\pi}[A(s,a)] = 0"}),"."]})]}),e.jsxs(b,{title:"Baseline Does Not Change Expected Gradient",id:"baseline-unbiased",children:[e.jsxs("p",{children:["For any baseline ",e.jsx(t.InlineMath,{math:"b(s)"})," that depends only on the state:"]}),e.jsx(t.BlockMath,{math:"\\mathbb{E}_{\\pi_\\theta}\\left[\\nabla_\\theta \\log \\pi_\\theta(a|s) \\cdot b(s)\\right] = 0"})]}),e.jsx(T,{title:"Proof: Baseline is Zero in Expectation",children:e.jsx(t.BlockMath,{math:"\\mathbb{E}_{a \\sim \\pi}\\left[\\nabla \\log \\pi(a|s) \\cdot b(s)\\right] = b(s) \\sum_a \\nabla \\pi(a|s) = b(s) \\nabla \\underbrace{\\sum_a \\pi(a|s)}_{=1} = 0"})}),e.jsx(G,{}),e.jsxs(m,{title:"Generalized Advantage Estimation (GAE)",children:[e.jsx("p",{children:"GAE interpolates between high-bias (1-step TD) and high-variance (MC) estimates:"}),e.jsx(t.BlockMath,{math:"\\hat{A}_t^{\\text{GAE}} = \\sum_{l=0}^{\\infty} (\\gamma \\lambda)^l \\delta_{t+l}"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)"})," is the TD residual.",e.jsx(t.InlineMath,{math:"\\lambda=0"})," gives 1-step TD, ",e.jsx(t.InlineMath,{math:"\\lambda=1"})," gives MC returns."]})]}),e.jsx(u,{title:"Choosing Lambda",children:e.jsxs("p",{children:["In practice, ",e.jsx(t.InlineMath,{math:"\\lambda = 0.95"})," works well for most tasks. It provides a good bias-variance tradeoff: mostly low-variance TD estimates with some Monte Carlo contribution for faster credit assignment over longer horizons."]})}),e.jsx(j,{title:"GAE Implementation",code:`import torch

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < len(values) else 0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values[:T]
    return advantages, returns

# Example usage
rewards = torch.tensor([1.0, 0.0, 0.0, 10.0])
values = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0])
dones = torch.tensor([0, 0, 0, 1])
adv, ret = compute_gae(rewards, values, dones)
print(f"Advantages: {adv}")
print(f"Returns: {ret}")`}),e.jsx(f,{type:"note",title:"From Baselines to Actor-Critic",children:e.jsxs("p",{children:["When we learn both a policy ",e.jsx(t.InlineMath,{math:"\\pi_\\theta"})," (actor) and a value function ",e.jsx(t.InlineMath,{math:"V_\\phi"})," (critic), we get an actor-critic algorithm. The critic provides the baseline, and both are trained simultaneously."]})})]})}const ue=Object.freeze(Object.defineProperty({__proto__:null,default:W},Symbol.toStringTag,{value:"Module"}));function H(){const[a,c]=p.useState(.5),i=300,l=200,s=150,n=100,r=80;return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Trust Region Constraint"}),e.jsx("div",{className:"flex items-center gap-4 mb-3",children:e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["delta = ",a.toFixed(2),e.jsx("input",{type:"range",min:.05,max:1.5,step:.05,value:a,onChange:o=>c(parseFloat(o.target.value)),className:"w-28 accent-violet-500"})]})}),e.jsxs("svg",{width:i,height:l,className:"mx-auto block",children:[e.jsx("circle",{cx:s,cy:n,r:a*r,fill:"none",stroke:"#7c3aed",strokeWidth:2,strokeDasharray:"5,3",opacity:.7}),e.jsx("circle",{cx:s,cy:n,r:a*r,fill:"#7c3aed",opacity:.08}),e.jsx("circle",{cx:s,cy:n,r:5,fill:"#7c3aed"}),e.jsx("text",{x:s+8,y:n-8,fill:"#7c3aed",fontSize:11,children:"theta_old"}),a>.3&&e.jsx("circle",{cx:s+a*r*.5,cy:n-a*r*.4,r:4,fill:"#f97316"}),a>.3&&e.jsx("text",{x:s+a*r*.5+8,y:n-a*r*.4,fill:"#f97316",fontSize:11,children:"theta_new"}),e.jsx("text",{x:s+a*r+4,y:n+4,fill:"#7c3aed",fontSize:10,opacity:.7,children:"KL ≤ delta"})]}),e.jsx("p",{className:"text-center text-xs text-gray-500 mt-1",children:"Larger delta allows bigger policy updates but risks instability"})]})}function K(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Trust Region Policy Optimization (TRPO) constrains each policy update to stay close to the current policy, guaranteeing monotonic improvement under certain conditions."}),e.jsxs(b,{title:"Surrogate Objective",id:"surrogate-objective",children:[e.jsx("p",{children:"TRPO maximizes a lower bound on the true policy improvement:"}),e.jsx(t.BlockMath,{math:"\\max_\\theta \\; \\mathbb{E}_{s,a \\sim \\pi_{\\theta_\\text{old}}}\\left[\\frac{\\pi_\\theta(a|s)}{\\pi_{\\theta_\\text{old}}(a|s)} \\hat{A}(s,a)\\right]"}),e.jsx(t.BlockMath,{math:"\\text{subject to} \\quad \\mathbb{E}_s\\left[D_\\text{KL}\\!\\left(\\pi_{\\theta_\\text{old}}(\\cdot|s) \\| \\pi_\\theta(\\cdot|s)\\right)\\right] \\le \\delta"})]}),e.jsxs(m,{title:"Natural Policy Gradient",children:[e.jsx("p",{children:"The natural gradient preconditions with the Fisher information matrix:"}),e.jsx(t.BlockMath,{math:"\\theta \\leftarrow \\theta + \\alpha F^{-1} \\nabla_\\theta J(\\theta)"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"F = \\mathbb{E}[\\nabla \\log \\pi \\cdot \\nabla \\log \\pi^\\top]"}),". TRPO approximates this using conjugate gradients to avoid computing ",e.jsx(t.InlineMath,{math:"F^{-1}"})," directly."]})]}),e.jsx(H,{}),e.jsxs(u,{title:"TRPO vs Vanilla Policy Gradient",children:[e.jsx("p",{children:"On the Humanoid-v2 MuJoCo task (376-dim state, 17-dim action):"}),e.jsxs("p",{children:[e.jsx("strong",{children:"Vanilla PG"}),": Training collapses after ~200 episodes due to large destructive updates."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"TRPO"}),": Monotonically improves, reaching 2000+ reward. The KL constraint prevents catastrophic policy changes."]})]}),e.jsx(j,{title:"TRPO Core: Conjugate Gradient Step",code:`import torch

def conjugate_gradient(Fvp, b, n_steps=10, residual_tol=1e-10):
    """Solve Fx = b using conjugate gradient, where Fvp computes F@v."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = r.dot(r)
    for _ in range(n_steps):
        Fp = Fvp(p)
        alpha = rdotr / (p.dot(Fp) + 1e-8)
        x += alpha * p
        r -= alpha * Fp
        new_rdotr = r.dot(r)
        if new_rdotr < residual_tol:
            break
        p = r + (new_rdotr / rdotr) * p
        rdotr = new_rdotr
    return x

def trpo_step(policy, get_loss, get_kl, max_kl=0.01):
    """One TRPO update step."""
    loss = get_loss()
    grads = torch.autograd.grad(loss, policy.parameters())
    flat_grad = torch.cat([g.view(-1) for g in grads])

    def Fvp(v):  # Fisher-vector product
        kl = get_kl()
        kl_grad = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
        flat_kl_grad = torch.cat([g.view(-1) for g in kl_grad])
        return torch.autograd.grad(flat_kl_grad.dot(v), policy.parameters())

    step_dir = conjugate_gradient(Fvp, flat_grad)
    shs = 0.5 * step_dir.dot(Fvp(step_dir))
    step_size = torch.sqrt(2 * max_kl / (shs + 1e-8))
    return step_size * step_dir`}),e.jsx(f,{type:"note",title:"TRPO to PPO",children:e.jsx("p",{children:"TRPO's constrained optimization with conjugate gradients is complex to implement and computationally expensive. PPO replaces the hard KL constraint with a simpler clipped surrogate objective, achieving similar or better performance with much simpler code."})})]})}const je=Object.freeze(Object.defineProperty({__proto__:null,default:K},Symbol.toStringTag,{value:"Module"}));function $(){const[a,c]=p.useState(4),i=["#7c3aed","#f97316","#10b981","#f43f5e","#3b82f6","#eab308","#8b5cf6","#ec4899"];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"A3C Parallel Workers"}),e.jsx("div",{className:"flex items-center gap-4 mb-3",children:e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["Workers: ",a,e.jsx("input",{type:"range",min:1,max:8,step:1,value:a,onChange:l=>c(parseInt(l.target.value)),className:"w-28 accent-violet-500"})]})}),e.jsxs("svg",{width:360,height:120,className:"mx-auto block",children:[e.jsx("rect",{x:140,y:10,width:80,height:30,rx:6,fill:"#7c3aed",opacity:.15,stroke:"#7c3aed",strokeWidth:1.5}),e.jsx("text",{x:180,y:30,textAnchor:"middle",fill:"#7c3aed",fontSize:11,fontWeight:"bold",children:"Shared Model"}),Array.from({length:a},(l,s)=>{const n=20+s*320/Math.max(a-1,1);return e.jsxs("g",{children:[e.jsx("line",{x1:180,y1:42,x2:n+20,y2:65,stroke:i[s%i.length],strokeWidth:1,opacity:.5}),e.jsx("rect",{x:n,y:68,width:40,height:40,rx:5,fill:i[s%i.length],opacity:.2,stroke:i[s%i.length],strokeWidth:1.5}),e.jsxs("text",{x:n+20,y:85,textAnchor:"middle",fill:i[s%i.length],fontSize:9,children:["Env ",s]}),e.jsx("text",{x:n+20,y:100,textAnchor:"middle",fill:i[s%i.length],fontSize:8,children:"Worker"})]},s)})]}),e.jsx("p",{className:"text-center text-xs text-gray-500 mt-1",children:"Each worker collects experience and sends gradients to the shared model"})]})}function U(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Actor-critic methods combine policy gradients (actor) with learned value functions (critic). A2C synchronizes parallel workers; A3C uses asynchronous gradient updates."}),e.jsxs(m,{title:"Actor-Critic Architecture",children:[e.jsx("p",{children:"Two networks (often sharing a backbone):"}),e.jsx(t.BlockMath,{math:"\\text{Actor: } \\pi_\\theta(a|s) \\qquad \\text{Critic: } V_\\phi(s)"}),e.jsx("p",{className:"mt-2",children:"The actor is updated with the policy gradient using advantage from the critic:"}),e.jsx(t.BlockMath,{math:"\\nabla_\\theta J = \\mathbb{E}\\left[\\nabla_\\theta \\log \\pi_\\theta(a|s) \\cdot \\hat{A}(s,a)\\right]"}),e.jsxs("p",{children:["The critic minimizes ",e.jsx(t.InlineMath,{math:"\\|V_\\phi(s) - G_t\\|^2"})," or uses TD targets."]})]}),e.jsxs(b,{title:"A3C: Asynchronous Advantage Actor-Critic",id:"a3c",children:[e.jsx("p",{children:"Key idea: run multiple workers in parallel, each with its own environment copy. Each worker:"}),e.jsx(t.BlockMath,{math:"\\text{1. Copy global params} \\to \\text{2. Collect n-step data} \\to \\text{3. Compute gradients} \\to \\text{4. Update global params}"}),e.jsx("p",{className:"mt-2",children:"Asynchronous updates provide implicit exploration through parameter diversity across workers."})]}),e.jsx($,{}),e.jsxs(u,{title:"A2C vs A3C",children:[e.jsxs("p",{children:[e.jsx("strong",{children:"A3C"}),": Workers update asynchronously (stale gradients possible). Simpler to scale across CPUs."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"A2C"}),": Workers synchronize before each update (no stale gradients). Often preferred because it is easier to implement with GPUs and gives equivalent or better results."]})]}),e.jsx(j,{title:"A2C Implementation",code:`import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)

def a2c_update(model, optimizer, states, actions, returns, gamma=0.99):
    logits, values = model(states)
    values = values.squeeze(-1)
    dist = Categorical(logits=logits)
    log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    advantages = returns - values.detach()
    actor_loss = -(log_probs * advantages).mean()
    critic_loss = nn.functional.mse_loss(values, returns)
    loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy  # entropy bonus

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    return loss.item()`}),e.jsx(f,{type:"note",title:"Entropy Bonus",children:e.jsxs("p",{children:["Adding an entropy term ",e.jsx(t.InlineMath,{math:"-\\beta H(\\pi)"})," to the loss encourages exploration by preventing the policy from becoming too deterministic too early. Typical values:",e.jsx(t.InlineMath,{math:"\\beta = 0.01"})," for discrete actions."]})})]})}const fe=Object.freeze(Object.defineProperty({__proto__:null,default:U},Symbol.toStringTag,{value:"Module"}));function J(){const[a,c]=p.useState(.2),[i,l]=p.useState(1),s=360,n=180,r=60,o=n-30,h=120,x=50,d=Array.from({length:101},(g,v)=>.2+v*.026),y=g=>r+(g-1)*h,w=g=>o-g*x,_=d.map(g=>g*i),N=d.map(g=>Math.min(Math.max(g,1-a),1+a)*i),k=d.map((g,v)=>i>=0?Math.min(_[v],N[v]):Math.max(_[v],N[v])),M=g=>d.map((v,S)=>`${S===0?"M":"L"}${y(v)},${w(g[S])}`).join(" ");return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"PPO Clipped Objective"}),e.jsxs("div",{className:"flex flex-wrap items-center gap-4 mb-3",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["epsilon = ",a.toFixed(2),e.jsx("input",{type:"range",min:.05,max:.5,step:.01,value:a,onChange:g=>c(parseFloat(g.target.value)),className:"w-24 accent-violet-500"})]}),e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["A = ",i.toFixed(1),e.jsx("input",{type:"range",min:-2,max:2,step:.1,value:i,onChange:g=>l(parseFloat(g.target.value)),className:"w-24 accent-violet-500"})]})]}),e.jsxs("svg",{width:s,height:n,className:"mx-auto block",children:[e.jsx("line",{x1:r,y1:10,x2:r,y2:o+10,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("line",{x1:r-10,y1:o,x2:s-10,y2:o,stroke:"#d1d5db",strokeWidth:.5}),e.jsx("path",{d:M(_),fill:"none",stroke:"#d1d5db",strokeWidth:1.5,strokeDasharray:"4,3"}),e.jsx("path",{d:M(k),fill:"none",stroke:"#7c3aed",strokeWidth:2.5}),e.jsx("line",{x1:y(1-a),y1:10,x2:y(1-a),y2:o,stroke:"#f97316",strokeWidth:1,strokeDasharray:"3,3"}),e.jsx("line",{x1:y(1+a),y1:10,x2:y(1+a),y2:o,stroke:"#f97316",strokeWidth:1,strokeDasharray:"3,3"}),e.jsx("text",{x:y(1),y:o+18,textAnchor:"middle",fill:"#374151",fontSize:10,children:"r=1"}),e.jsx("text",{x:s-30,y:o+18,fill:"#374151",fontSize:10,children:"r(theta)"})]}),e.jsxs("div",{className:"flex justify-center gap-4 text-xs mt-1",children:[e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-violet-500"})," PPO objective"]}),e.jsxs("span",{className:"flex items-center gap-1",children:[e.jsx("span",{className:"inline-block w-3 h-0.5 bg-gray-400",style:{borderTop:"1px dashed"}})," Unclipped"]})]})]})}function Z(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Proximal Policy Optimization (PPO) is the most widely used RL algorithm in practice, powering everything from game AI to RLHF for language models. It achieves TRPO-like stability with a much simpler clipped surrogate objective."}),e.jsxs(m,{title:"PPO-Clip Objective",children:[e.jsx(t.BlockMath,{math:"L^{\\text{CLIP}}(\\theta) = \\mathbb{E}_t\\left[\\min\\!\\left(r_t(\\theta)\\hat{A}_t,\\; \\text{clip}(r_t(\\theta), 1-\\varepsilon, 1+\\varepsilon)\\hat{A}_t\\right)\\right]"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"r_t(\\theta) = \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_\\text{old}}(a_t|s_t)}"})," is the probability ratio. Typical ",e.jsx(t.InlineMath,{math:"\\varepsilon = 0.2"}),"."]})]}),e.jsx(J,{}),e.jsxs(b,{title:"How Clipping Works",id:"clipping-mechanism",children:[e.jsxs("p",{children:["When ",e.jsx(t.InlineMath,{math:"\\hat{A} > 0"})," (good action): the objective is capped at ",e.jsx(t.InlineMath,{math:"(1+\\varepsilon)\\hat{A}"}),", preventing the ratio from growing too large."]}),e.jsxs("p",{children:["When ",e.jsx(t.InlineMath,{math:"\\hat{A} < 0"})," (bad action): the objective is capped at ",e.jsx(t.InlineMath,{math:"(1-\\varepsilon)\\hat{A}"}),", preventing the ratio from shrinking too much."]}),e.jsx("p",{className:"mt-2",children:"This creates a pessimistic bound: the policy cannot change too aggressively in either direction."})]}),e.jsx(u,{title:"PPO in Practice",children:e.jsxs("p",{children:["Standard PPO hyperparameters: ",e.jsx(t.InlineMath,{math:"\\varepsilon=0.2"}),", learning rate ",e.jsx(t.InlineMath,{math:"3 \\times 10^{-4}"}),", GAE ",e.jsx(t.InlineMath,{math:"\\lambda=0.95"}),", ",e.jsx(t.InlineMath,{math:"\\gamma=0.99"}),", minibatch size 64, 4 epochs per rollout. PPO is the algorithm behind InstructGPT and ChatGPT's RLHF stage."]})}),e.jsx(j,{title:"PPO Update Step",code:`import torch
import torch.nn as nn
from torch.distributions import Categorical

def ppo_update(model, optimizer, states, actions, old_log_probs,
               returns, advantages, clip_eps=0.2, epochs=4):
    for _ in range(epochs):
        logits, values = model(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Probability ratio
        ratio = torch.exp(log_probs - old_log_probs.detach())

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = nn.functional.mse_loss(values.squeeze(), returns)
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    return loss.item()`}),e.jsx(f,{type:"note",title:"PPO Variants",children:e.jsxs("p",{children:["PPO-Penalty uses an adaptive KL penalty instead of clipping. In practice, PPO-Clip dominates due to simplicity. For continuous control, the actor outputs Gaussian parameters ",e.jsx(t.InlineMath,{math:"(\\mu, \\sigma)"})," instead of categorical logits."]})})]})}const ye=Object.freeze(Object.defineProperty({__proto__:null,default:Z},Symbol.toStringTag,{value:"Module"}));function X(){const[a,c]=p.useState(.2),i=[2,1,.5,-.5],l=Math.max(...i),s=i.map(x=>Math.exp((x-l)/Math.max(a,.01))),n=s.reduce((x,d)=>x+d,0),r=s.map(x=>x/n),o=-r.reduce((x,d)=>x+(d>0?d*Math.log(d):0),0),h=["a0","a1","a2","a3"];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Entropy Temperature Effect"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3",children:[e.jsxs("label",{className:"flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",children:["alpha = ",a.toFixed(2),e.jsx("input",{type:"range",min:.01,max:2,step:.01,value:a,onChange:x=>c(parseFloat(x.target.value)),className:"w-28 accent-violet-500"})]}),e.jsxs("span",{className:"text-sm text-violet-600 dark:text-violet-400",children:["H = ",o.toFixed(3)]})]}),e.jsx("div",{className:"flex gap-2 justify-center items-end",style:{height:80},children:h.map((x,d)=>e.jsxs("div",{className:"flex flex-col items-center",children:[e.jsx("div",{className:"w-12 bg-violet-500 rounded-t",style:{height:r[d]*70}}),e.jsx("span",{className:"text-xs text-gray-500 mt-1",children:x}),e.jsxs("span",{className:"text-xs text-violet-600",children:[(r[d]*100).toFixed(1),"%"]})]},d))}),e.jsx("p",{className:"text-center text-xs text-gray-500 mt-2",children:"Higher alpha = more uniform (exploratory) policy"})]})}function Y(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Soft Actor-Critic (SAC) is the leading off-policy actor-critic algorithm for continuous control. It maximizes a combination of expected return and policy entropy, encouraging robust exploration."}),e.jsxs(m,{title:"Maximum Entropy RL Objective",children:[e.jsx(t.BlockMath,{math:"J(\\pi) = \\sum_{t=0}^T \\mathbb{E}\\left[r(s_t, a_t) + \\alpha \\mathcal{H}(\\pi(\\cdot|s_t))\\right]"}),e.jsxs("p",{className:"mt-2",children:["The temperature ",e.jsx(t.InlineMath,{math:"\\alpha"})," balances reward maximization and entropy. The soft value functions become:"]}),e.jsx(t.BlockMath,{math:"Q^{\\text{soft}}(s,a) = r + \\gamma \\mathbb{E}_{s'}\\left[V^{\\text{soft}}(s')\\right], \\quad V^{\\text{soft}}(s) = \\mathbb{E}_{a \\sim \\pi}\\left[Q(s,a) - \\alpha \\log \\pi(a|s)\\right]"})]}),e.jsxs(b,{title:"SAC Components",id:"sac-components",children:[e.jsx("p",{children:"SAC maintains five networks:"}),e.jsx(t.BlockMath,{math:"\\pi_\\theta \\text{ (actor)}, \\quad Q_{\\phi_1}, Q_{\\phi_2} \\text{ (twin critics)}, \\quad \\bar{Q}_{\\phi_1}, \\bar{Q}_{\\phi_2} \\text{ (target critics)}"}),e.jsx("p",{className:"mt-2",children:"Twin critics (clipped double Q) take the minimum to combat overestimation:"}),e.jsx(t.BlockMath,{math:"y = r + \\gamma \\left(\\min_{i=1,2} Q_{\\bar{\\phi}_i}(s', \\tilde{a}') - \\alpha \\log \\pi(\\tilde{a}'|s')\\right)"})]}),e.jsx(X,{}),e.jsxs(u,{title:"Automatic Temperature Tuning",children:[e.jsxs("p",{children:["SAC can automatically adjust ",e.jsx(t.InlineMath,{math:"\\alpha"})," to maintain a target entropy:"]}),e.jsx(t.BlockMath,{math:"\\alpha^* = \\arg\\min_\\alpha \\mathbb{E}_{a \\sim \\pi}\\left[-\\alpha \\log \\pi(a|s) - \\alpha \\bar{\\mathcal{H}}\\right]"}),e.jsxs("p",{children:["where ",e.jsx(t.InlineMath,{math:"\\bar{\\mathcal{H}} = -\\dim(\\mathcal{A})"})," is a common heuristic target for continuous actions."]})]}),e.jsx(j,{title:"SAC Critic and Actor Updates",code:`import torch
import torch.nn as nn

def sac_critic_loss(q1, q2, target_q1, target_q2, policy,
                    states, actions, rewards, next_states, dones,
                    alpha=0.2, gamma=0.99):
    with torch.no_grad():
        next_actions, next_log_probs = policy.sample(next_states)
        q1_next = target_q1(next_states, next_actions)
        q2_next = target_q2(next_states, next_actions)
        min_q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
        target = rewards + gamma * (1 - dones) * min_q_next

    q1_pred = q1(states, actions)
    q2_pred = q2(states, actions)
    loss = nn.functional.mse_loss(q1_pred, target) + \\
           nn.functional.mse_loss(q2_pred, target)
    return loss

def sac_actor_loss(policy, q1, q2, states, alpha=0.2):
    actions, log_probs = policy.sample(states)
    q_val = torch.min(q1(states, actions), q2(states, actions))
    loss = (alpha * log_probs - q_val).mean()  # maximize Q - alpha*logpi
    return loss`}),e.jsx(f,{type:"note",title:"SAC vs PPO",children:e.jsxs("p",{children:[e.jsx("strong",{children:"SAC"}),": Off-policy, sample-efficient, best for continuous control (robotics, locomotion). ",e.jsx("strong",{children:"PPO"}),": On-policy, simpler, better for discrete actions and language model fine-tuning. SAC reuses past data via replay buffers; PPO discards data after each update."]})})]})}const be=Object.freeze(Object.defineProperty({__proto__:null,default:Y},Symbol.toStringTag,{value:"Module"}));function ee(){const[a,c]=p.useState([3.2,1.8,4.1,2.5]),i=["Response A","Response B","Response C","Response D"];Math.max(...a);const l=a.map(o=>Math.exp(o)),s=l.reduce((o,h)=>o+h,0),n=l.map(o=>o/s),r=(o,h)=>{const x=[...a];x[o]=h,c(x)};return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"Reward Model Scores"}),e.jsx("div",{className:"space-y-2",children:i.map((o,h)=>e.jsxs("div",{className:"flex items-center gap-3",children:[e.jsx("span",{className:"text-sm text-gray-600 dark:text-gray-400 w-24",children:o}),e.jsx("input",{type:"range",min:0,max:5,step:.1,value:a[h],onChange:x=>r(h,parseFloat(x.target.value)),className:"w-24 accent-violet-500"}),e.jsx("div",{className:"w-32 h-5 bg-gray-200 dark:bg-gray-700 rounded overflow-hidden",children:e.jsx("div",{className:"h-full bg-violet-500 rounded",style:{width:`${n[h]*100}%`}})}),e.jsxs("span",{className:"text-xs text-violet-600 dark:text-violet-400 font-mono w-16",children:["r=",a[h].toFixed(1)," (",(n[h]*100).toFixed(0),"%)"]})]},h))}),e.jsx("p",{className:"text-center text-xs text-gray-500 mt-2",children:"Bradley-Terry probabilities from reward scores"})]})}function te(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Reward modeling learns a scalar reward function from human preference comparisons, enabling RL to optimize for objectives that are hard to specify programmatically."}),e.jsxs(m,{title:"Bradley-Terry Preference Model",children:[e.jsxs("p",{children:["Given two responses ",e.jsx(t.InlineMath,{math:"y_w"})," (preferred) and ",e.jsx(t.InlineMath,{math:"y_l"})," (dispreferred) to a prompt ",e.jsx(t.InlineMath,{math:"x"}),", the probability of the observed preference is:"]}),e.jsx(t.BlockMath,{math:"P(y_w \\succ y_l | x) = \\sigma\\!\\left(r_\\theta(x, y_w) - r_\\theta(x, y_l)\\right)"}),e.jsxs("p",{className:"mt-2",children:["where ",e.jsx(t.InlineMath,{math:"r_\\theta"})," is the reward model and ",e.jsx(t.InlineMath,{math:"\\sigma"})," is the sigmoid function."]})]}),e.jsxs(m,{title:"Reward Model Training Loss",children:[e.jsx(t.BlockMath,{math:"\\mathcal{L}(\\theta) = -\\mathbb{E}_{(x, y_w, y_l) \\sim \\mathcal{D}}\\left[\\log \\sigma\\!\\left(r_\\theta(x, y_w) - r_\\theta(x, y_l)\\right)\\right]"}),e.jsx("p",{className:"mt-2",children:"This is equivalent to binary cross-entropy on pairwise comparisons."})]}),e.jsx(ee,{}),e.jsxs(b,{title:"Reward Model Architecture",id:"rm-architecture",children:[e.jsx("p",{children:"Typically, the reward model is a pretrained language model with the unembedding head replaced by a scalar projection:"}),e.jsx(t.BlockMath,{math:"r_\\theta(x, y) = \\text{Linear}\\!\\left(\\text{LLM}_\\theta([x; y])_{\\text{last}}\\right) \\in \\mathbb{R}"}),e.jsx("p",{className:"mt-2",children:"The model is initialized from the SFT checkpoint to preserve language understanding."})]}),e.jsxs(u,{title:"Data Collection Pipeline",children:[e.jsx("p",{children:"1. Sample K responses from the SFT model for each prompt."}),e.jsx("p",{children:"2. Human annotators rank or compare pairs of responses."}),e.jsxs("p",{children:["3. From K responses, extract ",e.jsx(t.InlineMath,{math:"\\binom{K}{2}"})," pairwise comparisons."]}),e.jsx("p",{children:"InstructGPT used K=4 responses per prompt, giving 6 pairs each."})]}),e.jsx(j,{title:"Reward Model Training",code:`import torch
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model  # pretrained LLM
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]  # last token
        return self.reward_head(last_hidden).squeeze(-1)

def reward_model_loss(rm, chosen_ids, chosen_mask, rejected_ids, rejected_mask):
    r_chosen = rm(chosen_ids, chosen_mask)
    r_rejected = rm(rejected_ids, rejected_mask)
    loss = -torch.log(torch.sigmoid(r_chosen - r_rejected)).mean()
    accuracy = (r_chosen > r_rejected).float().mean()
    return loss, accuracy`}),e.jsx(f,{type:"note",title:"Reward Hacking",children:e.jsxs("p",{children:["The RL policy may find ways to achieve high reward scores that do not correspond to genuinely better outputs. This is ",e.jsx("strong",{children:"reward hacking"}),". Mitigations include KL penalties against the reference policy, reward model ensembles, and iterative data collection with updated policies."]})})]})}const _e=Object.freeze(Object.defineProperty({__proto__:null,default:te},Symbol.toStringTag,{value:"Module"}));function ae(){const[a,c]=p.useState(0),i=[{name:"Pre-training",desc:"Train LLM on large text corpus",color:"#7c3aed"},{name:"SFT",desc:"Fine-tune on demonstrations",color:"#f97316"},{name:"Reward Model",desc:"Train RM from comparisons",color:"#10b981"},{name:"PPO",desc:"Optimize policy with RM signal",color:"#f43f5e"}];return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"RLHF Pipeline Stages"}),e.jsxs("div",{className:"flex items-center gap-2 mb-4",children:[e.jsx("label",{className:"text-sm text-gray-600 dark:text-gray-400",children:"Stage:"}),i.map((l,s)=>e.jsx("button",{onClick:()=>c(s),className:`px-3 py-1 rounded text-xs font-medium transition-colors ${s===a?"text-white":"text-gray-600 bg-gray-100 dark:bg-gray-800"}`,style:s===a?{backgroundColor:l.color}:{},children:l.name},s))]}),e.jsx("div",{className:"flex items-center gap-1 justify-center",children:i.map((l,s)=>e.jsxs("div",{className:"flex items-center",children:[e.jsx("div",{className:`w-20 h-14 rounded-lg flex flex-col items-center justify-center text-xs transition-opacity ${s<=a?"opacity-100":"opacity-30"}`,style:{backgroundColor:l.color+"22",border:`2px solid ${l.color}`},children:e.jsx("span",{style:{color:l.color},className:"font-bold",children:l.name})}),s<3&&e.jsx("span",{className:"mx-1 text-gray-400",children:"→"})]},s))}),e.jsx("p",{className:"text-center text-sm text-gray-600 dark:text-gray-400 mt-3 font-medium",children:i[a].desc})]})}function se(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"The full RLHF pipeline combines supervised fine-tuning, reward modeling, and PPO-based optimization to align language models with human preferences."}),e.jsx(ae,{}),e.jsxs(m,{title:"KL-Penalized RL Objective",children:[e.jsx("p",{children:"The PPO stage optimizes:"}),e.jsx(t.BlockMath,{math:"\\max_\\pi \\; \\mathbb{E}_{x \\sim \\mathcal{D},\\, y \\sim \\pi}\\left[r_\\theta(x, y) - \\beta D_\\text{KL}(\\pi(\\cdot|x) \\| \\pi_\\text{ref}(\\cdot|x))\\right]"}),e.jsxs("p",{className:"mt-2",children:["The KL penalty prevents the policy from deviating too far from the SFT model ",e.jsx(t.InlineMath,{math:"\\pi_\\text{ref}"}),", reducing reward hacking. ",e.jsx(t.InlineMath,{math:"\\beta"})," is typically 0.01-0.2."]})]}),e.jsxs(b,{title:"Per-Token KL Computation",id:"per-token-kl",children:[e.jsx("p",{children:"In practice, KL divergence is computed token by token:"}),e.jsx(t.BlockMath,{math:"D_\\text{KL} = \\sum_{t=1}^T \\left[\\log \\pi(y_t|x, y_{<t}) - \\log \\pi_\\text{ref}(y_t|x, y_{<t})\\right]"}),e.jsx("p",{className:"mt-2",children:"This per-token KL is added as a penalty to the reward at each token position, shaping the reward to discourage divergence throughout generation."})]}),e.jsxs(u,{title:"InstructGPT Numbers",children:[e.jsxs("p",{children:[e.jsx("strong",{children:"Pre-training"}),": 175B params, ~300B tokens."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"SFT"}),": ~13K demonstrations from labelers."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"RM"}),": ~33K comparisons, trained for 1 epoch to avoid overfitting."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"PPO"}),": ~31K prompts, trained for a few epochs with KL penalty."]}),e.jsx("p",{children:"Result: 1.3B RLHF model preferred over 175B SFT model by human raters."})]}),e.jsx(j,{title:"RLHF PPO Training Loop (Simplified)",code:`import torch

def rlhf_ppo_step(policy, ref_policy, reward_model, optimizer,
                   prompts, kl_coeff=0.1, clip_eps=0.2):
    # 1. Generate responses from current policy
    with torch.no_grad():
        responses, old_log_probs = policy.generate(prompts)
        ref_log_probs = ref_policy.log_probs(prompts, responses)

    # 2. Score with reward model
    with torch.no_grad():
        rewards = reward_model(prompts, responses)

    # 3. Compute per-token KL penalty
    with torch.no_grad():
        kl_penalty = old_log_probs - ref_log_probs  # per token
        shaped_rewards = rewards - kl_coeff * kl_penalty.sum(dim=-1)

    # 4. Compute advantages (simplified - use GAE in practice)
    advantages = shaped_rewards - shaped_rewards.mean()
    advantages = advantages / (advantages.std() + 1e-8)

    # 5. PPO clipped update
    new_log_probs = policy.log_probs(prompts, responses)
    ratio = torch.exp(new_log_probs.sum(-1) - old_log_probs.sum(-1))
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    loss = -torch.min(surr1, surr2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"loss": loss.item(), "mean_reward": rewards.mean().item()}`}),e.jsx(P,{title:"Training Instabilities",children:e.jsxs("p",{children:["RLHF training can be unstable. Common issues: reward model overfitting (use early stopping), KL divergence explosion (clip or increase ",e.jsx(t.InlineMath,{math:"\\beta"}),"), mode collapse (monitor generation diversity), and reward hacking (verify with held-out evaluators)."]})}),e.jsx(f,{type:"note",title:"Beyond InstructGPT",children:e.jsx("p",{children:"Modern RLHF pipelines often use iterative training: collect new preferences on the latest policy outputs, retrain the reward model, and run another round of PPO. Constitutional AI (Anthropic) uses AI feedback instead of human labels for some stages."})})]})}const ve=Object.freeze(Object.defineProperty({__proto__:null,default:se},Symbol.toStringTag,{value:"Module"}));function re(){const[a,c]=p.useState("rlhf"),s=a==="rlhf"?["Collect preferences","Train reward model","Run PPO","Iterate"]:["Collect preferences","Train policy directly","Done"],n=a==="rlhf"?"#f97316":"#7c3aed";return e.jsxs("div",{className:"my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50",children:[e.jsx("h3",{className:"mb-1 text-base font-bold text-gray-800 dark:text-gray-200",children:"RLHF vs DPO Pipeline"}),e.jsxs("div",{className:"flex items-center gap-4 mb-3",children:[e.jsx("button",{onClick:()=>c("rlhf"),className:`px-3 py-1 rounded text-sm ${a==="rlhf"?"bg-orange-500 text-white":"bg-gray-100 text-gray-600 dark:bg-gray-800"}`,children:"RLHF"}),e.jsx("button",{onClick:()=>c("dpo"),className:`px-3 py-1 rounded text-sm ${a==="dpo"?"bg-violet-500 text-white":"bg-gray-100 text-gray-600 dark:bg-gray-800"}`,children:"DPO"})]}),e.jsx("div",{className:"flex items-center gap-2 justify-center flex-wrap",children:s.map((r,o)=>e.jsxs("div",{className:"flex items-center",children:[e.jsx("div",{className:"px-3 py-2 rounded-lg text-xs font-medium text-center",style:{backgroundColor:n+"22",border:`1.5px solid ${n}`,color:n},children:r}),o<s.length-1&&e.jsx("span",{className:"mx-1 text-gray-400",children:"→"})]},o))}),e.jsx("p",{className:"text-center text-xs text-gray-500 mt-2",children:"DPO eliminates the reward model and RL loop entirely"})]})}function ne(){return e.jsxs("div",{className:"space-y-6",children:[e.jsx("p",{className:"text-gray-700 dark:text-gray-300 leading-relaxed",children:"Direct Preference Optimization (DPO) bypasses reward modeling and RL entirely, optimizing the policy directly on preference data. It is analytically equivalent to RLHF with a specific reward parameterization."}),e.jsxs(b,{title:"DPO Key Insight",id:"dpo-derivation",children:[e.jsx("p",{children:"The optimal policy under the KL-constrained RLHF objective has a closed form:"}),e.jsx(t.BlockMath,{math:"\\pi^*(y|x) = \\frac{1}{Z(x)} \\pi_\\text{ref}(y|x) \\exp\\!\\left(\\frac{r(x,y)}{\\beta}\\right)"}),e.jsx("p",{className:"mt-2",children:"Inverting this gives the implicit reward:"}),e.jsx(t.BlockMath,{math:"r(x,y) = \\beta \\log \\frac{\\pi^*(y|x)}{\\pi_\\text{ref}(y|x)} + \\beta \\log Z(x)"})]}),e.jsxs(m,{title:"DPO Loss Function",children:[e.jsx("p",{children:"Substituting the implicit reward into the Bradley-Terry model, the partition function cancels:"}),e.jsx(t.BlockMath,{math:"\\mathcal{L}_\\text{DPO}(\\theta) = -\\mathbb{E}_{(x,y_w,y_l)}\\left[\\log \\sigma\\!\\left(\\beta \\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_\\text{ref}(y_w|x)} - \\beta \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_\\text{ref}(y_l|x)}\\right)\\right]"}),e.jsx("p",{className:"mt-2",children:"This is a simple classification loss that can be optimized with standard SGD."})]}),e.jsx(re,{}),e.jsxs(u,{title:"DPO Advantages",children:[e.jsxs("p",{children:[e.jsx("strong",{children:"Simplicity"}),": No reward model training, no RL loop, no value function."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Stability"}),": Standard supervised learning, no PPO hyperparameters to tune."]}),e.jsxs("p",{children:[e.jsx("strong",{children:"Efficiency"}),": Only needs forward/backward passes through the policy model."]}),e.jsx("p",{children:"DPO matches or exceeds RLHF performance on summarization and dialogue benchmarks."})]}),e.jsx(j,{title:"DPO Training Implementation",code:`import torch
import torch.nn.functional as F

def dpo_loss(policy, ref_policy, chosen_ids, rejected_ids,
             chosen_mask, rejected_mask, beta=0.1):
    # Compute log probabilities under both models
    pi_chosen = policy.log_probs(chosen_ids, chosen_mask)
    pi_rejected = policy.log_probs(rejected_ids, rejected_mask)

    with torch.no_grad():
        ref_chosen = ref_policy.log_probs(chosen_ids, chosen_mask)
        ref_rejected = ref_policy.log_probs(rejected_ids, rejected_mask)

    # DPO implicit reward difference
    chosen_reward = beta * (pi_chosen - ref_chosen)
    rejected_reward = beta * (pi_rejected - ref_rejected)

    # Bradley-Terry loss with implicit rewards
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()

    # Useful metrics
    with torch.no_grad():
        reward_margin = (chosen_reward - rejected_reward).mean()
        accuracy = (chosen_reward > rejected_reward).float().mean()
    return loss, {"margin": reward_margin.item(), "acc": accuracy.item()}`}),e.jsx(f,{type:"note",title:"Beyond DPO",children:e.jsxs("p",{children:["Variants include ",e.jsx("strong",{children:"IPO"})," (identity preference optimization, avoids overfitting),",e.jsx("strong",{children:"KTO"})," (Kahneman-Tversky optimization, works with binary feedback instead of pairs), and ",e.jsx("strong",{children:"ORPO"})," (odds ratio preference optimization, combines SFT and alignment in one step). The field is evolving rapidly toward simpler alignment methods."]})})]})}const ke=Object.freeze(Object.defineProperty({__proto__:null,default:ne},Symbol.toStringTag,{value:"Module"}));export{de as a,he as b,me as c,pe as d,xe as e,ge as f,ue as g,je as h,fe as i,ye as j,be as k,_e as l,ve as m,ke as n,ce as s};

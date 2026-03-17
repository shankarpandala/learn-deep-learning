import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function ReActStepSimulator() {
  const [step, setStep] = useState(0)
  const trace = [
    { type: 'Thought', content: 'I need to find the population of France and Germany to compare them.' },
    { type: 'Action', content: 'search("population of France 2024")' },
    { type: 'Observation', content: 'France population: approximately 68.2 million (2024)' },
    { type: 'Thought', content: 'Now I need Germany\'s population.' },
    { type: 'Action', content: 'search("population of Germany 2024")' },
    { type: 'Observation', content: 'Germany population: approximately 84.5 million (2024)' },
    { type: 'Thought', content: 'Germany (84.5M) has a larger population than France (68.2M) by about 16.3 million.' },
    { type: 'Answer', content: 'Germany has a larger population than France by approximately 16.3 million people.' },
  ]
  const colors = { Thought: 'bg-violet-50 dark:bg-violet-900/20 border-violet-200', Action: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200', Observation: 'bg-green-50 dark:bg-green-900/20 border-green-200', Answer: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200' }

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">ReAct Trace Walkthrough</h3>
      <div className="flex items-center gap-2 mb-3">
        <button onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0} className="px-2 py-1 rounded text-sm bg-gray-100 dark:bg-gray-800 disabled:opacity-40">Prev</button>
        <span className="text-sm text-gray-500">Step {step + 1}/{trace.length}</span>
        <button onClick={() => setStep(Math.min(trace.length - 1, step + 1))} disabled={step === trace.length - 1} className="px-2 py-1 rounded text-sm bg-gray-100 dark:bg-gray-800 disabled:opacity-40">Next</button>
      </div>
      <div className="space-y-2">
        {trace.slice(0, step + 1).map((t, i) => (
          <div key={i} className={`p-2 rounded border text-sm ${colors[t.type]} ${i === step ? 'ring-2 ring-violet-400' : 'opacity-70'}`}>
            <span className="font-medium text-xs">{t.type}:</span> <span className="text-gray-700 dark:text-gray-300">{t.content}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function AgentFrameworks() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        LLM agents combine reasoning, tool use, and memory to autonomously complete complex
        multi-step tasks. Frameworks like ReAct, Reflexion, and multi-agent systems provide
        structured approaches to agentic behavior.
      </p>

      <DefinitionBlock title="ReAct: Reasoning + Acting">
        <p>ReAct interleaves reasoning traces with tool actions in a loop:</p>
        <BlockMath math="\text{Thought}_t \to \text{Action}_t \to \text{Observation}_t \to \text{Thought}_{t+1} \to \cdots" />
        <p className="mt-2">The thought step enables the model to plan and reflect before acting, while observations from the environment ground reasoning in real-world feedback. This outperforms both reasoning-only and acting-only approaches.</p>
      </DefinitionBlock>

      <ReActStepSimulator />

      <ExampleBlock title="Agent Design Patterns">
        <ul className="list-disc list-inside space-y-1">
          <li><strong>ReAct:</strong> Single-turn thought-action-observation loop</li>
          <li><strong>Reflexion:</strong> Self-reflection on failures to improve on retry</li>
          <li><strong>Plan-and-Execute:</strong> Create full plan first, then execute steps</li>
          <li><strong>Multi-agent:</strong> Specialized agents (coder, reviewer, planner) collaborate</li>
          <li><strong>Hierarchical:</strong> Manager agent delegates subtasks to worker agents</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Minimal ReAct Agent Loop"
        code={`class ReActAgent:
    """Minimal ReAct agent with thought-action-observation loop."""
    def __init__(self, llm, tools, max_steps=10):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps

    def run(self, question):
        trajectory = [f"Question: {question}"]

        for step in range(self.max_steps):
            # Generate thought + action
            prompt = "\\n".join(trajectory)
            response = self.llm(prompt)  # Returns thought + action

            if "Answer:" in response:
                return response.split("Answer:")[-1].strip()

            # Parse action
            thought, action = self.parse_response(response)
            trajectory.append(f"Thought: {thought}")
            trajectory.append(f"Action: {action}")

            # Execute tool and get observation
            tool_name, args = self.parse_action(action)
            observation = self.tools[tool_name].execute(**args)
            trajectory.append(f"Observation: {observation}")

        return "Max steps reached without answer"

    def parse_response(self, response):
        # Extract thought and action from LLM output
        lines = response.strip().split("\\n")
        thought = lines[0].replace("Thought:", "").strip()
        action = lines[1].replace("Action:", "").strip() if len(lines) > 1 else ""
        return thought, action

    def parse_action(self, action_str):
        # Parse "tool_name(arg1, arg2)" format
        name = action_str.split("(")[0]
        args_str = action_str.split("(")[1].rstrip(")")
        return name, {"query": args_str}

print("ReAct loop: Think -> Act -> Observe -> Think -> ... -> Answer")`}
      />

      <WarningBlock title="Agent Reliability Challenges">
        <p>
          Current LLM agents face compounding errors: if each step has 90% accuracy, a 10-step
          task succeeds only <InlineMath math="0.9^{10} \approx 35\%" /> of the time. Error recovery,
          verification, and human-in-the-loop checkpoints are essential for real-world deployment.
          The field is actively working on improving single-step reliability and adding self-correction.
        </p>
      </WarningBlock>
    </div>
  )
}

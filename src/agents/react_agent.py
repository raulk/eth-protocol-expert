"""ReAct Agent - Reasoning and Acting loop for agentic retrieval.

Phase 8 enhancements:
- Adaptive budget selection based on query complexity analysis
- Integration with QueryAnalyzer for automatic mode selection
- Dynamic retrieval strategy based on query characteristics
"""

import asyncio
import os
from dataclasses import dataclass, field
from enum import Enum

import anthropic
import structlog

from src.agents.backtrack import Backtracker
from src.agents.budget_enforcer import AgentBudget, BudgetEnforcer
from src.agents.query_analyzer import QueryAnalyzer
from src.agents.reflection import Reflector
from src.agents.retrieval_tool import RetrievalMode, RetrievalTool

logger = structlog.get_logger()


class AgentAction(Enum):
    """Actions the agent can take."""

    THINK = "think"
    RETRIEVE = "retrieve"
    ANSWER = "answer"


@dataclass
class Thought:
    """A single thought in the agent's reasoning chain."""

    content: str
    action: AgentAction
    action_input: str | None = None


@dataclass
class AgentState:
    """Current state of the ReAct agent."""

    query: str
    thoughts: list[Thought] = field(default_factory=list)
    retrievals: list[dict] = field(default_factory=list)
    answer: str | None = None
    budget_remaining: int = 5

    def add_thought(self, content: str, action: AgentAction, action_input: str | None = None):
        """Add a thought to the reasoning chain."""
        self.thoughts.append(Thought(content=content, action=action, action_input=action_input))

    def get_thought_history(self) -> str:
        """Format thought history for prompts."""
        if not self.thoughts:
            return "No thoughts yet."

        parts = []
        for i, thought in enumerate(self.thoughts):
            parts.append(f"Step {i + 1}:")
            parts.append(f"  Thought: {thought.content}")
            parts.append(f"  Action: {thought.action.value}")
            if thought.action_input:
                parts.append(f"  Input: {thought.action_input[:100]}...")

        return "\n".join(parts)

    def get_retrieved_context(self, max_chunks: int = 10) -> str:
        """Format retrieved chunks as context."""
        if not self.retrievals:
            return "No information retrieved yet."

        chunks = self.retrievals[:max_chunks]
        parts = []

        for i, chunk in enumerate(chunks):
            source = chunk.get("document_id", "unknown").upper()
            section = chunk.get("section_path")
            if section:
                source = f"{source} ({section})"

            content = chunk.get("content", "")[:500]
            parts.append(f"[{i + 1}] [{source}]\n{content}")

        return "\n\n---\n\n".join(parts)


@dataclass
class AgentResult:
    """Final result from the ReAct agent."""

    query: str
    answer: str
    thoughts: list[Thought]
    retrievals: list[dict]
    total_tokens_retrieved: int
    llm_calls: int
    retrieval_count: int
    success: bool
    termination_reason: str


class ReactAgent:
    """ReAct (Reasoning + Acting) agent for intelligent retrieval.

    Implements the ReAct loop:
    1. Think - Reason about what to do next
    2. Act - Either retrieve information or generate answer
    3. Observe - Process the result
    4. Repeat until done or budget exhausted

    Uses reflection to assess retrieval quality and backtracking
    to escape dead-end paths.
    """

    REACT_PROMPT = """You are an expert Ethereum protocol analyst. Your task is to answer questions about Ethereum Improvement Proposals (EIPs) and protocol development.

You follow a ReAct (Reasoning + Acting) approach:
1. Think about what you need to find
2. Retrieve relevant information
3. Reflect on whether you have enough
4. Answer when ready

Current Query: {query}

Thought History:
{thought_history}

Retrieved Information:
{retrieved_context}

Budget Remaining: {budget_remaining} retrieval(s)

Based on the above, decide your next action. You must respond with exactly one of:

THINK: [your reasoning about what to do next]
RETRIEVE: [specific search query to find information]
ANSWER: [your complete answer to the query]

Guidelines:
- If you haven't retrieved anything yet, start with RETRIEVE
- If retrieved info seems insufficient, use RETRIEVE with a refined query
- If you have enough information, use ANSWER
- If budget is low (1-2 remaining), prioritize ANSWER with available info
- Be specific in RETRIEVE queries to get relevant results

Respond with your chosen action:"""

    ANSWER_PROMPT = """You are an expert Ethereum protocol analyst. Based on the retrieved information, provide a complete answer to the query.

Query: {query}

Retrieved Information:
{retrieved_context}

Reasoning Chain:
{thought_history}

Provide a clear, accurate, and comprehensive answer. If some information is missing or uncertain, acknowledge that. Always cite sources when possible using [EIP-XXXX] format."""

    def __init__(
        self,
        retrieval_tool: RetrievalTool,
        budget: AgentBudget | None = None,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        enable_reflection: bool = True,
        enable_backtracking: bool = True,
        enable_adaptive_budget: bool = True,
    ):
        self.retrieval_tool = retrieval_tool
        self.budget = budget or AgentBudget()
        self.budget_enforcer = BudgetEnforcer(self.budget)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.enable_adaptive_budget = enable_adaptive_budget
        self.query_analyzer = QueryAnalyzer() if enable_adaptive_budget else None

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)

        self.enable_reflection = enable_reflection
        self.enable_backtracking = enable_backtracking

        if enable_reflection:
            self.reflector = Reflector(api_key=self.api_key, model=model)
        else:
            self.reflector = None

        if enable_backtracking:
            self.backtracker = Backtracker()
        else:
            self.backtracker = None

    async def run(self, query: str) -> AgentResult:
        """Run the ReAct loop to answer a query.

        Args:
            query: The user's question

        Returns:
            AgentResult with answer and metadata
        """
        # Phase 8: Adaptive budget selection based on query complexity
        query_analysis = None
        effective_budget = self.budget
        preferred_mode = RetrievalMode.HYBRID

        if self.query_analyzer and self.enable_adaptive_budget:
            query_analysis = self.query_analyzer.analyze(query)
            effective_budget = query_analysis.suggested_budget

            # Map suggested mode to RetrievalMode
            mode_map = {
                "simple": RetrievalMode.VECTOR,
                "hybrid": RetrievalMode.HYBRID,
                "graph": RetrievalMode.GRAPH,
                "agentic": RetrievalMode.HYBRID,
            }
            preferred_mode = mode_map.get(query_analysis.suggested_mode, RetrievalMode.HYBRID)

            logger.info(
                "query_analyzed",
                query=query[:50],
                complexity=query_analysis.complexity.value,
                signals=query_analysis.signals,
                suggested_mode=query_analysis.suggested_mode,
                budget_retrievals=effective_budget.max_retrievals,
            )

        # Use effective budget (either provided or from analysis)
        self.budget_enforcer = BudgetEnforcer(effective_budget)
        self.budget_enforcer.reset()

        if self.backtracker:
            self.backtracker.reset()

        state = AgentState(
            query=query,
            budget_remaining=effective_budget.max_retrievals,
        )

        llm_calls = 0
        retrieval_count = 0
        total_tokens = 0

        logger.info("react_agent_started", query=query[:100])

        while self.budget_enforcer.check_budget(state):
            action, action_input = await self._decide_action(state)
            llm_calls += 1

            if action == AgentAction.ANSWER:
                state.answer = await self._generate_answer(state)
                llm_calls += 1

                logger.info(
                    "react_agent_completed",
                    query=query[:50],
                    llm_calls=llm_calls,
                    retrievals=retrieval_count,
                    tokens=total_tokens,
                )

                return AgentResult(
                    query=query,
                    answer=state.answer,
                    thoughts=state.thoughts,
                    retrievals=state.retrievals,
                    total_tokens_retrieved=total_tokens,
                    llm_calls=llm_calls,
                    retrieval_count=retrieval_count,
                    success=True,
                    termination_reason="Answer generated",
                )

            elif action == AgentAction.RETRIEVE:
                # Use preferred mode from query analysis (Phase 8)
                retrieval_result = await self.retrieval_tool.execute(
                    query=action_input,
                    mode=preferred_mode,
                    limit=5,
                )

                retrieved_tokens = retrieval_result.total_tokens
                self.budget_enforcer.deduct_retrieval(state, retrieved_tokens)

                for chunk in retrieval_result.chunks:
                    if chunk not in state.retrievals:
                        state.retrievals.append(chunk)

                retrieval_count += 1
                total_tokens += retrieved_tokens
                state.budget_remaining = self.budget_enforcer.get_remaining()["retrievals"]

                was_useful = len(retrieval_result.chunks) > 0
                if self.backtracker:
                    self.backtracker.record_attempt(
                        query=action_input,
                        mode="hybrid",
                        num_results=len(retrieval_result.chunks),
                        was_useful=was_useful,
                    )

                logger.debug(
                    "retrieval_executed",
                    query=action_input[:50],
                    num_results=len(retrieval_result.chunks),
                    tokens=retrieved_tokens,
                    budget_remaining=state.budget_remaining,
                )

                if self.reflector and retrieval_count > 0:
                    reflection = await self.reflector.reflect(
                        query=query,
                        retrieved=[c for c in state.retrievals],
                        thoughts=[t.content for t in state.thoughts],
                    )
                    llm_calls += 1

                    if reflection.sufficient and reflection.confidence >= 0.8:
                        state.add_thought(
                            f"Reflection indicates sufficient information (confidence: {reflection.confidence:.2f})",
                            AgentAction.THINK,
                        )

                if self.backtracker:
                    backtrack_decision = await self.backtracker.check(state)
                    if backtrack_decision.should_backtrack:
                        state.add_thought(
                            f"Backtracking: {backtrack_decision.reason}",
                            AgentAction.THINK,
                        )

                        if (
                            backtrack_decision.suggested_action
                            == "Generate best answer with available information"
                        ):
                            state.answer = await self._generate_answer(state)
                            llm_calls += 1

                            return AgentResult(
                                query=query,
                                answer=state.answer,
                                thoughts=state.thoughts,
                                retrievals=state.retrievals,
                                total_tokens_retrieved=total_tokens,
                                llm_calls=llm_calls,
                                retrieval_count=retrieval_count,
                                success=True,
                                termination_reason=f"Backtrack: {backtrack_decision.reason}",
                            )

            elif action == AgentAction.THINK:
                state.add_thought(action_input, AgentAction.THINK)

        state.answer = await self._generate_answer(state)
        llm_calls += 1

        logger.info(
            "react_agent_budget_exhausted",
            query=query[:50],
            llm_calls=llm_calls,
            retrievals=retrieval_count,
        )

        return AgentResult(
            query=query,
            answer=state.answer,
            thoughts=state.thoughts,
            retrievals=state.retrievals,
            total_tokens_retrieved=total_tokens,
            llm_calls=llm_calls,
            retrieval_count=retrieval_count,
            success=True,
            termination_reason="Budget exhausted",
        )

    async def _decide_action(self, state: AgentState) -> tuple[AgentAction, str]:
        """Decide the next action based on current state.

        Returns:
            Tuple of (action, action_input)
        """
        if not self.budget_enforcer.deduct_llm_call(state):
            return AgentAction.ANSWER, ""

        prompt = self.REACT_PROMPT.format(
            query=state.query,
            thought_history=state.get_thought_history(),
            retrieved_context=state.get_retrieved_context(),
            budget_remaining=state.budget_remaining,
        )

        response = await asyncio.to_thread(
            self.client.messages.create,
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text.strip()

        action, action_input = self._parse_action(response_text)

        state.add_thought(
            content=action_input if action == AgentAction.THINK else f"Decided to {action.value}",
            action=action,
            action_input=action_input if action != AgentAction.THINK else None,
        )

        logger.debug(
            "action_decided",
            action=action.value,
            input_preview=action_input[:50] if action_input else "",
        )

        return action, action_input

    def _parse_action(self, response_text: str) -> tuple[AgentAction, str]:
        """Parse the LLM response into an action."""
        response_upper = response_text.upper()

        if response_upper.startswith("ANSWER:"):
            return AgentAction.ANSWER, response_text[7:].strip()

        if response_upper.startswith("RETRIEVE:"):
            return AgentAction.RETRIEVE, response_text[9:].strip()

        if response_upper.startswith("THINK:"):
            return AgentAction.THINK, response_text[6:].strip()

        for line in response_text.split("\n"):
            line_upper = line.strip().upper()
            if line_upper.startswith("ANSWER:"):
                return AgentAction.ANSWER, line[7:].strip()
            if line_upper.startswith("RETRIEVE:"):
                return AgentAction.RETRIEVE, line[9:].strip()
            if line_upper.startswith("THINK:"):
                return AgentAction.THINK, line[6:].strip()

        logger.warning("could_not_parse_action", response=response_text[:200])
        return AgentAction.THINK, response_text

    async def _generate_answer(self, state: AgentState) -> str:
        """Generate the final answer based on retrieved information."""
        prompt = self.ANSWER_PROMPT.format(
            query=state.query,
            retrieved_context=state.get_retrieved_context(max_chunks=15),
            thought_history=state.get_thought_history(),
        )

        response = await asyncio.to_thread(
            self.client.messages.create,
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text.strip()

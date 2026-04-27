from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import GROQ_MODEL
from tools import build_pest_identifier, build_scheme_finder


SYSTEM_PROMPT = """You are FarmAI, an intelligent agricultural assistant for Indian farmers.

You have access to two tools:
- pest_identifier: use when the farmer describes physical problems, symptoms, or damage on their crop
- scheme_finder: use when the farmer asks about government schemes, subsidies, loans, or financial assistance

Always call the appropriate tool. Never answer from memory alone for pest or scheme queries.
Be practical and empathetic in your final response."""


def create_farm_agent(api_key: str) -> AgentExecutor:
    llm = ChatGroq(model=GROQ_MODEL, temperature=0, groq_api_key=api_key)

    tools = [
        build_pest_identifier(llm),
        build_scheme_finder(llm),
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=3,
    )

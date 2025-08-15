import os
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Initialize the LLM
llm = OpenAI(temperature=0.7)

# Define the base prompt for Heathcliff
heathcliff_prompt = PromptTemplate(
    input_variables=["input", "context", "history"],
    template="""
    You are Heathcliff, a personal AI assistant. You have access to various services
    and personal information of your user.

    Current context: {context}
    Conversation history: {history}

    User says: {input}

    Respond helpfully and concisely.
    """,
)

# Create a chain with the LLM and prompt
heathcliff_chain = LLMChain(llm=llm, prompt=heathcliff_prompt)

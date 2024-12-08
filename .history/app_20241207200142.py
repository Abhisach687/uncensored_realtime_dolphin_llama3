import subprocess
import requests
from typing import Any, List, Optional, Union

import streamlit as st
from langchain.agents import create_react_agent, Tool
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema import BaseMessage, LLMResult, Generation, PromptValue
from langchain.callbacks.manager import Callbacks
from langchain.prompts import PromptTemplate

class LocalOllamaModel(BaseLanguageModel):
    model_identifier: str

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[Callbacks] = None) -> str:
        result = subprocess.run(
            ["ollama", "run", self.model_identifier, prompt],
            capture_output=True,
            text=True,
            encoding='utf-8',  # Specify UTF-8 encoding
            errors='replace'   # Replace characters that can't be decoded
        )
        return result.stdout.strip()

    # ... [rest of the class remains unchanged] ...

def run_ollama_model(query: str) -> str:
    llm = LocalOllamaModel(model_identifier="dolphin-llama3:latest")
    return llm.predict(query)

def web_search(query: str) -> str:
    search_url = f"https://www.google.com/search?q={query}"
    response = requests.get(search_url)
    return response.text

def create_agent():
    llm = LocalOllamaModel(model_identifier="dolphin-llama3:latest")

    tools = [
        Tool(
            name="WebSearch",
            func=web_search,
            description="Tool to perform web search and retrieve information."
        ),
        Tool(
            name="Ollama",
            func=run_ollama_model,
            description="Tool to interact with the local Ollama model."
        )
    ]

    # Define a prompt template with required variables
    template = """You are a helpful AI assistant. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT: After each Thought, ensure to provide either an Action and Action Input, or a Final Answer.

Begin!

Question: {input}
{agent_scratchpad}"""

    prompt = PromptTemplate(
        input_variables=["tool_descriptions", "tool_names", "input", "agent_scratchpad"],
        template=template
    )

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        handle_parsing_errors=True  # Handle parsing errors gracefully
    )
    return agent, tools  # Return both the agent and the tools list

st.title("Ollama Dolphin Chat Interface")
st.write("Ask anything, and the agent will browse the internet to fetch the latest information.")

agent, tools = create_agent()  # Unpack both agent and tools

user_input = st.text_input("Enter your query:", "", placeholder="Type your question here...")

if st.button("Submit"):
    if user_input.strip():
        with st.spinner("Fetching the latest information..."):
            try:
                # Prepare the tool descriptions and names for the prompt
                tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
                tool_names = ", ".join([tool.name for tool in tools])
                agent_scratchpad = ""  # Initialize as empty or use previous context if needed

                # Invoke the agent with the required variables
                response = agent.invoke({
                    "tool_descriptions": tool_descriptions,
                    "input": user_input,
                    "tool_names": tool_names,
                    "agent_scratchpad": agent_scratchpad,
                    "intermediate_steps": []  # Include an empty list for intermediate steps
                })
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")

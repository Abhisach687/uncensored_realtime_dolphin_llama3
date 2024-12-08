import subprocess
import tempfile
import requests
from typing import Any, List, Optional, Union
import os

import streamlit as st
from langchain.agents import create_react_agent, Tool, AgentExecutor, initialize_agent
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema import BaseMessage, LLMResult, Generation, PromptValue
from langchain.callbacks.manager import Callbacks
from langchain.prompts import PromptTemplate
from pydantic import Field

class LocalOllamaModel(BaseLanguageModel):
    model_identifier: str = Field(default="dolphin-llama3:latest")

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[Callbacks] = None) -> str:
        # Create a temporary file to store the prompt
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
            temp_file.write(prompt)
            temp_file_path = temp_file.name

        # Execute the command using the temporary file
        result = subprocess.run(
            ["ollama", "run", self.model_identifier, "--prompt-file", temp_file_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        # Clean up the temporary file
        os.remove(temp_file_path)

        return result.stdout.strip()

    def invoke(self, input: Union[str, List[BaseMessage], PromptValue], config: Optional[dict] = None, **kwargs: Any) -> str:
        if isinstance(input, str):
            return self._call(input)
        elif isinstance(input, list) and all(isinstance(msg, BaseMessage) for msg in input):
            combined_input = " ".join(msg.content for msg in input)
            return self._call(combined_input)
        elif isinstance(input, PromptValue):
            return self._call(input.to_string())
        else:
            raise TypeError("Unsupported input type for invoke method.")

    async def agenerate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Callbacks] = None,
        **kwargs: Any
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            output = self._call(prompt.to_string(), stop, callbacks)
            generations.append([Generation(text=output)])
        return LLMResult(generations=generations)

    def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Callbacks] = None,
        **kwargs: Any
    ) -> LLMResult:
        return self.agenerate_prompt(prompts, stop, callbacks, **kwargs)

    async def apredict(
        self,
        text: str,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        return self._call(text, stop)

    def predict(
        self,
        text: str,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        return self._call(text, stop)

    async def apredict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> BaseMessage:
        text = " ".join(msg.content for msg in messages)
        response = await self.apredict(text, stop=stop, **kwargs)
        return BaseMessage(content=response)

    def predict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> BaseMessage:
        text = " ".join(msg.content for msg in messages)
        response = self.predict(text, stop=stop, **kwargs)
        return BaseMessage(content=response)

    @property
    def _llm_type(self) -> str:
        return "custom"

def run_ollama_model(query: str) -> str:
    llm = LocalOllamaModel()
    return llm.predict(query)

def web_search(query: str) -> str:
    search_url = f"https://www.google.com/search?q={query}"
    response = requests.get(search_url)
    return response.text

def create_agent():
    llm = LocalOllamaModel()

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

{tools}

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
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
        template=template
    )

    agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",  # simpler agent
    verbose=True
)

    # Create an AgentExecutor without iteration or time limits
    agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    max_iterations=3,  # give it a small number of steps
)

    return agent_executor, tools

st.title("Ollama Dolphin Chat Interface")
st.write("Ask anything, and the agent will browse the internet to fetch the latest information.")

agent_executor, tools = create_agent()

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
                response = agent_executor.invoke({
                    "tools": tool_descriptions,
                    "tool_names": tool_names,
                    "input": user_input,
                    "agent_scratchpad": agent_scratchpad
                })

                # Extract the 'output' key from the response dictionary
                final_output = response.get('output', 'No response received.')

                # Display the final output in Streamlit
                st.write(final_output)

            except Exception as e:
                st.error(f"An error occurred: {e}")
 

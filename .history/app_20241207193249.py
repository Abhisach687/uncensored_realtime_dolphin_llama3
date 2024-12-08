import subprocess
import requests
from typing import Any, List, Optional, Union

import streamlit as st
from langchain.agents import create_react_agent, Tool
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema import BaseMessage, LLMResult, Generation, PromptValue
from langchain.callbacks.manager import Callbacks

class LocalOllamaModel(BaseLanguageModel):
    model_identifier: str  # Define it as a class field so Pydantic knows about it

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[Callbacks] = None) -> str:
        result = subprocess.run(
            ["ollama", "run", self.model_identifier, prompt],
            capture_output=True,
            text=True
        )
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
        # Just call the async version in a sync manner
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
    # Instantiate the model properly
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

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        verbose=True
    )
    return agent


st.title("Ollama Dolphin Chat Interface")
st.write("Ask anything, and the agent will browse the internet to fetch the latest information.")

agent = create_agent()

user_input = st.text_input("Enter your query:", "", placeholder="Type your question here...")

if st.button("Submit"):
    if user_input.strip():
        with st.spinner("Fetching the latest information..."):
            try:
                response = agent.invoke(user_input)
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")

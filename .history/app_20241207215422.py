import subprocess
import tempfile
import requests
from typing import Any, List, Optional, Union
import os
import asyncio

import streamlit as st
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema import BaseMessage, LLMResult, Generation, PromptValue
from langchain.callbacks.manager import Callbacks
from pydantic import Field
from langchain import LLMChain, PromptTemplate

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

    def generate_prompt(
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

    def predict(
        self,
        text: str,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        return self._call(text, stop)

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

    # Async methods required by BaseLanguageModel:
    async def agenerate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Callbacks] = None,
        **kwargs: Any
    ) -> LLMResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_prompt, prompts, stop, callbacks, **kwargs)

    async def apredict(
        self,
        text: str,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict, text, stop, **kwargs)

    async def apredict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> BaseMessage:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict_messages, messages, stop, **kwargs)


# A simple prompt template that tells the model to just answer the user's query.
template = """You are a helpful AI assistant. Answer the user's question as best as possible.
If you don't know the answer, say you don't know.

Question: {query}
Answer:"""

prompt = PromptTemplate(
    input_variables=["query"],
    template=template
)

llm = LocalOllamaModel()
chain = LLMChain(llm=llm, prompt=prompt)

st.title("Ollama Dolphin Chat Interface")
st.write("Ask anything, and the model will answer directly.")

user_input = st.text_input("Enter your query:", "", placeholder="Type your question here...")

if st.button("Submit"):
    if user_input.strip():
        with st.spinner("Fetching the latest information..."):
            try:
                final_output = chain.run(query=user_input)
                st.write(final_output)
            except Exception as e:
                st.error(f"An error occurred: {e}")

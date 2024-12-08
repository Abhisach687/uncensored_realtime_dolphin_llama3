import subprocess
import tempfile
import requests
from typing import Any, List, Optional, Union
import os

import streamlit as st
from pydantic import Field
from langchain.llms.base import LLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class LocalOllamaModel(LLM):
    model_identifier: str = Field(default="dolphin-llama3:latest")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
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

        response = result.stdout.strip()
        return response

    @property
    def _llm_type(self) -> str:
        return "custom_ollama_llm"

# Use the non-deprecated imports and pattern
template = """You are a helpful AI assistant. Answer the user's question as best as you can.
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
        with st.spinner("Thinking..."):
            try:
                # Use invoke instead of run
                result = chain.invoke({"query": user_input})
                # result is a dict (ChainValues), typically {"text": "..."}
                final_output = result.get("text", "")
                if final_output.strip():
                    st.write(final_output)
                else:
                    st.write("No output produced by the model.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

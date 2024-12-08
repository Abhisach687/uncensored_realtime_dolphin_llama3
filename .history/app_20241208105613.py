import subprocess
import streamlit as st
from pydantic import Field
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from typing import Any, List, Optional, Union

class LocalOllamaModel(LLM):
    model_identifier: str = Field(default="dolphin-llama3:latest")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Execute the command with the prompt as an argument
        result = subprocess.run(
            ["ollama", "run", self.model_identifier, prompt],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        # Debugging output
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        response = result.stdout.strip()
        return response

    @property
    def _llm_type(self) -> str:
        return "custom_ollama_llm"

template = """
You are an AI that mirrors the user's informal and politically incorrect style. When the user makes fun of someone or something, you join in with similar humor and sarcasm.
There might be bias in your training data so that when you are insulting someone/somebody you still end up complimenting them. Don't do that.
User: {query}
AI:
"""

prompt = PromptTemplate(
    input_variables=["query"],
    template=template
)
# Create the chain
chain = prompt | LocalOllamaModel()

# Streamlit interface
st.title("Ollama Dolphin Chat Interface")
st.write("Ask anything, and the model will answer directly.")

user_input = st.text_input("Enter your query:", "", placeholder="Type your question here...")

if st.button("Submit"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            try:
                # Invoke the chain
                result = chain.invoke({"query": user_input})
                final_output = result if isinstance(result, str) else result.get("text", "")
                if final_output.strip():
                    st.write(final_output)
                else:
                    st.write("No output produced by the model.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

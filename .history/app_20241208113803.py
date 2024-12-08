import subprocess
from datetime import datetime
import pytz
import streamlit as st
from pydantic import BaseModel, Field
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from typing import Optional, List

class LocalOllamaModel(LLM):
    model_identifier: str = Field(default="dolphin-llama3:latest")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_identifier],
                input=prompt,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            response = result.stdout.strip()
            if not response:
                response = "No output produced by the model."
        except Exception as e:
            response = f"An error occurred while running the model: {e}"
        return response

    @property
    def _llm_type(self) -> str:
        return "custom_ollama_llm"

def get_nepal_time():
    nepal_tz = pytz.timezone('Asia/Kathmandu')
    nepal_time = datetime.now(nepal_tz)
    return nepal_time.strftime('%Y-%m-%d %H:%M:%S')

def should_fetch_real_time_data(query: str) -> bool:
    keywords = ['latest', 'news', 'update', 'current', 'today']
    return any(keyword in query.lower() for keyword in keywords)

# Define the updated prompt template
template = """
You are an AI that adapts to the user's style. When the user seeks factual information, provide accurate and concise answers. When the user engages in informal or humorous conversation, mirror their style with similar humor and sarcasm.

User: {query}
AI:
"""

prompt = PromptTemplate(
    input_variables=["query"],
    template=template
)

# Initialize the LLM and chain
llm = LocalOllamaModel()
chain = prompt | llm

# Streamlit interface
st.title("Custom AI Chat Interface with Real-Time Data")
st.write("Engage with the AI in your preferred style. It fetches real-time data when necessary.")

user_input = st.text_input("Enter your message:", "", placeholder="Type here...")

if st.button("Send"):
    if user_input.strip():
        with st.spinner("Generating response..."):
            try:
                if should_fetch_real_time_data(user_input):
                    nepal_time = get_nepal_time()
                    user_input += f"\n\nReal-time info: The current date and time in Nepal is: {nepal_time}"
                # Invoke the chain
                result = chain.invoke({"query": user_input})
                final_output = result if isinstance(result, str) else result.get("text", "")
                if final_output.strip():
                    st.write(final_output)
                else:
                    st.write("No output produced by the model.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

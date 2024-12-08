import subprocess
import tempfile
import requests
from bs4 import BeautifulSoup
import streamlit as st
from pydantic import BaseModel, Field
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from typing import Optional, List

class LocalOllamaModel(LLM):
    model_identifier: str = Field(default="dolphin-llama3:latest")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Create a temporary file to store the prompt
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
            temp_file.write(prompt)
            temp_file_path = temp_file.name

        try:
            # Execute the command using the temporary file
            result = subprocess.run(
                ["ollama", "run", self.model_identifier, "--prompt-file", temp_file_path],
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
        finally:
            # Clean up the temporary file
            os.remove(temp_file_path)

        return response

    @property
    def _llm_type(self) -> str:
        return "custom_ollama_llm"

def fetch_real_time_data(query: str) -> str:
    """
    Fetches real-time data based on the query using web scraping.
    """
    search_url = f"https://www.google.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract relevant information (this example fetches the first result snippet)
    result = soup.find('div', class_='BNeawe').text
    return result

def should_fetch_real_time_data(query: str) -> bool:
    """
    Determines whether real-time data is needed based on the query context.
    """
    keywords = ['latest', 'news', 'update', 'current', 'today']
    return any(keyword in query.lower() for keyword in keywords)

# Define the prompt template
template = """
You are an AI that mirrors the user's informal and politically incorrect style. When the user makes fun of someone or something, you join in with similar humor and sarcasm.

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
                # Check if real-time data is needed
                if should_fetch_real_time_data(user_input):
                    real_time_info = fetch_real_time_data(user_input)
                    user_input += f"\n\nReal-time info: {real_time_info}"

                # Invoke the chain
                result = chain.invoke({"query": user_input})
                final_output = result if isinstance(result, str) else result.get("text", "")
                if final_output.strip():
                    st.write(final_output)
                else:
                    st.write("No output produced by the model.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

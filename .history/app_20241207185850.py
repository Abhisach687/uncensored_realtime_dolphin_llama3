import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.schema.language_model import BaseLanguageModel
import subprocess
import requests

# Wrapper class for the local Ollama model
class LocalOllamaModel(BaseLanguageModel):
    def __init__(self, model_name):
        self.model_name = model_name

    def _call(self, prompt, stop=None, run_manager=None):
        result = subprocess.run(
            ["ollama", "run", self.model_name, prompt],
            capture_output=True,
            text=True
        )
        return result.stdout

    @property
    def _llm_type(self):
        return "custom"

# Function to run the local Ollama model
def run_ollama_model(query):
    llm = LocalOllamaModel("dolphin-llama3:latest")
    return llm(query)

# Custom function to perform web search
def web_search(query):
    search_url = f"https://www.google.com/search?q={query}"
    response = requests.get(search_url)
    return response.text

# Function to initialize the LangChain agent
def create_agent():
    # Initialize the local Ollama model
    llm = LocalOllamaModel("dolphin-llama3:latest")

    # Define tools
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

    # Initialize the agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent

# Initialize Streamlit app
st.title("Ollama Dolphin Chat Interface")
st.write("Ask anything, and the agent will browse the internet to fetch the latest information.")

# Create a chat agent
agent = create_agent()

# User input
user_input = st.text_input("Enter your query:", "", placeholder="Type your question here...")

# Handle user query
if st.button("Submit"):
    if user_input.strip():
        with st.spinner("Fetching the latest information..."):
            try:
                # Get response from the agent
                response = agent.run(user_input)
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")

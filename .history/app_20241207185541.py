import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
import subprocess
import requests

# Function to run the local Ollama model
def run_ollama_model(query):
    result = subprocess.run(
        ["ollama", "run", "dolphin-llama3:latest", query],
        capture_output=True,
        text=True
    )
    return result.stdout

# Custom function to perform web search
def web_search(query):
    search_url = f"https://www.google.com/search?q={query}"
    response = requests.get(search_url)
    return response.text

# Function to initialize the LangChain agent
def create_agent():
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
        llm=None,  # No LLM needed as we are using the local model
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
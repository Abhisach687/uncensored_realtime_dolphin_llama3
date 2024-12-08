import streamlit as st
from langchain.agents import create_react_agent, Tool
import requests

# Function to run the local Ollama model
def run_ollama_model(query: str) -> str:
    llm = LocalOllamaModel(model_name="dolphin-llama3:latest")
    return llm.predict(query)

# Custom function to perform web search
def web_search(query: str) -> str:
    search_url = f"https://www.google.com/search?q={query}"
    response = requests.get(search_url)
    return response.text

# Function to initialize the LangChain agent
def create_agent():
    # Initialize the local Ollama model
    llm = LocalOllamaModel(model_name="dolphin-llama3:latest")

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

    # Initialize the agent using the new method
    agent = create_react_agent(
        llm=llm,
        tools=tools,
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
                # Get response from the agent using the invoke method
                response = agent.invoke(user_input)
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")

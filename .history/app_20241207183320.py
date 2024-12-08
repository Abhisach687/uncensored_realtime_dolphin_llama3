import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import Ollama
from langchain.tools import WebBrowser

# Function to initialize the LangChain agent
def create_agent():
    # Initialize the Ollama Dolphin model
    llm = Ollama.from_model("dolphin")

    # Initialize the browser tool for web search
    browser_tool = WebBrowser()

    # Define tools
    tools = [
        Tool(
            name="Browser",
            func=browser_tool.run,
            description="Tool to browse the internet and retrieve information."
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
                st.success("Response fetched successfully!")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a valid query.")

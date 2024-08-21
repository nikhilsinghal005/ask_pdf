import os
import requests
from dotenv import load_dotenv
import os
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.agents import tool, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from utils.embeddings_utils import EmbeddingsUtils

load_dotenv()

class CustomQueryAgent:
    def __init__(self, index, text_chunks):
        self.index = index
        self.text_chunks = text_chunks
        self.embeddings_utils = EmbeddingsUtils()
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_KEY_PROJECT"))

    def search_index(self, query, top_k=5):
        query_embedding = self.embeddings_utils.create_embeddings([query])
        query_embedding_np = np.array(query_embedding).astype('float32')
        D, I = self.index.search(query_embedding_np, top_k)
        return [self.text_chunks[i] for i in I[0]]

    def create_agent(self):
        # Define the search tool using the @tool decorator
        @tool
        def search_tool(query: str) -> str:
            """Searches and returns relevant text chunks for a given query."""
            return self.search_index(query)

        # Define the Slack posting tool using the @tool decorator
        @tool
        def post_to_slack(message: str, channel: str = "#social") -> str:
            """Posts a message to a specified Slack channel."""
            print("Post to slack")
            slack_token = os.getenv("SLACK_BOT_TOKEN")
            if not slack_token:
                return "Error: Slack API token not found."

            url = "https://slack.com/api/chat.postMessage"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {slack_token}",
            }
            data = {
                "channel": "testchannel",
                "text": "Hi",
            }
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                return f"Message posted to {channel}"
            else:
                return f"Failed to post message: {response.text}"

        tools = [search_tool, post_to_slack]

        # Create the prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a powerful assistant with access to a custom search tool."),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Bind tools to the LLM
        llm_with_tools = self.llm.bind_tools(tools)

        # Create the agent
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )

        # Initialize the AgentExecutor
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        return agent_executor

    def query(self, user_input):
        # Create the agent
        agent_executor = self.create_agent()

        # Use the `invoke` method instead of `run`
        response = agent_executor.invoke({"input": user_input})

        return response

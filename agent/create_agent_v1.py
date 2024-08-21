import os
import json
import requests
import numpy as np
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from utils.embeddings_utils import EmbeddingsUtils

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

    def post_to_slack(self, message: str, channel: str = "#testchannel") -> str:
        slack_token = os.getenv("SLACK_BOT_TOKEN")
        if not slack_token:
            return "Error: Slack API token not found."

        url = "https://slack.com/api/chat.postMessage"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {slack_token}",
        }
        data = {
            "channel": channel,
            "text": message,
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return f"Message posted to {channel}"
        else:
            return f"Failed to post message: {response.text}"

    def create_agent(self):
        search_tool = {
            "name": "SearchTool",
            "func": self.search_index,
            "description": "Searches and returns relevant text chunks for a given query."
        }

        slack_tool = {
            "name": "SlackTool",
            "func": self.post_to_slack,
            "description": "Posts a message to a specified Slack channel."
        }

        prompt_template = """
        You are an AI assistant tasked with answering questions and posting results to Slack based on a given dataset.
        Use the following tools to help you:

        SearchTool: {search_tool_description}
        SlackTool: {slack_tool_description}

        User Query: {input}

        **Search Completed:** {search_completed}

        Follow these steps carefully:
        1) If `Search Completed` is False, start by using the SearchTool to gather relevant information.
        2) If `Search Completed` is True, proceed directly to using the SlackTool to post the search results.
        3) After posting to Slack, you may provide the final answer to the user.

        Your response should be in the following JSON format:
        {{
            "thoughts": "Your step-by-step reasoning",
            "action": "SearchTool, SlackTool, or FinalAnswer",
            "action_input": "The query for the tool or the final answer"
        }}

        Begin your response:
        """



        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input", "search_tool_description", "slack_tool_description"]
        )

        return prompt, search_tool, slack_tool

    def run_agent(self, user_query):
        prompt, search_tool, slack_tool = self.create_agent()

        search_completed = False
        max_iterations = 10
        for i in range(max_iterations):
            formatted_prompt = prompt.format(
                input=user_query,
                search_tool_description=search_tool["description"],
                slack_tool_description=slack_tool["description"],
                search_completed=search_completed
            )

            response = self.llm.predict(formatted_prompt)

            try:
                parsed_response = json.loads(response)
                print(parsed_response)

                if parsed_response["action"] == "SearchTool":
                    print(parsed_response["action"])
                    search_results = search_tool["func"](parsed_response["action_input"])
                    user_query += f"\n\nSearch results: {search_results}"
                    search_completed = True  # Set search_completed to True after search is done
                elif parsed_response["action"] == "SlackTool":
                    print(parsed_response["action"])
                    slack_message = parsed_response["action_input"]
                    post_result = slack_tool["func"](slack_message)
                    return post_result
                elif parsed_response["action"] == "FinalAnswer":
                    print(parsed_response["action"])
                    return parsed_response["action_input"]
                else:
                    return "Error: Invalid action specified by the agent."
            except json.JSONDecodeError:
                return "Error: Failed to parse the agent's response."

        return "Error: Maximum iterations reached without a final answer."

    def query(self, user_input):
        return self.run_agent(user_input)

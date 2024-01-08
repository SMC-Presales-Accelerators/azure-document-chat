# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import re
import asyncio
import random
import requests
import json
from concurrent.futures import ThreadPoolExecutor
from langchain.chat_models import AzureChatOpenAI
from langchain.utilities import BingSearchAPIWrapper
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import CosmosDBChatMessageHistory
from langchain.agents import ConversationalChatAgent, AgentExecutor, Tool
from typing import Any, Dict, List, Optional, Union
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.schema import AgentAction, AgentFinish, LLMResult

#custom libraries that we will use later in the app
from prompts import WELCOME_MESSAGE, CUSTOM_CHATBOT_PREFIX, CUSTOM_CHATBOT_SUFFIX
from utils import DocSearchAgent, ChatGPTTool, BingSearchAgent, run_agent, model_tokens_limit

from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount, Activity, ActivityTypes

from dotenv import load_dotenv
load_dotenv()

# Env variables needed by langchain
os.environ["OPENAI_API_VERSION"] = os.environ.get("AZURE_OPENAI_API_VERSION")


# Callback hanlder used for the bot service to inform the client of the thought process before the final response
class BotServiceCallbackHandler(BaseCallbackHandler):
    """Callback handler to use in Bot Builder Application"""
    
    def __init__(self, turn_context: TurnContext) -> None:
        self.tc = turn_context

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        asyncio.run(self.tc.send_activity(f"LLM Error: {error}\n"))

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        asyncio.run(self.tc.send_activity(f"Tool: {serialized['name']}"))

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        if "Action Input" in action.log:
            action = action.log.split("Action Input:")[1]
            asyncio.run(self.tc.send_activity(f"\u2611 Searching: {action} ..."))
            asyncio.run(self.tc.send_activity(Activity(type=ActivityTypes.typing)))

            
# Bot Class
class MyBot(ActivityHandler):
    
    def __init__(self):
        self.model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME") 
    
    # Function to show welcome message to new users
    async def on_members_added_activity(self, members_added: ChannelAccount, turn_context: TurnContext):
        for member_added in members_added:
            if member_added.id != turn_context.activity.recipient.id:
                await turn_context.send_activity(WELCOME_MESSAGE)
    
    
    # See https://aka.ms/about-bot-activity-message to learn more about the message and other activity types.
    async def on_message_activity(self, turn_context: TurnContext):
             
        # Extract info from TurnContext - You can change this to whatever , this is just one option 
        session_id = turn_context.activity.conversation.id
        user_id = turn_context.activity.from_property.id + "-" + turn_context.activity.channel_id
        input_text_metadata = dict()
        input_text_metadata["local_timestamp"] = turn_context.activity.local_timestamp.strftime("%I:%M:%S %p, %A, %B %d of %Y")
        input_text_metadata["local_timezone"] = turn_context.activity.local_timezone
        input_text_metadata["locale"] = turn_context.activity.locale

        # Setting the query to send to OpenAI
        input_text = turn_context.activity.text + "\n\n metadata:\n" + str(input_text_metadata)    
            
        # Set Callback Handler
        cb_handler = BotServiceCallbackHandler(turn_context)
        cb_manager = CallbackManager(handlers=[cb_handler])

        # Set LLM 
        llm = AzureChatOpenAI(deployment_name=self.model_name, temperature=0.5, max_tokens=1000, callback_manager=cb_manager)

        # Initialize our Tools/Experts
        text_indexes = ["cogsrch-index-files"]
        doc_search = DocSearchAgent(llm=llm, vector_only_indexes=text_indexes,
                           k=10, similarity_k=4, reranker_th=1,
                           callback_manager=cb_manager, return_direct=True)
        www_search = BingSearchAgent(llm=llm, k=5, callback_manager=cb_manager, return_direct=True)
        chatgpt_search = ChatGPTTool(llm=llm, callback_manager=cb_manager, return_direct=True)


        tools = [www_search, doc_search, chatgpt_search]

        # Set brain Agent with persisten memory in CosmosDB
        cosmos = CosmosDBChatMessageHistory(
                        cosmos_endpoint=os.environ['AZURE_COSMOSDB_ENDPOINT'],
                        cosmos_database=os.environ['AZURE_COSMOSDB_NAME'],
                        cosmos_container=os.environ['AZURE_COSMOSDB_CONTAINER_NAME'],
                        connection_string=os.environ['AZURE_COMOSDB_CONNECTION_STRING'],
                        session_id=session_id,
                        user_id=user_id
                    )
        cosmos.prepare_cosmos()
        # Limit memory tokens to max minus 1500 to allow answer room plus buffer
        history_token_limit = model_tokens_limit(self.model_name) - 1500
        memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=30, chat_memory=cosmos, max_token_limit=history_token_limit)
        agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools,system_message=CUSTOM_CHATBOT_PREFIX,human_message=CUSTOM_CHATBOT_SUFFIX)
        agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory, handle_parsing_errors=True)

        await turn_context.send_activity(Activity(type=ActivityTypes.typing))
        
        # Please note below that running a non-async function like run_agent in a separate thread won't make it truly asynchronous. It allows the function to be called without blocking the event loop, but it may still have synchronous behavior internally.
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(ThreadPoolExecutor(), run_agent, input_text, agent_chain)
        
        await turn_context.send_activity(answer)




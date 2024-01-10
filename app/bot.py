# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import json
import asyncio

from operator import itemgetter

from concurrent.futures import ThreadPoolExecutor
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.chat_message_histories import CosmosDBChatMessageHistory
from typing import Any, Dict, List, Optional, Union
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.docstore.document import Document
from langchain_core.messages import BaseMessage

from langchain.globals import set_debug
set_debug(False)

#custom libraries that we will use later in the app
from utils import AzureAiSearchIndexSchema, BlobStorageProperties, create_service_sas_blob

from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount, Activity, ActivityTypes

from dotenv import load_dotenv
load_dotenv()

# Env variables needed by langchain
os.environ["OPENAI_API_VERSION"] = os.environ.get("AZURE_OPENAI_API_VERSION")

# Env Var to override default fields for Azure Search because langchain doesn't let you set them via API
os.environ["AZURESEARCH_FIELDS_CONTENT"] = "chunk"
os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "chunkVector"

from langchain_community.vectorstores.azuresearch import AzureSearch

EXPERTISE = os.environ["EXPERTISE_DESCRIPTION"]

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
        WELCOME_MESSAGE = """
Hello and welcome! \U0001F44B

I'm a smart virtual assistant designed to help you find answers to your questions!

Feel free to ask me anything about """ + EXPERTISE + """ and I'll do my best to answer it for you.
"""

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

        embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.environ['AZURE_OPENAI_EMBEDDING_DEPLOYMENT'],
            chunk_size=1,
            azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT']
        )

        # We need to define our Azure Search schema for Langchain to understand how to search

        vector_store: AzureSearch = AzureSearch(
            azure_search_endpoint=os.environ['AZURE_SEARCH_ENDPOINT'],
            azure_search_key=os.environ['AZURE_SEARCH_KEY'],
            index_name="cogsrch-index-files",
            embedding_function=embeddings.embed_query,
            fields=AzureAiSearchIndexSchema(),
            semantic_configuration_name="openai-vectordb-semantic-configuration"
        )

        retriever = vector_store.as_retriever()

        model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME")

        llm = AzureChatOpenAI(deployment_name=model_name, temperature=0.5, max_tokens=2000)

        # Set brain Agent with persisten memory in CosmosDB
        cosmos = CosmosDBChatMessageHistory(
                        cosmos_endpoint=os.environ['AZURE_COSMOSDB_ENDPOINT'],
                        cosmos_database=os.environ['AZURE_COSMOSDB_NAME'],
                        cosmos_container=os.environ['AZURE_COSMOSDB_CONTAINER_NAME'],
                        connection_string=os.environ['AZURE_COMOSDB_CONNECTION_STRING'],
                        session_id=session_id,
                        user_id=user_id,
                        ttl=60*60*24*7, # Store history for 1 week
                    )
        cosmos.prepare_cosmos()

        prompt_topic = EXPERTISE

        latent_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You're an assistant who's good at " + prompt_topic + " and based on the question and conversation history provided, you will provide 10 reworded questions that help best answer the original question."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        def get_session_history(passed_session_id: str) -> List[BaseMessage]:
            print(passed_session_id)
            return cosmos.messages
    

        latent_chain = (
            {"question": itemgetter("question"), "history": RunnableLambda(get_session_history)}   
            | latent_prompt
            | llm
            | StrOutputParser()
        )

        latent_condensed_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You're an assistant who's good at " + prompt_topic + " and based on the original question, new questions provided, and conversation history, provide the best possible worded question to get the original questions answer."),
                ("human", """
                 #### history: 

                 {history}


                 ## original question: 
                 
                 {question}
                 

                 ### new questions:

                 {questions}"""),
            ]
        )

        latent_condensed_chain = (
            {"question": itemgetter("question"), "questions": itemgetter("new_questions"), "history": RunnableLambda(get_session_history)}
            | latent_condensed_prompt
            | llm
            | StrOutputParser()
        )

        template = """You're an assistant who's good at """ + prompt_topic + """. Answer the question and provide more details based only on the following context sources are provided in the location metadata field:
        {context}

        Question: {generated_question}

        You will get 40 points if you include the source of your answer with it's full url including parameters in a markdown link
        You will get 30 more points if you use markdown to format your answer in a pleasing manner
        """
        prompt = ChatPromptTemplate.from_template(template)

        blob_storage = BlobStorageProperties()

        def _add_sas_urls(documents: List[Document]) -> List[Document]:
            for doc in documents:
                full_blob_name = doc.metadata['location'].split(blob_storage.container_name+"/")[1]
                sas_key = create_service_sas_blob(full_blob_name, blob_storage)
                doc.metadata["location"] = doc.metadata["location"] + "?" + sas_key
            return documents

        _context = {
            "context": itemgetter("condensed_question") | retriever | _add_sas_urls,
            "generated_question": itemgetter("condensed_question")
        }

        chain = (
            _context
            | prompt
            | llm
            | StrOutputParser()
        )

        def add_history(chat):
            cosmos.add_user_message(chat["question"])
            cosmos.add_ai_message(chat["answer"])
            return chat["answer"]

        history_chain = (
            {"question": itemgetter("question"), "answer": itemgetter("final_answer")}
            | RunnableLambda(add_history)
        )

        seq_chain = (
            RunnablePassthrough.assign(new_questions=latent_chain) 
            | RunnablePassthrough.assign(condensed_question=latent_condensed_chain)
            | RunnablePassthrough.assign(final_answer=chain) 
            | history_chain
        )

        await turn_context.send_activity(Activity(type=ActivityTypes.typing))
        
        # Please note below that running a non-async function like run_agent in a separate thread won't make it truly asynchronous. It allows the function to be called without blocking the event loop, but it may still have synchronous behavior internally.
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(ThreadPoolExecutor(), seq_chain.invoke, {"question": input_text})
        
        await turn_context.send_activity(answer)




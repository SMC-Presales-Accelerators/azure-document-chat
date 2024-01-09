import re
import os
import datetime
from io import BytesIO
from typing import List

import tiktoken
from azure.storage.blob import generate_blob_sas, BlobSasPermissions

from azure.search.documents.indexes.models import (
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
)

from langchain.docstore.document import Document

from dotenv import load_dotenv
load_dotenv()

# Returns the num of tokens used on a string
def num_tokens_from_string(string: str) -> int:
    encoding_name ='cl100k_base'
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Returning the token limit based on model selection
def model_tokens_limit(model: str) -> int:
    """Returns the number of tokens limits in a text model."""
    if model == "gpt-35-turbo":
        token_limit = 4096
    elif model == "gpt-4":
        token_limit = 8192
    elif model == "gpt-35-turbo-16k":
        token_limit = 16384
    elif model == "gpt-4-32k":
        token_limit = 32768
    else:
        token_limit = 4096
    return token_limit

# Returns num of toknes used on a list of Documents objects
def num_tokens_from_docs(docs: List[Document]) -> int:
    num_tokens = 0
    for i in range(len(docs)):
        num_tokens += num_tokens_from_string(docs[i].page_content)
    return num_tokens

class BlobStorageProperties:
    connection_string = os.environ['BLOB_CHAT_CONNECTION_STRING']

    def __init__(self):

        self.account_name = re.search('AccountName=([^;]*);', self.connection_string).group(1)
        self.account_key = re.search('AccountKey=([^;]*);', self.connection_string).group(1)
        self.container_name = os.environ['BLOB_CHAT_CONTAINER_NAME']

def create_service_sas_blob(blob_name, blob_properties: BlobStorageProperties = BlobStorageProperties()):
    # Create a SAS token that's valid for one day, as an example
    start_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=10)
    expiry_time = start_time + datetime.timedelta(days=1)
    sas_token = generate_blob_sas(
        account_name=blob_properties.account_name,
        container_name=blob_properties.container_name,
        blob_name=blob_name,
        account_key=blob_properties.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=expiry_time,
        start=start_time
    )

    return sas_token

def AzureAiSearchIndexSchema():
    fields = [
        SearchableField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
            analyzer="keyword"
        ),
        SearchableField(
            name="parent_id",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SearchableField(
            name="chunk",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        SearchField(
            name="chunkVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536
        ),
        SearchableField(
            name="name",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        # Additional field to store the title
        SearchableField(
            name="title",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        # Additional field for filtering on document source
        SimpleField(
            name="location",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
    ]

    return fields
 
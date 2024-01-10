import os

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

from dotenv import load_dotenv
load_dotenv()

class AzureSearchApi:
    def __init__(self):
        self.endpoint = os.environ['AZURE_SEARCH_ENDPOINT']
        self.key = os.environ['AZURE_SEARCH_KEY']
        self.index_name = "cogsrch-index-files"
        self.client = SearchClient(endpoint=self.endpoint,
                                   index_name=self.index_name,
                                   credential=AzureKeyCredential(self.key))

    def get_document_title(self, document_name):
        results = self.client.search(search_text=f"name:{document_name}", top=1, select="title")
        title = ""
        for result in results:
            title = result["title"]
        return {"title": title}
import os
import json
import requests

from dotenv import load_dotenv
load_dotenv()

from create_datasource import create_datasource
from create_skillset import create_skillset
from create_index import create_index
from create_indexer import create_indexer

datasource_name = "cogsrch-datasource-files"
skillset_name = "cogsrch-skillset-files"
index_name = "cogsrch-index-files"
indexer_name = "cogsrch-indexer-files"

AZURE_SEARCH_ENDPOINT = os.environ['AZURE_SEARCH_ENDPOINT']
AZURE_SEARCH_MANAGEMENT_KEY = os.environ['AZURE_SEARCH_MANAGEMENT_KEY']
BLOB_CONNECTION_STRING = os.environ['BLOB_CONNECTION_STRING']
BLOB_CONTAINER_NAME = os.environ['BLOB_CONTAINER_NAME']
AZURE_OPENAI_ENDPOINT = os.environ['AZURE_OPENAI_ENDPOINT']
AZURE_OPENAI_API_KEY = os.environ['AZURE_OPENAI_API_KEY']
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ['AZURE_OPENAI_EMBEDDING_DEPLOYMENT']

def exists(name, type, api_version = "2023-10-01-Preview"):
    headers = {'Content-Type': 'application/json','api-key': AZURE_SEARCH_MANAGEMENT_KEY}
    params = {'api-version': api_version}
    r = requests.get("%s/%s/%s" % (AZURE_SEARCH_ENDPOINT, type, name), headers=headers, params=params)
    if(r.ok):
        return True
    elif(r.status_code == 404):
        return False
    else:
        r.raise_for_status()

def create_cogsearch_environment():
    if(not exists(datasource_name, "datasources")):
        create_datasource(datasource_name, AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_MANAGEMENT_KEY, BLOB_CONNECTION_STRING, BLOB_CONTAINER_NAME)
    else:
        print("Data Source %s already exists, moving on." % (datasource_name))

    if(not exists(index_name, "indexes")):
        create_index(index_name, AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_MANAGEMENT_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_EMBEDDING_DEPLOYMENT, AZURE_OPENAI_API_KEY)
    else:
        print("Index %s already exists, moving on." % (index_name))

    if(not exists(skillset_name, "skillsets")):
        create_skillset(skillset_name, AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_MANAGEMENT_KEY, index_name, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_EMBEDDING_DEPLOYMENT, AZURE_OPENAI_API_KEY)
    else:
        print("Skillset %s already exists, moving on." % (skillset_name))

    if(not exists(indexer_name, "indexers")):
        create_indexer(indexer_name, AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_MANAGEMENT_KEY, datasource_name, index_name, skillset_name)
    else:
        print("Indexer %s already exists, moving on." % (indexer_name))


if __name__ == "__main__":
    try:
        create_cogsearch_environment()
    except Exception as error:
        raise error
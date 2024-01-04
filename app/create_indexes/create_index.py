import json
import requests

index_payload = {
    "name": "cogsrch-index-files",
    "fields": [
        {"name": "id", "type": "Edm.String", "searchable": "true", "filterable": "true", "retrievable": "true", "sortable": "true", "facetable": "true", "key": "true", "analyzer": "keyword"},
        {"name": "parent_id", "type": "Edm.String", "searchable": "true", "filterable": "true", "retrievable": "true", "sortable": "true", "facetable": "true", "key": "false"},
        {"name": "title","type": "Edm.String","searchable": "true","retrievable": "true"},
        {"name": "chunk","type": "Edm.String","searchable": "true","retrievable": "true"},
        {"name": "chunkVector","type": "Collection(Edm.Single)","searchable": "true","retrievable": "true","dimensions": 1536,"vectorSearchProfile": "openai-vectordb-profile"},
        {"name": "name", "type": "Edm.String", "searchable": "true", "retrievable": "true", "sortable": "false", "filterable": "false", "facetable": "false"},
        {"name": "location", "type": "Edm.String", "searchable": "false", "retrievable": "true", "sortable": "false", "filterable": "false", "facetable": "false"},

    ],
    "semantic": {
    "defaultConfiguration": "openai-vectordb-semantic-configuration",
    "configurations": [
      {
        "name": "openai-vectordb-semantic-configuration",
        "prioritizedFields": {
          "titleField": {
            "fieldName": "title"
          },
          "prioritizedContentFields": [
            {
              "fieldName": "chunk"
            }
          ],
          "prioritizedKeywordsFields": []
        }
      }
    ]
  },
  "vectorSearch": {
    "algorithms": [
      {
        "name": "openai-vectordb-algorithm",
        "kind": "hnsw",
        "hnswParameters": {
          "metric": "cosine",
          "m": 4,
          "efConstruction": 400,
          "efSearch": 500
        }
      }
    ],
    "profiles": [
      {
        "name": "openai-vectordb-profile",
        "algorithm": "openai-vectordb-algorithm",
        "vectorizer": "openai-vectordb-vectorizer"
      }
    ],
    "vectorizers": [
      {
        "name": "openai-vectordb-vectorizer",
        "kind": "azureOpenAI",
        "azureOpenAIParameters": {
          "resourceUri": "",
          "deploymentId": "embedding",
          "apiKey": ""
        }
      }
    ]
  }
}

def create_index(index_name, azure_search_endpoint, api_key, openai_endpoint, openai_embedding_deployment_name, openai_api_key, api_version = "2023-10-01-Preview"):
    print("Creating Index for Azure AI Search...")
    headers = {'Content-Type': 'application/json','api-key': api_key}
    params = {'api-version': api_version}

    index_payload["name"] = index_name
    index_payload["vectorSearch"]["vectorizers"][0]["azureOpenAIParameters"]["resourceUri"] = openai_endpoint
    index_payload["vectorSearch"]["vectorizers"][0]["azureOpenAIParameters"]["deploymentId"] = openai_embedding_deployment_name
    index_payload["vectorSearch"]["vectorizers"][0]["azureOpenAIParameters"]["apiKey"] = openai_api_key

    r = requests.put(azure_search_endpoint + "/indexes/" + index_name,
                 data=json.dumps(index_payload), headers=headers, params=params)
    if(r.ok):
        print("Index %s created" % (index_name))
    else:
        print("Index creation has failed with error:")
        print(r.json().get('error').get('message'))
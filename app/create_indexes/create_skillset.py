import json
import requests

skillset_payload = {
  "name": "openai-vectordb-skillset",
  "description": "Skillset to chunk documents and generate embeddings",
  "skills": [
    {
      "@odata.type": "#Microsoft.Skills.Text.SplitSkill",
      "name": "#1",
      "description": "Split skill to chunk documents",
      "context": "/document",
      "defaultLanguageCode": "en",
      "textSplitMode": "pages",
      "maximumPageLength": 2000,
      "pageOverlapLength": 500,
      "maximumPagesToTake": 0,
      "inputs": [
        {
          "name": "text",
          "source": "/document/content"
        }
      ],
      "outputs": [
        {
          "name": "textItems",
          "targetName": "pages"
        }
      ]
    },
    {
      "@odata.type": "#Microsoft.Skills.Text.AzureOpenAIEmbeddingSkill",
      "name": "#2",
      "context": "/document/pages/*",
      "resourceUri": "OpenAiEndpoint",
      "apiKey": "OpenAIKey",
      "deploymentId": "embedding",
      "inputs": [
        {
          "name": "text",
          "source": "/document/pages/*"
        }
      ],
      "outputs": [
        {
          "name": "embedding",
          "targetName": "vector"
        }
      ]
    }
  ],
  "indexProjections": {
    "selectors": [
      {
        "targetIndexName": "openai-vectordb",
        "parentKeyFieldName": "parent_id",
        "sourceContext": "/document/pages/*",
        "mappings": [
          {
            "name": "chunk",
            "source": "/document/pages/*",
            "inputs": []
          },
          {
            "name": "chunkVector",
            "source": "/document/pages/*/vector",
            "inputs": []
          },
          {
            "name": "title",
            "source": "/document/metadata_title",
            "inputs": []
          },
          {
            "name": "name",
            "source": "/document/metadata_storage_name",
            "inputs": []
          },
          {
            "name": "location",
            "source": "/document/metadata_storage_path",
            "inputs": []
          }
        ]
      }
    ],
    "parameters": {
      "projectionMode": "skipIndexingParentDocuments"
    }
  }
}

def create_skillset(skillset_name, azure_search_endpoint, api_key, index_name, openai_endpoint, openai_embedding_deployment_name, openai_api_key, api_version = "2023-10-01-Preview"):
    print("Creating Skillset for Azure AI Search...")
    headers = {'Content-Type': 'application/json','api-key': api_key}
    params = {'api-version': api_version}

    skillset_payload["name"] = skillset_name

    skillset_payload["skills"][0]["resourceUri"] = openai_endpoint
    skillset_payload["skills"][0]["deploymentId"] = openai_embedding_deployment_name
    skillset_payload["skills"][0]["apiKey"] = openai_api_key

    skillset_payload["indexProjections"]["selectors"][0]["targetIndexName"] = index_name

    r = requests.put(azure_search_endpoint + "/skillsets/" + skillset_name,
                 data=json.dumps(skillset_payload), headers=headers, params=params)
    if(r.ok):
        print("Skillset %s created using OpenAI Endpoint %s" % (skillset_name, openai_endpoint))
    else:
        print("Skillset creation has failed with error:")
        print(r.json().get('error').get('message'))
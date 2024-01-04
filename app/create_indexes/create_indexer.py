import json
import requests

# Create an indexer
indexer_payload = {
    "name": "cogsrch-indexer-files",
    "dataSourceName": "cogsrch-datasource",
    "targetIndexName": "cogsrch-index-files",
    "skillsetName": "openai-vectordb-skillset",
    "schedule" : { "interval" : "PT2H"}, # How often do you want to check for new content in the data source
    "fieldMappings": [
        {
          "sourceFieldName" : "metadata_title",
          "targetFieldName" : "title"
        },
        {
          "sourceFieldName" : "metadata_storage_name",
          "targetFieldName" : "name"
        },
        {
          "sourceFieldName" : "metadata_storage_path",
          "targetFieldName" : "location"
        }
    ],
    "outputFieldMappings": [],
    "parameters":
    {
        "maxFailedItems": -1,
        "maxFailedItemsPerBatch": -1,
        "configuration":
        {
            "dataToExtract": "contentAndMetadata",
            "imageAction": "generateNormalizedImages"
        }
    }
}

def create_indexer(indexer_name, azure_search_endpoint, api_key, datasource_name, index_name, skillset_name, api_version = "2023-10-01-Preview"):
    print("Creating Indexer for Azure AI Search...")
    headers = {'Content-Type': 'application/json','api-key': api_key}
    params = {'api-version': api_version}

    indexer_payload["name"] = indexer_name
    indexer_payload["dataSourceName"] = datasource_name
    indexer_payload["targetIndexName"] = index_name
    indexer_payload["skillsetName"] = skillset_name

    r = requests.put(azure_search_endpoint + "/indexers/" + indexer_name,
                 data=json.dumps(indexer_payload), headers=headers, params=params)
    if(r.ok):
        print("Indexer %s created" % (indexer_name))
    else:
        print("Indexer creation has failed with error:")
        print(r.json().get('error').get('message'))
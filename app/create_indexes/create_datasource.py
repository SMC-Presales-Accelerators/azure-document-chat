import json
import requests

datasource_payload = {
  "name": "vectordb-datasource",
  "type": "azureblob",
  "credentials": {
    "connectionString": "ConnectionString"
  },
  "container": {
    "name": "documents"
  },
  "dataDeletionDetectionPolicy": {
    "@odata.type": "#Microsoft.Azure.Search.NativeBlobSoftDeleteDeletionDetectionPolicy"
  }
}

def create_datasource(datasource_name, azure_search_endpoint, api_key, azure_blob_connection_string, azure_blob_container, api_version = "2023-10-01-Preview"):
    print("Creating Data Source for Azure AI Search...")
    headers = {'Content-Type': 'application/json','api-key': api_key}
    params = {'api-version': api_version}

    datasource_payload["name"] = datasource_name
    datasource_payload["credentials"]["connectionString"] = azure_blob_connection_string
    datasource_payload["container"]["name"] = azure_blob_container

    r = requests.put(azure_search_endpoint + "/datasources/" + datasource_name,
                 data=json.dumps(datasource_payload), headers=headers, params=params)
    if(r.ok):
        print("Data Source %s created with container %s" % (datasource_name, azure_blob_container))
    else:
        print("Data Source creation has failed with error:")
        print(r.json().get('error').get('message'))

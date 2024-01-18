@allowed([
  'new'
  'existing'
])
@description('Do you already have an existing storage account with your documents or would you like to create a new one?')
param UseExistingStorageAccount string = 'new'

@description('Storage account name for your source documents to index and chat with')
param storageAccountName string = uniqueString(website_name, 'storage')

@description('The Container name where your data resides, if you have chose to create at new Storage Account, we will create this container.')
param storageContainerName string

@description('Describe the data that the bot will be discussing')
param expertise string

@description('Name for the app service, this will be the beginning of the FQDN.')
param website_name string

@description('Account name for the OpenAI deployment')
param openai_account_name string = uniqueString(website_name, 'openai')

@description('OpenAI Deployment name for the model deployment.')
param openai_deployment_name string = 'gpt3516k'

@description('Required. Active Directory App ID.')
param appId string

@description('Required. Active Directory App Secret Value.')
@secure()
param appPassword string

@description('The resource group your current blob storage account containing your documents is in.')
param resourceGroupSearch string = resourceGroup().name

@description('Optional. The Azure Search Service Name.')
param azureSearchName string = uniqueString(website_name, 'search')

@description('Optional. The API version for the Azure Search service.')
param azureSearchAPIVersion string = '2023-10-01-Preview'

@description('Optional. The model name for the Azure OpenAI service.')
param azureOpenAIModelName string = 'gpt-35-turbo-16k'

param azureOpenAIModelVersion string = '0613'

@description('Optional. The API version for the Azure OpenAI service.')
param azureOpenAIAPIVersion string = '2023-05-15'

@description('Optional. The name of the Azure CosmosDB.')
param cosmosDBAccountName string = 'cosmos-${uniqueString(resourceGroup().id)}'

@description('The name for the SQL API database')
param cosmosDBDatabaseName string = 'openai'

@description('Required. The name of the Azure CosmosDB container.')
param cosmosDBContainerName string = 'logs'

@description('Optional. The globally unique and immutable bot ID. Also used to configure the displayName of the bot, which is mutable.')
param botId string = 'BotId-${uniqueString(resourceGroup().id)}'

@description('Optional, defaults to S1. The pricing tier of the Bot Service Registration. Acceptable values are F0 and S1.')
@allowed([
  'F0'
  'S1'
])
param botSKU string = 'S1'

@description('Optional. The name of the new App Service Plan.')
param appServicePlanName string = 'AppServicePlan-Backend-${uniqueString(resourceGroup().id)}'

@description('Optional, defaults to resource group location. The location of the resources.')
param location string = resourceGroup().location

var siteHost = '${website_name}.azurewebsites.net'
var botEndpoint = 'https://${siteHost}/api/messages'

resource saNew 'Microsoft.Storage/storageAccounts@2022-09-01' = if (UseExistingStorageAccount == 'new') {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  resource blob_service 'blobServices@2022-09-01' = {
    name: 'default'
    resource container 'containers@2022-09-01' = {
      name: storageContainerName
    }
  }
}

resource saExisting 'Microsoft.Storage/storageAccounts@2022-09-01' existing = if (UseExistingStorageAccount == 'existing') {
  name: storageAccountName
  scope: resourceGroup(resourceGroupSearch)
}

var storageAccountKey = ((UseExistingStorageAccount == 'new') ? saNew.listKeys().keys[0].value : saExisting.listKeys().keys[0].value)

resource search 'Microsoft.Search/searchServices@2020-08-01' = {
  name: azureSearchName
  location: location
  sku: {
    name: 'standard'
  }
  properties: {
    replicaCount: 1
    partitionCount: 1
    hostingMode: 'default'
  }
}

resource openai_account 'Microsoft.CognitiveServices/accounts@2023-10-01-preview' = {
  kind: 'OpenAI'
  location: location
  name: openai_account_name
  properties: {
    customSubDomainName: openai_account_name
    networkAcls: {
      defaultAction: 'Allow'
      ipRules: []
      virtualNetworkRules: []
    }
    publicNetworkAccess: 'Enabled'
  }
  sku: {
    name: 'S0'
  }
}

resource openaiChatDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-10-01-preview' = {
  parent: openai_account
  name: openai_deployment_name
  properties: {
    model: {
      format: 'OpenAI'
      name: azureOpenAIModelName
      version: azureOpenAIModelVersion
    }
  }
  sku: {
    capacity: 120
    name: 'Standard'
  }
}

resource openaiEmbeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-10-01-preview' = {
  parent: openai_account
  name: 'embedding'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'text-embedding-ada-002'
      version: '2'
    }
  }
  sku: {
    capacity: 120
    name: 'Standard'
  }
}

resource cosmosDbAccount 'Microsoft.DocumentDB/databaseAccounts@2022-05-15' = {
  name: toLower(cosmosDBAccountName)
  location: location
  properties: {
    databaseAccountOfferType: 'Standard'
    consistencyPolicy: {
      defaultConsistencyLevel: 'Session'
    }
    locations: [
      {
        locationName: location
      }
    ]
  }
}

resource database 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2022-05-15' = {
  parent: cosmosDbAccount
  name: cosmosDBDatabaseName
  properties: {
    resource: {
      id: cosmosDBDatabaseName
    }
  }
  resource throughput 'throughputSettings@2023-11-15' = {
    name: 'default'
    properties: {
      resource: {
        throughput: 100
        autoscaleSettings: {
          maxThroughput: 1000
        }
      }
    }
  }
}

resource container 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2022-05-15' = {
  parent: database
  name: cosmosDBContainerName
  properties: {
    resource: {
      id: cosmosDBContainerName
      partitionKey: {
        paths: [
          '/id'
        ]
        kind: 'Hash'
      }
      indexingPolicy: {
        indexingMode: 'consistent'
        includedPaths: [
          {
            path: '/*'
          }
        ]
        excludedPaths: [
          {
            path: '/_etag/?'
          }
        ]
      }
    }
  }
}

resource bot 'Microsoft.BotService/botServices@2022-09-15' = {
  name: botId
  location: 'global'
  kind: 'azurebot'
  sku: {
    name: botSKU
  }
  properties: {
    displayName: botId
    iconUrl: 'https://docs.botframework.com/static/devportal/client/images/bot-framework-default.png'
    endpoint: botEndpoint
    msaAppId: appId
    luisAppIds: []
    schemaTransformationVersion: '1.3'
    isCmekEnabled: false
  }
}

resource directLineChannel 'Microsoft.BotService/botServices/channels@2022-09-15' = {
  parent: bot
  name: 'DirectLineChannel'
  properties: {
    channelName: 'DirectLineChannel'
    properties: {
      sites: [
        {
          siteName: 'Default Site'
          isEnabled: true
          isV1Enabled: true
          isV3Enabled: true
          isSecureSiteEnabled: false
          isBlockUserUploadEnabled: true
        }
      ]
    }
  }
}

// Create a new Linux App Service Plan if no existing App Service Plan name was passed in.
resource appServicePlan 'Microsoft.Web/serverfarms@2022-09-01' = {
  name: appServicePlanName
  location: location
  sku: {
    name: 'P0v3'
  }
  kind: 'linux'
  properties: {
    reserved: true
  }
}

// Create a Web App using a Linux App Service Plan.
resource webApp 'Microsoft.Web/sites@2022-09-01' = {
  name: website_name
  location: location
  properties: {
    serverFarmId: appServicePlan.id
    siteConfig: {
      linuxFxVersion: 'DOCKER|docker.io/smcpresalesaccelerators/azure-document-chat:latest'
      appSettings: [
        {
          name: 'EXPERTISE_DESCRIPTION'
          value: expertise
        }
        {
          name: 'MICROSOFT_APP_ID'
          value: appId
        }
        {
          name: 'MICROSOFT_APP_PASSWORD'
          value: appPassword
        }
        {
          name: 'BLOB_CHAT_CONNECTION_STRING'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccountName};AccountKey=${storageAccountKey};EndpointSuffix=${environment().suffixes.storage}'
        }
        {
          name: 'BLOB_CHAT_CONTAINER_NAME'
          value: storageContainerName
        }
        {
          name: 'AZURE_SEARCH_ENDPOINT'
          value: 'https://${azureSearchName}.search.windows.net'
        }
        {
          name: 'AZURE_SEARCH_KEY'
          value: search.listAdminKeys().primaryKey
        }
        {
          name: 'AZURE_SEARCH_API_VERSION'
          value: azureSearchAPIVersion
        }
        {
          name: 'AZURE_OPENAI_ENDPOINT'
          value: openai_account.properties.endpoint
        }
        {
          name: 'AZURE_OPENAI_API_KEY'
          value: openai_account.listKeys().key1
        }
        {
          name: 'AZURE_OPENAI_MODEL_NAME'
          value: azureOpenAIModelName
        }
        {
          name: 'AZURE_OPENAI_CHATGPT_MODEL'
          value: azureOpenAIModelName
        }
        {
          name: 'AZURE_OPENAI_CHATGPT_DEPLOYMENT'
          value: openai_deployment_name
        }
        {
          name: 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT'
          value: 'embedding'
        }
        {
          name: 'AZURE_OPENAI_API_VERSION'
          value: azureOpenAIAPIVersion
        }
        {
          name: 'AZURE_COSMOSDB_ENDPOINT'
          value: 'https://${cosmosDBAccountName}.documents.azure.com:443/'
        }
        {
          name: 'AZURE_COSMOSDB_NAME'
          value: cosmosDBDatabaseName
        }
        {
          name: 'AZURE_COSMOSDB_CONTAINER_NAME'
          value: cosmosDBContainerName
        }
        {
          name: 'AZURE_COMOSDB_CONNECTION_STRING'
          value: cosmosDbAccount.listConnectionStrings().connectionStrings[0].connectionString
        }
        {
          name: 'BOT_DIRECTLINE_SECRET_KEY'
          value: directLineChannel.listChannelWithKeys().properties.sites[0].key
        }
      ]
    }
  }
}

output botServiceName string = bot.name
output webAppName string = webApp.name
output webAppUrl string = webApp.properties.defaultHostName

failure_detected=0
declare -a required_vars=("AZURE_COSMOSDB_ENDPOINT" "AZURE_COMOSDB_CONNECTION_STRING" "AZURE_COSMOSDB_CONTAINER_NAME" "AZURE_COSMOSDB_NAME" "AZURE_OPENAI_API_KEY" "AZURE_OPENAI_API_VERSION" "AZURE_OPENAI_CHATGPT_DEPLOYMENT" "AZURE_OPENAI_CHATGPT_MODEL" "AZURE_OPENAI_EMBEDDING_DEPLOYMENT" "AZURE_OPENAI_ENDPOINT" "AZURE_OPENAI_MODEL_NAME" "AZURE_SEARCH_API_VERSION" "AZURE_SEARCH_ENDPOINT" "AZURE_SEARCH_KEY" "BLOB_CHAT_CONNECTION_STRING" "BLOB_CHAT_CONTAINER_NAME" "BOT_DIRECTLINE_SECRET_KEY" "EXPERTISE_DESCRIPTION" "MICROSOFT_APP_ID" "MICROSOFT_APP_PASSWORD")
for i in "${required_vars[@]}"
do
    if ! [ -v $i ]; then
        echo "Missing environment variable $i"
        failure_detected=1
    fi
done

if [[ $failure_detected -eq 1 ]]; then
    echo "Missing required settings, Please set values for the items above and try again."
    exit 1
fi

python ./create_indexes/create_cogsearch_env.py &
gunicorn --bind 0.0.0.0:3978 --worker-class aiohttp.worker.GunicornWebWorker --timeout 600 app:APP
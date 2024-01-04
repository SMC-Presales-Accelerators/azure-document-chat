python ./create_indexes/create_cogsearch_env.py
gunicorn --bind 0.0.0.0:3978 --worker-class aiohttp.worker.GunicornWebWorker --timeout 600 app:APP
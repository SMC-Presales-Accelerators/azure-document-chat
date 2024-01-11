FROM python:3.11 AS backend
WORKDIR /app
COPY app .
RUN pip install --no-cache-dir -r requirements.txt
RUN chmod +x run.sh
EXPOSE 3978
CMD ["bash", "-c", "./run.sh"]
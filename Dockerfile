FROM python:3.12-alpine AS backend
WORKDIR /app
COPY app .
RUN apk add --no-cache bash
RUN pip install --no-cache-dir -r requirements.txt
RUN chmod +x run.sh
EXPOSE 3978
CMD ["bash", "-c", "./run.sh"]
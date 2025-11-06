FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV CHROMA_DIR=/app/chroma_db
EXPOSE 8000
CMD ["python","main.py"]

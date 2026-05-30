# Dockerfile for Credit Risk Model API
FROM python:3.14-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create directory for MLflow models
RUN mkdir -p mlruns

# Expose the port the app runs on
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

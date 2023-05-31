FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

RUN apt-get update && apt-get install git

# RUN git clone https://huggingface.co/Dorost/resume

# Set working directory
WORKDIR .

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --default-timeout=100 -r requirements.txt

# Copy entire project
COPY . .

# Expose port 8000
EXPOSE 8000

# Start the server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]


# THIS DOCKERFILE IS FOR "FASTAPI"

# Use an official lightweight Python image.
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies
COPY requirements_fastAPI.txt .
RUN pip install -r requirements_fastAPI.txt

# Copy the FastAPI app files to the container
COPY ./app.py /app/
COPY ./config.json /app/
COPY utils/ /app/utils

# Expose the port the app runs on 
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

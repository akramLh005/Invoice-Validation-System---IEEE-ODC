# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy the main application files
COPY ./main.py /app/main.py
COPY ./config.json /app/config.json
COPY utils/ /app/utils/
COPY middleware/ /app/middleware/
COPY classifier/ /app/classifier/

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

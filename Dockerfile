# Use a base image with Python installed. I used the latest version from pytorch/pytorch
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

# Set the working directory in the container
WORKDIR /workspace

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies in the Docker container
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Set the default command to run your application
CMD ["python", "your_application.py"]
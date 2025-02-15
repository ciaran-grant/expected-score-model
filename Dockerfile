# Import Python Image
FROM python:3.11-slim

# Update package list
RUN apt-get update
# Install git
RUN apt-get install -y git

# Copy files in
WORKDIR /app
COPY app.py /app/app.py
COPY requirements.txt /app/requirements.txt

# Install package
RUN pip install git+https://github.com/ciaran-grant/expected-score-model
RUN pip install -r requirements.txt

# Run app
CMD ["python", "app.py"] 
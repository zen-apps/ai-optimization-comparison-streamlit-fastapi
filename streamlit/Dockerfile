FROM python:3.9-slim

WORKDIR /app

# Copy and install requirements from root directory
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the streamlit directory content
COPY streamlit ./streamlit

# Set environment variables
ENV PORT 8080

# Command to run the application with correct path
CMD streamlit run --server.port=$PORT --server.address=0.0.0.0 streamlit/🏠_home.py
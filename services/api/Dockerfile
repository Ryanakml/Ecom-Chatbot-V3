# Stage 1: Build stage with dependencies
FROM python:3.10-slim as builder

WORKDIR /app

# Install poetry for dependency management (optional but good practice)
# RUN pip install poetry

# Copy dependency files and install them
# For now, we use requirements.txt for simplicity
COPY ./requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Stage 2: Final application stage
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy the application code
# This will be created in a later step
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
# We'll create main.py in a future step.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
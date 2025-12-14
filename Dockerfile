FROM python:3.10-slim

# Use separate layer for dependencies to leverage Docker cache
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy application code
COPY . ./

# Expose a generic entrypoint that dispatches to train or evaluation according to Hydra flags
ENTRYPOINT ["python", "main.py"]

# Optional: override CMD via docker run to pass Hydra arguments
CMD []

# Yahoo Finance MCP Server - Apply.build Optimized
# Platform: Apply.build (0.5 CPU, 512MB RAM)
# Image: nameeswarchivatam/yahoo-finance-mcp:latest

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (using pre-built wheels to avoid gcc requirement)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables for resource optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=1
ENV PORT=8080

# Expose the port (Apply.build may use PORT env var)
EXPOSE 8080

# Lightweight health check - no curl dependency
HEALTHCHECK --interval=60s --timeout=5s --start-period=10s --retries=2 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=3)" || exit 1

# Run the server with resource-constrained settings
# - Single Python process to minimize memory usage
# - Optimized for 512MB RAM environment
CMD ["python", "-u", "server.py", "--host", "0.0.0.0", "--port", "8080"]

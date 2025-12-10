#!/bin/bash
set -e  # Exit on any error

echo "Starting Milvus REST Proxy..."
echo "Working directory: $(pwd)"
echo "Python version: $(python3 --version)"

# Check if required files exist
for file in "rest-proxy-multimodal.py" "requirements.txt"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: Required file $file not found!"
        exit 1
    fi
done

# Check if MILVUS_HOST is set
if [ -z "$MILVUS_HOST" ]; then
    echo "ERROR: MILVUS_HOST environment variable is required!"
    echo ""
    echo "Please set MILVUS_HOST in one of the following ways:"
    echo "1. Railway Variables tab: MILVUS_HOST=your-milvus-host"
    echo "2. Railway Start Command: MILVUS_HOST=your-milvus-host bash start.sh"
    echo ""
    echo "Examples:"
    echo "  MILVUS_HOST=localhost"
    echo "  MILVUS_HOST=grpc-reverse-proxy-production-039b.up.railway.app"
    echo "  MILVUS_HOST=in01-xxxxxxxx.aws-us-west-2.vectordb.zilliz.com"
    exit 1
fi

echo "MILVUS_HOST: $MILVUS_HOST"
echo "MILVUS_PORT: ${MILVUS_PORT:-443}"

# Try to import and run the Python script
echo "Starting Python application..."
exec python3 rest-proxy-multimodal.py
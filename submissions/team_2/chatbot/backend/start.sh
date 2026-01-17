#!/bin/bash
# Start MetaKGP RAG System + FastAPI Service

set -e

echo " Starting MetaKGP RAG + FastAPI Server"

# Get the team_2 root directory (5 levels up from start.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEAM_2_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
ENV_FILE="$TEAM_2_ROOT/.env"

# Load environment variables from unified .env at team_2 root
if [ -f "$ENV_FILE" ]; then
    echo " Loading .env from: $ENV_FILE"
    export $(cat "$ENV_FILE" | grep -v '^#' | xargs)
else
    echo " .env file not found at: $ENV_FILE"
    echo " Please create one from: $TEAM_2_ROOT/.env.example"
    exit 1
fi

# Validate required environment variables
# Check if database config is provided (either individual components or full URL)
if [ -z "$DB_HOST" ] && [ -z "$DATABASE_URL" ]; then
    echo " Database configuration not set in .env"
    echo "   Set either DB_HOST, DB_NAME, DB_USER, DB_PASSWORD or DATABASE_URL"
    exit 1
fi

if [ -z "$MODAL_URL" ]; then
    echo " MODAL_URL not set in .env"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo " Virtual environment not found. Please run: uv venv && uv pip install -e ."
    exit 1
fi

# Create necessary directories
mkdir -p cache chroma_data logs

# Get absolute paths for consistency
BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHROMA_ABS_PATH="${BACKEND_DIR}/${CHROMA_DIR:-chroma_data}"
CACHE_ABS_PATH="${BACKEND_DIR}/${CACHE_DIR:-cache}"

echo ""
echo " Starting RAG Indexer..."
python src/services/indexing/indexer.py \
    --chroma-dir "$CHROMA_ABS_PATH" \
    --cache-dir "$CACHE_ABS_PATH" \
    --batch-size "${BATCH_SIZE:-100}" \
    > logs/indexer.log 2>&1 &

INDEXER_PID=$!
echo " Indexer started (PID: $INDEXER_PID)"

# Wait for indexer to initialize
sleep 2

echo ""
echo " Starting FastAPI Server..."
# Set CHROMA_DIR and CACHE_DIR as environment variables for query service
export CHROMA_DIR="$CHROMA_ABS_PATH"
export CACHE_DIR="$CACHE_ABS_PATH"
uvicorn src.app.main:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    > logs/fastapi_server.log 2>&1 &

FastAPI_PID=$!
echo " FastAPI Server started (PID: $FastAPI_PID)"

echo ""
echo " All services started successfully!"
echo ""
echo " Logs:"
echo "   Indexer:       tail -f logs/indexer.log"
echo "   FastAPI Server: tail -f logs/fastapi_server.log"
echo ""
echo " API:"
echo "   Docs:   http://localhost:${PORT:-8000}/docs"
echo ""
echo "Stop: ./stop.sh"

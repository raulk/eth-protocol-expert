#!/bin/bash
# Batch import all Ethereum client codebases
# Usage: ./scripts/ingest_all_clients.sh [--limit N]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Parse optional limit argument
LIMIT_ARG=""
if [ "$1" = "--limit" ] && [ -n "$2" ]; then
    LIMIT_ARG="--limit $2"
    echo "Running with limit: $2 files per repo"
fi

echo "=============================================="
echo "Ethereum Client Codebase Ingestion"
echo "=============================================="
echo ""

# Array of clients to ingest
CLIENTS=("go-ethereum" "prysm" "reth" "lighthouse")

for client in "${CLIENTS[@]}"; do
    echo "----------------------------------------------"
    echo "Ingesting: $client"
    echo "----------------------------------------------"

    if uv run python scripts/ingest_client.py --repo "$client" $LIMIT_ARG; then
        echo "✅ $client ingestion complete"
    else
        echo "❌ $client ingestion failed"
    fi

    echo ""
done

echo "=============================================="
echo "All client ingestion complete!"
echo "=============================================="

# Show final stats
echo ""
echo "Final corpus statistics:"
uv run python -c "
import asyncio
from dotenv import load_dotenv
load_dotenv()
from src.storage.pg_vector_store import PgVectorStore

async def main():
    store = PgVectorStore()
    await store.connect()
    result = await store.pool.fetch('''
        SELECT document_id, COUNT(*) as chunks
        FROM chunks
        WHERE document_id LIKE 'code:%'
        GROUP BY document_id
        ORDER BY chunks DESC
    ''')
    total = sum(r['chunks'] for r in result)
    print(f'Client code chunks: {total:,}')
    for r in result:
        name = r['document_id'].replace('code:', '')
        print(f'  - {name}: {r[\"chunks\"]:,} chunks')
    await store.close()

asyncio.run(main())
"

#!/bin/bash
# usage: ./deploy.sh

echo ">>> 1. Deploying legacy decoder service (GPU)..."
modal deploy services/decoder_legacy.py

echo ">>> 2. Deploying latest decoder service (GPU)..."
modal deploy services/decoder_latest.py

echo ">>> 3. Deploying latest decoder tokenizer service (CPU)..."
modal deploy services/decoder_latest_tokenizer.py

echo ">>> 4. Deploying encoder service (CPU)..."
modal deploy services/encoder_cpu.py

echo ">>> 5. Deploying encoder service (GPU)..."
modal deploy services/encoder_gpu.py

echo ">>> 6. Deploying curator service..."
modal deploy services/curator.py

echo ">>> 7. Deploying collection handler service..."
modal deploy services/collection_handler.py

echo ">>> 8. Deploying storage handler service..."
modal deploy services/storage_handler.py

echo ">>> 9. Deploying LLM judge service..."
modal deploy services/llm_judge.py

echo ">>> 10. Deploying Qdrant server endpoint..."
modal deploy services/qdrant_server.py

echo ">>> 11. Deploying crawler agent (yearly cron)..."
modal deploy services/crawler_agent.py

echo ">>> 12. Deploying email agent (daily cron)..."
modal deploy services/email_agent.py

echo ">>> Done! Full fleet is live."

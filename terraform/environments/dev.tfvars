# =============================================================================
# Development Environment
# =============================================================================

aws_region  = "us-east-1"
environment = "dev"

# Project
project_name = "investment-research-rag"

# Embedding configuration
embedding_provider  = "openai"
embedding_model     = "text-embedding-3-small"
embedding_dimension = 1536

# LLM configuration
llm_provider = "openai"
llm_model    = "gpt-4o"

# Lambda sizing (smaller for dev)
lambda_memory_mb       = 512
lambda_timeout_seconds = 120

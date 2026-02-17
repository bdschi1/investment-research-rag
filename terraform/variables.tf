variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment (dev, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "Environment must be 'dev' or 'prod'."
  }
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "investment-research-rag"
}

variable "embedding_provider" {
  description = "Embedding provider (e.g., openai, bedrock, sentence-transformers)"
  type        = string
  default     = "openai"
}

variable "embedding_model" {
  description = "Embedding model name"
  type        = string
  default     = "text-embedding-3-small"
}

variable "embedding_dimension" {
  description = "Embedding vector dimension"
  type        = number
  default     = 1536
}

variable "llm_provider" {
  description = "LLM provider (e.g., openai, bedrock, anthropic)"
  type        = string
  default     = "openai"
}

variable "llm_model" {
  description = "LLM model name"
  type        = string
  default     = "gpt-4o"
}

variable "lambda_memory_mb" {
  description = "Memory allocated to Lambda functions (MB)"
  type        = number
  default     = 512
}

variable "lambda_timeout_seconds" {
  description = "Lambda function timeout (seconds)"
  type        = number
  default     = 120
}

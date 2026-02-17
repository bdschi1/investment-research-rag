variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
}

variable "environment" {
  description = "Deployment environment (dev, prod)"
  type        = string
}

variable "embedding_dimension" {
  description = "Embedding vector dimension for index configuration"
  type        = number
  default     = 1536
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
}

variable "environment" {
  description = "Deployment environment (dev, prod)"
  type        = string
}

variable "ecr_repository_url" {
  description = "ECR repository URL for container images"
  type        = string
}

variable "execution_role_arn" {
  description = "IAM role ARN for Lambda execution"
  type        = string
}

variable "sqs_queue_arn" {
  description = "SQS queue ARN to trigger this Lambda"
  type        = string
}

variable "memory_size" {
  description = "Lambda memory allocation in MB"
  type        = number
  default     = 512
}

variable "timeout" {
  description = "Lambda timeout in seconds"
  type        = number
  default     = 120
}

variable "environment_variables" {
  description = "Environment variables for the Lambda function"
  type        = map(string)
  default     = {}
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
}

variable "environment" {
  description = "Deployment environment (dev, prod)"
  type        = string
}

variable "query_lambda_arn" {
  description = "ARN of the query Lambda function"
  type        = string
}

variable "query_lambda_invoke_arn" {
  description = "Invoke ARN of the query Lambda function"
  type        = string
}

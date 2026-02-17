variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
}

variable "environment" {
  description = "Deployment environment (dev, prod)"
  type        = string
}

variable "s3_bucket_arn" {
  description = "ARN of the S3 bucket allowed to send messages to this queue"
  type        = string
}

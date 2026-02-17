output "api_endpoint" {
  description = "API Gateway endpoint URL for query requests"
  value       = module.api.api_endpoint
}

output "s3_bucket" {
  description = "S3 bucket name for document uploads"
  value       = module.storage.bucket_id
}

output "opensearch_endpoint" {
  description = "OpenSearch Serverless collection endpoint"
  value       = module.vectordb.collection_endpoint
}

output "ecr_repository" {
  description = "ECR repository URL for Lambda container images"
  value       = module.storage.ecr_repository_url
}

output "ingest_function_name" {
  description = "Ingest Lambda function name"
  value       = module.ingest.function_name
}

output "query_function_name" {
  description = "Query Lambda function name"
  value       = module.query.function_name
}

output "sqs_queue_url" {
  description = "SQS queue URL for document ingestion"
  value       = module.queue.queue_url
}

output "dlq_url" {
  description = "Dead letter queue URL for failed ingestion messages"
  value       = module.queue.dlq_url
}

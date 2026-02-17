output "queue_arn" {
  description = "SQS queue ARN"
  value       = aws_sqs_queue.ingest.arn
}

output "queue_url" {
  description = "SQS queue URL"
  value       = aws_sqs_queue.ingest.url
}

output "queue_name" {
  description = "SQS queue name"
  value       = aws_sqs_queue.ingest.name
}

output "dlq_arn" {
  description = "Dead letter queue ARN"
  value       = aws_sqs_queue.dlq.arn
}

output "dlq_url" {
  description = "Dead letter queue URL"
  value       = aws_sqs_queue.dlq.url
}

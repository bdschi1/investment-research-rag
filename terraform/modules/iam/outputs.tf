output "ingest_role_arn" {
  description = "IAM role ARN for ingest Lambda"
  value       = aws_iam_role.ingest_lambda.arn
}

output "ingest_role_name" {
  description = "IAM role name for ingest Lambda"
  value       = aws_iam_role.ingest_lambda.name
}

output "query_role_arn" {
  description = "IAM role ARN for query Lambda"
  value       = aws_iam_role.query_lambda.arn
}

output "query_role_name" {
  description = "IAM role name for query Lambda"
  value       = aws_iam_role.query_lambda.name
}

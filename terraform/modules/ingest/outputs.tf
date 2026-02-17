output "function_name" {
  description = "Ingest Lambda function name"
  value       = aws_lambda_function.ingest.function_name
}

output "function_arn" {
  description = "Ingest Lambda function ARN"
  value       = aws_lambda_function.ingest.arn
}

output "invoke_arn" {
  description = "Ingest Lambda invoke ARN"
  value       = aws_lambda_function.ingest.invoke_arn
}

output "function_name" {
  description = "Query Lambda function name"
  value       = aws_lambda_function.query.function_name
}

output "function_arn" {
  description = "Query Lambda function ARN"
  value       = aws_lambda_function.query.arn
}

output "invoke_arn" {
  description = "Query Lambda invoke ARN"
  value       = aws_lambda_function.query.invoke_arn
}

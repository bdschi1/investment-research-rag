# -----------------------------------------------------------------------------
# CloudWatch Log Group — Query Lambda
# -----------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "query" {
  name              = "/aws/lambda/${var.project_name}-${var.environment}-query"
  retention_in_days = 30
}

# -----------------------------------------------------------------------------
# Lambda Function — Query (container image)
# -----------------------------------------------------------------------------

resource "aws_lambda_function" "query" {
  function_name = "${var.project_name}-${var.environment}-query"
  role          = var.execution_role_arn
  package_type  = "Image"
  image_uri     = "${var.ecr_repository_url}:query-latest"
  memory_size   = var.memory_size
  timeout       = var.timeout

  image_config {
    command = ["lambda.query_handler.handler"]
  }

  tracing_config {
    mode = "Active"
  }

  environment {
    variables = var.environment_variables
  }

  depends_on = [aws_cloudwatch_log_group.query]

  lifecycle {
    ignore_changes = [image_uri]
  }
}

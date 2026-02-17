# -----------------------------------------------------------------------------
# CloudWatch Log Group — Ingest Lambda
# -----------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "ingest" {
  name              = "/aws/lambda/${var.project_name}-${var.environment}-ingest"
  retention_in_days = 30
}

# -----------------------------------------------------------------------------
# Lambda Function — Ingest (container image)
# -----------------------------------------------------------------------------

resource "aws_lambda_function" "ingest" {
  function_name = "${var.project_name}-${var.environment}-ingest"
  role          = var.execution_role_arn
  package_type  = "Image"
  image_uri     = "${var.ecr_repository_url}:ingest-latest"
  memory_size   = var.memory_size
  timeout       = var.timeout

  image_config {
    command = ["lambda.ingest_handler.handler"]
  }

  tracing_config {
    mode = "Active"
  }

  environment {
    variables = var.environment_variables
  }

  depends_on = [aws_cloudwatch_log_group.ingest]

  lifecycle {
    ignore_changes = [image_uri]
  }
}

# -----------------------------------------------------------------------------
# SQS Event Source Mapping — Trigger ingest Lambda from queue
# -----------------------------------------------------------------------------

resource "aws_lambda_event_source_mapping" "sqs_trigger" {
  event_source_arn                   = var.sqs_queue_arn
  function_name                      = aws_lambda_function.ingest.arn
  batch_size                         = 1
  maximum_batching_window_in_seconds = 0
  enabled                            = true

  function_response_types = ["ReportBatchItemFailures"]
}

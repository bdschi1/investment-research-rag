# -----------------------------------------------------------------------------
# HTTP API Gateway
# -----------------------------------------------------------------------------

resource "aws_apigatewayv2_api" "query" {
  name          = "${var.project_name}-${var.environment}-api"
  protocol_type = "HTTP"

  cors_configuration {
    allow_headers = ["Content-Type", "Authorization", "X-Amz-Date"]
    allow_methods = ["POST", "OPTIONS"]
    allow_origins = ["*"]
    max_age       = 3600
  }
}

# -----------------------------------------------------------------------------
# CloudWatch Log Group — API Gateway access logs
# -----------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/apigateway/${var.project_name}-${var.environment}"
  retention_in_days = 30
}

# -----------------------------------------------------------------------------
# Default Stage with auto-deploy and access logging
# -----------------------------------------------------------------------------

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.query.id
  name        = "$default"
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api.arn
    format = jsonencode({
      requestId        = "$context.requestId"
      ip               = "$context.identity.sourceIp"
      requestTime      = "$context.requestTime"
      httpMethod       = "$context.httpMethod"
      routeKey         = "$context.routeKey"
      status           = "$context.status"
      protocol         = "$context.protocol"
      responseLength   = "$context.responseLength"
      integrationError = "$context.integrationErrorMessage"
    })
  }
}

# -----------------------------------------------------------------------------
# Lambda Integration — AWS_PROXY (payload format 2.0)
# -----------------------------------------------------------------------------

resource "aws_apigatewayv2_integration" "query_lambda" {
  api_id                 = aws_apigatewayv2_api.query.id
  integration_type       = "AWS_PROXY"
  integration_uri        = var.query_lambda_invoke_arn
  integration_method     = "POST"
  payload_format_version = "2.0"
}

# -----------------------------------------------------------------------------
# Route — POST /query
# -----------------------------------------------------------------------------

resource "aws_apigatewayv2_route" "post_query" {
  api_id    = aws_apigatewayv2_api.query.id
  route_key = "POST /query"
  target    = "integrations/${aws_apigatewayv2_integration.query_lambda.id}"
}

# -----------------------------------------------------------------------------
# Lambda Permission — Allow API Gateway to invoke query Lambda
# -----------------------------------------------------------------------------

resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = var.query_lambda_arn
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.query.execution_arn}/*/*"
}

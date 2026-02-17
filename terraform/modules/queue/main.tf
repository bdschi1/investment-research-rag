# -----------------------------------------------------------------------------
# SQS Dead Letter Queue
# -----------------------------------------------------------------------------

resource "aws_sqs_queue" "dlq" {
  name                      = "${var.project_name}-${var.environment}-ingest-dlq"
  message_retention_seconds = 1209600 # 14 days

  tags = {
    Name = "${var.project_name}-${var.environment}-ingest-dlq"
  }
}

# -----------------------------------------------------------------------------
# SQS Queue — Document ingestion
# -----------------------------------------------------------------------------

resource "aws_sqs_queue" "ingest" {
  name                       = "${var.project_name}-${var.environment}-ingest"
  visibility_timeout_seconds = 600
  message_retention_seconds  = 86400 # 1 day
  receive_wait_time_seconds  = 20    # Long polling

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dlq.arn
    maxReceiveCount     = 3
  })

  tags = {
    Name = "${var.project_name}-${var.environment}-ingest"
  }
}

# -----------------------------------------------------------------------------
# SQS Queue Policy — Allow S3 to send messages
# -----------------------------------------------------------------------------

resource "aws_sqs_queue_policy" "allow_s3" {
  queue_url = aws_sqs_queue.ingest.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowS3SendMessage"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
        Action   = "sqs:SendMessage"
        Resource = aws_sqs_queue.ingest.arn
        Condition = {
          ArnEquals = {
            "aws:SourceArn" = var.s3_bucket_arn
          }
        }
      }
    ]
  })
}

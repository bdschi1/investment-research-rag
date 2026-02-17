# -----------------------------------------------------------------------------
# OpenSearch Serverless — Encryption Policy
# -----------------------------------------------------------------------------

resource "aws_opensearchserverless_security_policy" "encryption" {
  name = "${var.project_name}-${var.environment}-enc"
  type = "encryption"

  policy = jsonencode({
    Rules = [
      {
        ResourceType = "collection"
        Resource = [
          "collection/${var.project_name}-${var.environment}"
        ]
      }
    ]
    AWSOwnedKey = true
  })
}

# -----------------------------------------------------------------------------
# OpenSearch Serverless — Network Policy
# AllowFromPublic for demo; restrict in production via VPC endpoint
# -----------------------------------------------------------------------------

resource "aws_opensearchserverless_security_policy" "network" {
  name = "${var.project_name}-${var.environment}-net"
  type = "network"

  policy = jsonencode([
    {
      Description = "Allow public access to collection endpoint"
      Rules = [
        {
          ResourceType = "collection"
          Resource = [
            "collection/${var.project_name}-${var.environment}"
          ]
        },
        {
          ResourceType = "dashboard"
          Resource = [
            "collection/${var.project_name}-${var.environment}"
          ]
        }
      ]
      AllowFromPublic = true
    }
  ])
}

# -----------------------------------------------------------------------------
# OpenSearch Serverless — Collection (VECTORSEARCH)
# -----------------------------------------------------------------------------

resource "aws_opensearchserverless_collection" "vectors" {
  name = "${var.project_name}-${var.environment}"
  type = "VECTORSEARCH"

  depends_on = [
    aws_opensearchserverless_security_policy.encryption,
    aws_opensearchserverless_security_policy.network,
  ]
}

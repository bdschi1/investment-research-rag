# -----------------------------------------------------------------------------
# Data sources
# -----------------------------------------------------------------------------

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  account_id  = data.aws_caller_identity.current.account_id
  region      = data.aws_region.current.name
  name_prefix = "${var.project_name}-${var.environment}"
}

# -----------------------------------------------------------------------------
# Module: Storage (S3 + ECR)
# -----------------------------------------------------------------------------

module "storage" {
  source = "./modules/storage"

  project_name = var.project_name
  environment  = var.environment
}

# -----------------------------------------------------------------------------
# Module: Queue (SQS + DLQ)
# -----------------------------------------------------------------------------

module "queue" {
  source = "./modules/queue"

  project_name  = var.project_name
  environment   = var.environment
  s3_bucket_arn = module.storage.bucket_arn
}

# -----------------------------------------------------------------------------
# Module: VectorDB (OpenSearch Serverless)
# -----------------------------------------------------------------------------

module "vectordb" {
  source = "./modules/vectordb"

  project_name        = var.project_name
  environment         = var.environment
  embedding_dimension = var.embedding_dimension
}

# -----------------------------------------------------------------------------
# Module: IAM (Lambda execution roles)
# -----------------------------------------------------------------------------

module "iam" {
  source = "./modules/iam"

  project_name              = var.project_name
  environment               = var.environment
  s3_bucket_arn             = module.storage.bucket_arn
  sqs_queue_arn             = module.queue.queue_arn
  opensearch_collection_arn = module.vectordb.collection_arn
}

# -----------------------------------------------------------------------------
# Module: Ingest Lambda
# -----------------------------------------------------------------------------

module "ingest" {
  source = "./modules/ingest"

  project_name       = var.project_name
  environment        = var.environment
  ecr_repository_url = module.storage.ecr_repository_url
  execution_role_arn = module.iam.ingest_role_arn
  sqs_queue_arn      = module.queue.queue_arn
  memory_size        = var.lambda_memory_mb
  timeout            = var.lambda_timeout_seconds

  environment_variables = {
    RAG_VECTORSTORE__BACKEND             = "opensearch"
    RAG_VECTORSTORE__OPENSEARCH_ENDPOINT = module.vectordb.collection_endpoint
    RAG_VECTORSTORE__OPENSEARCH_REGION   = var.aws_region
    RAG_VECTORSTORE__COLLECTION          = "investment_docs"
    RAG_EMBEDDING__PROVIDER              = var.embedding_provider
    RAG_EMBEDDING__MODEL                 = var.embedding_model
  }
}

# -----------------------------------------------------------------------------
# Module: Query Lambda
# -----------------------------------------------------------------------------

module "query" {
  source = "./modules/query"

  project_name       = var.project_name
  environment        = var.environment
  ecr_repository_url = module.storage.ecr_repository_url
  execution_role_arn = module.iam.query_role_arn
  memory_size        = var.lambda_memory_mb
  timeout            = var.lambda_timeout_seconds

  environment_variables = {
    RAG_VECTORSTORE__BACKEND             = "opensearch"
    RAG_VECTORSTORE__OPENSEARCH_ENDPOINT = module.vectordb.collection_endpoint
    RAG_VECTORSTORE__OPENSEARCH_REGION   = var.aws_region
    RAG_VECTORSTORE__COLLECTION          = "investment_docs"
    RAG_EMBEDDING__PROVIDER              = var.embedding_provider
    RAG_EMBEDDING__MODEL                 = var.embedding_model
    RAG_LLM__PROVIDER                    = var.llm_provider
    RAG_LLM__MODEL                       = var.llm_model
  }
}

# -----------------------------------------------------------------------------
# Module: API Gateway
# -----------------------------------------------------------------------------

module "api" {
  source = "./modules/api"

  project_name            = var.project_name
  environment             = var.environment
  query_lambda_arn        = module.query.function_arn
  query_lambda_invoke_arn = module.query.invoke_arn
}

# -----------------------------------------------------------------------------
# S3 -> SQS Event Notification
# Placed in root to avoid circular dependency between storage and queue modules
# -----------------------------------------------------------------------------

resource "aws_s3_bucket_notification" "document_upload" {
  bucket = module.storage.bucket_id

  queue {
    queue_arn     = module.queue.queue_arn
    events        = ["s3:ObjectCreated:*"]
    filter_prefix = "uploads/"
    filter_suffix = ".pdf"
  }

  queue {
    queue_arn     = module.queue.queue_arn
    events        = ["s3:ObjectCreated:*"]
    filter_prefix = "uploads/"
    filter_suffix = ".txt"
  }

  queue {
    queue_arn     = module.queue.queue_arn
    events        = ["s3:ObjectCreated:*"]
    filter_prefix = "uploads/"
    filter_suffix = ".html"
  }

  depends_on = [module.queue]
}

# -----------------------------------------------------------------------------
# OpenSearch Serverless Data Access Policy
# Placed in root because it needs IAM role ARNs from the iam module
# and the collection name from the vectordb module (avoids circular deps)
# -----------------------------------------------------------------------------

resource "aws_opensearchserverless_access_policy" "data_access" {
  name = "${local.name_prefix}-data-access"
  type = "data"

  policy = jsonencode([
    {
      Description = "Allow ingest and query Lambda roles to access OpenSearch collection"
      Rules = [
        {
          ResourceType = "index"
          Resource = [
            "index/${module.vectordb.collection_name}/*"
          ]
          Permission = [
            "aoss:CreateIndex",
            "aoss:DeleteIndex",
            "aoss:UpdateIndex",
            "aoss:DescribeIndex",
            "aoss:ReadDocument",
            "aoss:WriteDocument"
          ]
        },
        {
          ResourceType = "collection"
          Resource = [
            "collection/${module.vectordb.collection_name}"
          ]
          Permission = [
            "aoss:CreateCollectionItems",
            "aoss:DeleteCollectionItems",
            "aoss:UpdateCollectionItems",
            "aoss:DescribeCollectionItems"
          ]
        }
      ]
      Principal = [
        module.iam.ingest_role_arn,
        module.iam.query_role_arn
      ]
    }
  ])
}

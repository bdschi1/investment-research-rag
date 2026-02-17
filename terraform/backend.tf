terraform {
  backend "s3" {
    bucket         = "investment-research-rag-tfstate"
    key            = "terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}

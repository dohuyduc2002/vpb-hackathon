terraform {
  required_version = ">= 1.3.2"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.95, < 6.0.0"
    }
    time = {
      source  = "hashicorp/time"
      version = ">= 0.9"
    }
    tls = {
      source  = "hashicorp/tls"
      version = ">= 3.0"
    }
  }
}


module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "19.15.1"

  cluster_name                   = local.name
  cluster_endpoint_public_access = true

  cluster_addons = {
    coredns = { most_recent = true }
    kube-proxy = { most_recent = true }
    vpc-cni = { most_recent = true }
    aws-ebs-csi-driver = { most_recent = true }
  }

  vpc_id                   = module.vpc.vpc_id
  subnet_ids               = module.vpc.private_subnets
  control_plane_subnet_ids = module.vpc.intra_subnets

  eks_managed_node_group_defaults = {
    ami_type       = "AL2023_x86_64_STANDARD"
    instance_types = ["t3a.2xlarge"]
    attach_cluster_primary_security_group = true
  }

  eks_managed_node_groups = {
    main = {
      min_size     = 1
      max_size     = 1
      desired_size = 1

      instance_types = ["t3a.2xlarge"]

      tags = {
        ExtraTag = "vpb-hackathon"
      }
    }
  }

  tags = local.tags 
}

resource "aws_iam_role_policy_attachment" "ec2_full_access" {
  role       = module.eks.eks_managed_node_groups["main"].iam_role_name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2FullAccess"
}
resource "aws_iam_role_policy_attachment" "s3_full_access" {
  role       = module.eks.eks_managed_node_groups["main"].iam_role_name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}
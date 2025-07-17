module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "19.15.1"

  cluster_name                   = local.name
  cluster_endpoint_public_access = true

  cluster_addons = {
    coredns               = { most_recent = true }
    kube-proxy            = { most_recent = true }
    vpc-cni               = { most_recent = true }
    aws-ebs-csi-driver    = { most_recent = true }
  }

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.public_subnets

  eks_managed_node_group_defaults = {
    ami_type       = "AL2023_x86_64_STANDARD"
    instance_types = ["t3a.2xlarge"]
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

  node_security_group_tags = {
    "kubernetes.io/cluster/${local.name}" = null
  }

  cluster_security_group_tags = {}
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "node_full_access" {
  role       = module.eks.eks_managed_node_groups["main"].iam_role_name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}
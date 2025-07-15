module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 4.0"

  name = local.name
  cidr = local.vpc_cidr

  azs            = local.azs
  public_subnets = local.public_subnets

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }
  
  map_public_ip_on_launch = true

}
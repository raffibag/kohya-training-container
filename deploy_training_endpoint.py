#!/usr/bin/env python3
"""
Deploy Kohya Training Container to SageMaker Endpoint
Creates endpoint for running training container interactively for tasks like captioning
"""

import boto3
import json
import time
import argparse
from datetime import datetime

class KohyaTrainingEndpointDeployer:
    def __init__(self, profile='raffibag', region='us-west-2'):
        """Initialize AWS clients"""
        if profile != 'default':
            session = boto3.Session(profile_name=profile)
            self.sagemaker = session.client('sagemaker', region_name=region)
            self.sts = session.client('sts', region_name=region)
        else:
            self.sagemaker = boto3.client('sagemaker', region_name=region)
            self.sts = boto3.client('sts', region_name=region)
        
        self.region = region
        self.account_id = self.sts.get_caller_identity()['Account']
        
    def get_execution_role_arn(self):
        """Get the SageMaker execution role ARN"""
        # Use the existing execution role from the infrastructure
        return f"arn:aws:iam::{self.account_id}:role/CharacterAIPipelineStack-SageMakerExecutionRole7843-JiJ98R0jHRub"
    
    def deploy_endpoint(self, 
                       endpoint_name="kohya-training-endpoint",
                       instance_type="ml.g4dn.xlarge",
                       force_recreate=False):
        """Deploy the training container as a SageMaker endpoint"""
        
        timestamp = str(int(time.time()))
        model_name = f"kohya-training-model-{timestamp}"
        config_name = f"kohya-training-config-{timestamp}"
        
        print(f"🚀 Deploying Kohya Training Endpoint: {endpoint_name}")
        print(f"   Instance Type: {instance_type}")
        
        # Check if endpoint already exists
        try:
            response = self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
            if response['EndpointStatus'] == 'InService' and not force_recreate:
                print(f"✅ Endpoint {endpoint_name} already exists and is InService")
                return response['EndpointArn']
            elif force_recreate:
                print(f"🔄 Force recreating endpoint {endpoint_name}")
                self.sagemaker.delete_endpoint(EndpointName=endpoint_name)
                print("   Waiting for endpoint deletion...")
                self._wait_for_endpoint_deletion(endpoint_name)
        except self.sagemaker.exceptions.ClientError:
            print(f"   Endpoint {endpoint_name} does not exist, creating new...")
        
        # 1. Create Model
        print(f"📦 Creating model: {model_name}")
        image_uri = f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/kohya-training:latest"
        execution_role = self.get_execution_role_arn()
        
        model_response = self.sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': image_uri,
                'Mode': 'SingleModel'
            },
            ExecutionRoleArn=execution_role
        )
        print(f"   ✅ Model created: {model_response['ModelArn']}")
        
        # 2. Create Endpoint Configuration
        print(f"⚙️  Creating endpoint config: {config_name}")
        config_response = self.sagemaker.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    'VariantName': 'primary',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1
                }
            ]
        )
        print(f"   ✅ Config created: {config_response['EndpointConfigArn']}")
        
        # 3. Create Endpoint
        print(f"🎯 Creating endpoint: {endpoint_name}")
        endpoint_response = self.sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        print(f"   ✅ Endpoint creation started: {endpoint_response['EndpointArn']}")
        
        # 4. Wait for endpoint to be in service
        print("⏳ Waiting for endpoint to be InService...")
        self._wait_for_endpoint_creation(endpoint_name)
        
        print(f"🎉 Endpoint {endpoint_name} is ready!")
        print(f"   ARN: {endpoint_response['EndpointArn']}")
        
        return endpoint_response['EndpointArn']
    
    def _wait_for_endpoint_creation(self, endpoint_name, max_wait_time=1200):
        """Wait for endpoint to be InService"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                response = self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
                status = response['EndpointStatus']
                
                if status == 'InService':
                    print(f"   ✅ Endpoint is InService!")
                    return True
                elif status == 'Failed':
                    print(f"   ❌ Endpoint creation failed: {response.get('FailureReason', 'Unknown error')}")
                    return False
                else:
                    print(f"   ⏳ Status: {status} (elapsed: {int(time.time() - start_time)}s)")
                    time.sleep(30)
                    
            except Exception as e:
                print(f"   ⚠️  Error checking status: {e}")
                time.sleep(30)
        
        print(f"   ⏰ Timeout after {max_wait_time}s")
        return False
    
    def _wait_for_endpoint_deletion(self, endpoint_name, max_wait_time=600):
        """Wait for endpoint to be deleted"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
                print(f"   ⏳ Deleting... (elapsed: {int(time.time() - start_time)}s)")
                time.sleep(30)
            except self.sagemaker.exceptions.ClientError:
                print(f"   ✅ Endpoint deleted")
                return True
        
        print(f"   ⏰ Deletion timeout after {max_wait_time}s")
        return False
    
    def delete_endpoint(self, endpoint_name="kohya-training-endpoint"):
        """Delete the endpoint"""
        try:
            print(f"🗑️  Deleting endpoint: {endpoint_name}")
            self.sagemaker.delete_endpoint(EndpointName=endpoint_name)
            self._wait_for_endpoint_deletion(endpoint_name)
            print(f"   ✅ Endpoint {endpoint_name} deleted")
        except Exception as e:
            print(f"   ❌ Error deleting endpoint: {e}")
    
    def get_endpoint_status(self, endpoint_name="kohya-training-endpoint"):
        """Get endpoint status"""
        try:
            response = self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            print(f"📊 Endpoint {endpoint_name}: {status}")
            
            if status == 'InService':
                print(f"   Instance Type: {response['ProductionVariants'][0]['CurrentInstanceCount']}x {response['ProductionVariants'][0]['InstanceType']}")
                print(f"   Created: {response['CreationTime']}")
            elif status == 'Failed':
                print(f"   Failure Reason: {response.get('FailureReason', 'Unknown')}")
            
            return status
        except Exception as e:
            print(f"❌ Error getting status: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description="Deploy Kohya Training Container to SageMaker Endpoint")
    parser.add_argument("--action", choices=["create", "delete", "status"], default="create",
                       help="Action to perform")
    parser.add_argument("--endpoint-name", default="kohya-training-endpoint",
                       help="Endpoint name")
    parser.add_argument("--instance-type", default="ml.g4dn.xlarge",
                       help="SageMaker instance type")
    parser.add_argument("--profile", default="raffibag",
                       help="AWS profile")
    parser.add_argument("--region", default="us-west-2",
                       help="AWS region")
    parser.add_argument("--force", action="store_true",
                       help="Force recreate if endpoint exists")
    
    args = parser.parse_args()
    
    deployer = KohyaTrainingEndpointDeployer(
        profile=args.profile,
        region=args.region
    )
    
    if args.action == "create":
        deployer.deploy_endpoint(
            endpoint_name=args.endpoint_name,
            instance_type=args.instance_type,
            force_recreate=args.force
        )
    elif args.action == "delete":
        deployer.delete_endpoint(args.endpoint_name)
    elif args.action == "status":
        deployer.get_endpoint_status(args.endpoint_name)

if __name__ == "__main__":
    main()
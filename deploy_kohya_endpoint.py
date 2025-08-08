#!/usr/bin/env python3
"""Deploy Kohya training container to SageMaker endpoint for testing"""

import boto3
import time
from datetime import datetime

# Initialize clients
sagemaker = boto3.client('sagemaker', region_name='us-west-2')
iam = boto3.client('iam', region_name='us-west-2')

# Configuration
ROLE_ARN = "arn:aws:iam::796245059390:role/CharacterAIPipelineStack-SageMakerExecutionRole7843-JiJ98R0jHRub"
IMAGE_URI = "796245059390.dkr.ecr.us-west-2.amazonaws.com/kohya-training:ultra-lean"
INSTANCE_TYPE = "ml.g5.2xlarge"  # GPU instance for testing

def create_model():
    """Create SageMaker model from ECR image"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"kohya-training-model-{timestamp}"
    
    print(f"Creating model: {model_name}")
    
    response = sagemaker.create_model(
        ModelName=model_name,
        ExecutionRoleArn=ROLE_ARN,
        PrimaryContainer={
            'Image': IMAGE_URI,
            'ModelDataUrl': 's3://vibez-model-registry-796245059390-us-west-2/placeholder/model.tar.gz',  # Required but not used
            'Environment': {
                'SAGEMAKER_PROGRAM': 'serve',
                'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
            }
        }
    )
    
    print(f"Model created: {response['ModelArn']}")
    return model_name

def create_endpoint_config(model_name):
    """Create endpoint configuration"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_name = f"kohya-training-config-{timestamp}"
    
    print(f"Creating endpoint config: {config_name}")
    
    response = sagemaker.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            'VariantName': 'primary',
            'ModelName': model_name,
            'InitialInstanceCount': 1,
            'InstanceType': INSTANCE_TYPE,
            'InitialVariantWeight': 1.0
        }]
    )
    
    print(f"Config created: {response['EndpointConfigArn']}")
    return config_name

def create_endpoint(config_name):
    """Create and deploy endpoint"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    endpoint_name = f"kohya-training-endpoint-{timestamp}"
    
    print(f"Creating endpoint: {endpoint_name}")
    
    response = sagemaker.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name
    )
    
    print(f"Endpoint creation started: {response['EndpointArn']}")
    return endpoint_name

def wait_for_endpoint(endpoint_name):
    """Wait for endpoint to be ready"""
    print(f"Waiting for endpoint {endpoint_name} to be InService...")
    
    while True:
        response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        status = response['EndpointStatus']
        print(f"Status: {status}")
        
        if status == 'InService':
            print("Endpoint is ready!")
            break
        elif status == 'Failed':
            print(f"Endpoint failed: {response.get('FailureReason', 'Unknown')}")
            raise Exception("Endpoint deployment failed")
        
        time.sleep(30)

def main():
    """Deploy Kohya training container to SageMaker endpoint"""
    try:
        # Create model
        model_name = create_model()
        
        # Create endpoint config
        config_name = create_endpoint_config(model_name)
        
        # Create endpoint
        endpoint_name = create_endpoint(config_name)
        
        # Wait for deployment
        wait_for_endpoint(endpoint_name)
        
        print(f"\nSuccessfully deployed Kohya endpoint: {endpoint_name}")
        print(f"Test with: aws sagemaker-runtime invoke-endpoint --endpoint-name {endpoint_name} ...")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
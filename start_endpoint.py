#!/usr/bin/env python3
"""Start Kohya training SageMaker endpoint"""

import boto3
import time
from datetime import datetime

# Configuration
ROLE_ARN = "arn:aws:iam::796245059390:role/CharacterAIPipelineStack-SageMakerExecutionRole7843-JiJ98R0jHRub"
IMAGE_URI = "796245059390.dkr.ecr.us-west-2.amazonaws.com/kohya-training:ultra-lean"
INSTANCE_TYPE = "ml.g4dn.2xlarge"
ENDPOINT_NAME = "kohya-training-endpoint"

def main():
    sagemaker = boto3.client('sagemaker', region_name='us-west-2')
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    try:
        # Check if endpoint already exists
        try:
            response = sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
            print(f"Endpoint {ENDPOINT_NAME} already exists with status: {response['EndpointStatus']}")
            if response['EndpointStatus'] == 'InService':
                print("Endpoint is already running!")
                return
        except sagemaker.exceptions.ClientError:
            pass  # Endpoint doesn't exist, continue
        
        # Create model
        model_name = f"kohya-training-model-{timestamp}"
        print(f"Creating model: {model_name}")
        
        sagemaker.create_model(
            ModelName=model_name,
            ExecutionRoleArn=ROLE_ARN,
            PrimaryContainer={
                'Image': IMAGE_URI,
                'ModelDataUrl': 's3://vibez-model-registry-796245059390-us-west-2/placeholder/model.tar.gz',
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'serve',
                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
                }
            }
        )
        
        # Create endpoint config
        config_name = f"kohya-training-config-{timestamp}"
        print(f"Creating endpoint config: {config_name}")
        
        sagemaker.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[{
                'VariantName': 'primary',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': INSTANCE_TYPE,
                'InitialVariantWeight': 1.0
            }]
        )
        
        # Create endpoint
        print(f"Creating endpoint: {ENDPOINT_NAME}")
        sagemaker.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name
        )
        
        # Wait for deployment
        print(f"Waiting for endpoint to be InService...")
        while True:
            response = sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
            status = response['EndpointStatus']
            print(f"Status: {status}")
            
            if status == 'InService':
                print(f"✅ Endpoint {ENDPOINT_NAME} is ready!")
                break
            elif status == 'Failed':
                print(f"❌ Endpoint failed: {response.get('FailureReason', 'Unknown')}")
                raise Exception("Endpoint deployment failed")
            
            time.sleep(30)
            
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
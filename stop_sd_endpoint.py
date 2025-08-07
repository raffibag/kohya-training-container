#!/usr/bin/env python3
"""Stop SD inference SageMaker endpoint"""

import boto3
import time

ENDPOINT_NAME = "sd-inference-endpoint"

def main():
    sagemaker = boto3.client('sagemaker', region_name='us-west-2')
    
    try:
        # Check if endpoint exists
        try:
            response = sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
            status = response['EndpointStatus']
            print(f"Endpoint {ENDPOINT_NAME} status: {status}")
            
            if status == 'OutOfService':
                print("Endpoint is already stopped!")
                return
                
        except sagemaker.exceptions.ClientError as e:
            if 'ValidationException' in str(e):
                print(f"Endpoint {ENDPOINT_NAME} does not exist")
                return
            raise
        
        # Delete endpoint
        print(f"Deleting endpoint: {ENDPOINT_NAME}")
        sagemaker.delete_endpoint(EndpointName=ENDPOINT_NAME)
        
        # Wait for deletion
        print("Waiting for endpoint deletion...")
        while True:
            try:
                response = sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
                status = response['EndpointStatus']
                print(f"Status: {status}")
                time.sleep(10)
            except sagemaker.exceptions.ClientError as e:
                if 'ValidationException' in str(e):
                    print(f"✅ Endpoint {ENDPOINT_NAME} deleted successfully!")
                    break
                raise
                
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
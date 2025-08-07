#!/usr/bin/env python3
"""Test SD inference endpoint with simple settings"""

import boto3
import json
import base64
from datetime import datetime
import os

# Configuration
ENDPOINT_NAME = "sd-inference-endpoint"
OUTPUT_DIR = "../../_images/outputs/ultra-lean-test/"

def test_inference():
    runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Simple test payload - lower resolution, disable xformers
    payload = {
        "prompt": "a simple red cube on white background",
        "negative_prompt": "blurry, complex",
        "width": 512,
        "height": 512,
        "num_inference_steps": 10,
        "guidance_scale": 7.5,
        "seed": 42,
        "enable_xformers": False  # Disable xformers optimization
    }
    
    print(f"🎨 Testing with simple prompt and settings...")
    print(f"Prompt: {payload['prompt']}")
    print(f"Resolution: {payload['width']}x{payload['height']}")
    
    try:
        # Invoke endpoint
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        
        if 'image' in result:
            # Decode base64 image
            image_data = base64.b64decode(result['image'])
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simple_test_{timestamp}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            with open(filepath, 'wb') as f:
                f.write(image_data)
            
            file_size = len(image_data) / 1024  # KB
            print(f"✅ Image generated successfully!")
            print(f"📁 Saved to: {filepath}")
            print(f"📏 Size: {file_size:.1f} KB")
            
            return True
            
        else:
            print(f"❌ No image in response: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_inference()
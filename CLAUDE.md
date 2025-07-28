# Claude Code Configuration

## AWS Settings
- **Profile**: raffibag
- **Region**: us-west-2
- **Account Context**: Production PyTorch 2.4 + Kohya + CUDA + DreamBooth + LoRA training container.

## Node.js Version
- **Required**: Node.js v20.19.3 (LTS)
- **Switch command**: `nvm use 20.19.3`
- **Set as default**: `nvm alias default 20.19.3`

## Default Commands
When working with AWS resources in this project, always use:
```bash
AWS_PROFILE=raffibag AWS_DEFAULT_REGION=us-west-2 <aws-command>
```

## CDK Deployment
```bash
# Ensure correct Node version first
nvm use 20.19.3
AWS_PROFILE=raffibag source venv/bin/activate && cdk deploy --require-approval never
```
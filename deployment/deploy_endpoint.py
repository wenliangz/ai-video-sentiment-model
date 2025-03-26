from sagemaker.pytorch import PyTorchModel
import sagemaker
import os

os.environ['AWS_PROFILE'] = 'saml'

def deploy_endpoint():
    sagemaker.Session()
    role = "arn:aws:iam::942524897102:role/CUSPFE-multimodal-deploy-endpoint-role"

    model_uri = "s3://multimodal-analysis-saas/inference/model.tar.gz"

    model = PyTorchModel(
        model_data=model_uri,
        role=role,
        framework_version="2.5.1",
        py_version="py311",
        entry_point="inference.py",
        source_dir=".",
        name="multimodal-analysis-model",
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.g5.xlarge",
        endpoint_name="multimodal-analysis-endpoint",
    )


if __name__ == "__main__":
    deploy_endpoint()

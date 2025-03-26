import boto3
import json
import os


os.environ['AWS_PROFILE'] = 'saml'

# Initialize the clients
s3_client = boto3.client("s3")
sagemaker_runtime = boto3.client("sagemaker-runtime")

# Define the endpoint name
endpoint_name = "multimodal-analysis-endpoint"

# S3 path of the video file
s3_video_path = "s3://multimodal-analysis-saas/dataset/test/output_repeated_splits_test/dia157_utt4.mp4"

# Construct the payload (assuming the model can handle an S3 path)
payload = json.dumps({"video_path": s3_video_path})


# Invoke the endpoint
response = sagemaker_runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",  # Adjust if required
    Body=payload
)

# Process the response
result = response["Body"].read().decode("utf-8")
print("Model Response:", result)

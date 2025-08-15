from typing import (
    Any,
    Optional,
)
from helpers import (
    list_obj_s3
)
import json
import base64
import os

class BatchInference():
    def create_input_jsonl(self) -> None:
        """
        Function to create input.jsonl file for invoking the model. Check if the input.jsonl file already exists in the S3 bucket first.
        """
        
        list_of_images = list_obj_s3(s3_client=self.s3_client,
                                    bucket_name=self.bucket_name,
                                    folder_name=self.folder_name)

        input_json_file = []
        for image_filename in list_of_images:
            image = self.s3_client.get_object(Bucket=self.bucket_name,
                                            Key=image_filename)
            image_binary = image["Body"].read()
            image_bytes = base64.b64encode(image_binary).decode('utf-8')
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_bytes
                    }
                }
            ]

            json_obj = {
                "recordId": os.path.basename(image_filename),
                "modelInput": {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1024,
                    "system": self.creation_prompt,
                    "messages": [
                        {
                            "role": "user",
                            "content": content
                        }
                    ]
                }
            }
            input_json_file.append(json_obj)
        
        with open('input.jsonl', 'w') as f:
            for json_obj in input_json_file:
                f.write(json.dumps(json_obj) + "\n")
        print("\x1b[32mCreated JSONL file and store local copy as input.jsonl\x1b[0m")
        
        # Upload JSONL file to S3.
        try:
            print(f"\x1b[31mUploading input.jsonl file to S3 bucket at path {self.bucket_name}/input.jsonl\x1b[0m")
            with open('input.jsonl', 'rb') as f:
                self.s3_client.put_object(Bucket=self.bucket_name,
                                    Key='input.jsonl',
                                    Body=f)
            print("\x1b[32mUploaded file\x1b[0m")
        except Exception as e:
            print(e)

    def __init__(self, 
                 bedrock_client: Any,
                 s3_client: Any,
                 bucket_name: str,
                 folder_name: str,
                 output_folder: str, 
                 model_id: str,
                 creation_prompt: str,
                 role_arn: str,
                 job_name: str
                 ):
        """
        Tool to run a batch inference job. The process can be divided into three steps.
        1. Creation of batch inference job (start_batch_inference_job).
        2. Polling of job status (poll_job).
        3. Post processing of output JSONL file (post_processing).

        Prerequisites include creating a role to allow batch inference job. Output folder 
        where outputs will be saved. By default, tool will look at the latest folder for post processing.

        Parameters:
            bedrock_client (Any): Bedrock client object.
            s3_client (Any): S3 client object.
            bucket_name (str): S3 bucket name.
            folder_name (str): Folder where files are present to ingest.
            output_folder (str): Output folder name/path (it should already exist).
            moel_id (str): Inference profile ID of model that allows batch inferencing. 
                           Check Service Quotas in AWS console for more information.
            creation_prompt (str): System prompt for each record.
            role_arn (str): ARN of role that allows batch inferencing job. For more info refer
            https://docs.aws.amazon.com/bedrock/latest/userguide/batch-iam-sr.html
            job_name (str): Unique job name for each batch inference job.

        """
        self.bedrock_client = bedrock_client
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.folder_name = folder_name
        self.output_folder = output_folder
        self.model_id = model_id
        self.creation_prompt = creation_prompt
        self.role_arn = role_arn
        self.job_name = job_name

    def start_batch_inference_job(self) -> str:
        """
        Method to start batch inference job. First checks if input.jsonl file is present in S3 bucket or not.
        Creates a new one if it isn't present and starts the job.

        Returns:
            jobArn: ARN of batch inference job. Use this to poll status of job.
        """
        # Check if input.jsonl file exists or not first.
        input_jsonl_yes_no = list_obj_s3(s3_client=self.s3_client,
                                 bucket_name=self.bucket_name,
                                 folder_name='input.jsonl')

        if not input_jsonl_yes_no:
            print("\x1b[31mInput jsonl file does not exist. Creating new one...\x1b[0m")
            self.create_input_jsonl()
        else:
            print("\x1b[32mInput jsonl file already exists. No need to create a new one.\x1b[0m")

        inputDataConfig = {
            "s3InputDataConfig": {
                "s3InputFormat": "JSONL",
                "s3Uri": f"s3://{self.bucket_name}/input.jsonl"
            }
        }

        outputDataConfig = {
            's3OutputDataConfig': {
                's3Uri': f's3://{self.bucket_name}/{self.output_folder}'
            }
        }

        print("\x1b[34mStarting model invocation job...\x1b[0m")

        response = self.bedrock_client.create_model_invocation_job(
            jobName=self.job_name,
            modelId=self.model_id,
            inputDataConfig=inputDataConfig,
            outputDataConfig=outputDataConfig,
            roleArn=self.role_arn,
        )
        print(f"Model invocation job created with ARN: {response['jobArn']}")

        return response['jobArn']

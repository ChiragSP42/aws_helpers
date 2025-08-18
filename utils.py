from typing import (
    Any,
    Optional,
)
from .helpers import (
    list_obj_s3
)
import json
import base64
import os
import sys
import time
import pandas as pd

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
                "recordId": f"s3://{self.bucket_name}/{image_filename}",
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
        print(f"\x1b[32mProcessed {len(list_of_images)} images and created JSONL file and store local copy as input.jsonl\x1b[0m")
        
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
            model_id (str): Inference profile ID of model that allows batch inferencing. 
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
    
    def poll_invocation_job(self, jobArn: str) -> Optional[bool]:
        """Function to poll the status of the model invocation job.

        Parameters:
            jobArn (Optinal[str]): ARN of the model invocation job to poll.

        Returns:
            str: Status of the job.
        """

        # If you're polling a random batch inference job.
        if jobArn:
            counter = 0
            while True:
                status = self.bedrock_client.get_model_invocation_job(jobIdentifier=jobArn)['status']
                dots = "." * (counter % 4)
                sys.stdout.write(f"\r{status}{dots}".ljust(len(status) + 4))
                sys.stdout.flush()
                time.sleep(0.5)
                counter += 1
                if status == 'Completed':
                    return True
                elif status == 'Failed':
                    return False
                time.sleep(5)
        # If you're trying to poll nothing.
        elif not jobArn and not hasattr('self', 'jobArn'):
            print("\x1b[31mEither enter ARN of batch inference job or first start a batch inference job and poll the same object\x1b[0m")

    def process_batch_inference_output(self, local_copy: Optional[bool]=None):
        """
        Function to post process the jsonl file after batch inference job. The outputs are stored as input.jsonl.out in 
        the folder mentioned during inference job creation in the S3DataConfig parameter. The function looks at the first folder 
        in the output folder. Modify the code as necessary.

        Currently output JSON file is of the format;

        {
            "output": [
                {
                    "s3_uri": <recordId>,
                    "license_plate": <license plate number>,
                    "year": <year>,
                    "make": <make>,
                    "model": <model>,
                    "color": <color>,
                    "identifiers": <text>
                }
            ]
        }

        Parameters:
            local_copy (Optional[bool]): Whether you can a local copy as a csv file.
        Returns:
        """
        OUTPUT_FILENAME = 'created_data'

        print("\x1b[31mProcessing output jsonl file\x1b[0m")

        list_folders_output = list_obj_s3(s3_client=self.s3_client,
                               bucket_name=self.bucket_name,
                               folder_name=self.output_folder,
                               delimiter='/')[-1]
        
        response_binary = self.s3_client.get_object(Bucket=self.bucket_name,
                                        Key=os.path.join(list_folders_output, "input.jsonl.out"))["Body"]
        
        output_json_list = []
        processed_counter = 0
        success_counter = 0
        failed_counter = 0
        justOnce = False
        for response in response_binary.iter_lines():
            processed_counter += 1
            try:
                json_obj = json.loads(response.decode('utf-8'))
                text = json_obj["modelOutput"]["content"][0]["text"]
                text = json.loads(text)
                if not justOnce:
                    print(text)
                    justOnce = True
                record_id = json_obj["recordId"] # Contains the filename
                output_json = {
                    "s3_uri": record_id,
                    "license_plate": text.get("license_plate"),
                    "year": text.get("year"),
                    "make": text.get("make"),
                    "model": text.get("model"),
                    "color": text.get("color"),
                    "car_type": text.get("car_type"),
                    "unique_identifiers": text["unique_identifiers"]
                }
                output_json_list.append(output_json)
                success_counter += 1
            except Exception as e:
                json_obj = json.loads(response.decode('utf-8'))
                # text = json_obj["modelOutput"]["content"][0]["text"]
                record_id = json_obj["recordId"] # Contains the filename
                print(f"\x1b[31mJSON extraction failed for {json_obj['recordId']}\x1b[0m")
                # print(text)

                print(f"\x1b[31m{e}\x1b[0m")
                failed_counter += 1
            # print(json.dumps(json.loads(json_obj), indent = 2))
            
        
        output_json = {
            "output": output_json_list
        }
        print("\x1b[32mProcessed JSONl file as a JSON file\x1b[0m")
        print(f"Processed: {processed_counter}\nSuccess: {success_counter}\nFailed: {failed_counter}")
        print("\x1b[31mUploading JSON file\x1b[0m")
        self.s3_client.put_object(Bucket=self.bucket_name,
                            Key=os.path.join(list_folders_output, f"{OUTPUT_FILENAME}.json"),
                            Body=json.dumps(output_json, indent = 2),
                            ContentType='application/json')
        print(f"\x1b[32mUploaded JSON file to S3 bucket of same directory {os.path.join(self.bucket_name, list_folders_output, f'{OUTPUT_FILENAME}.json')}\x1b[0m")

        if local_copy:
            df = pd.DataFrame(output_json['output'])
            df.to_csv(f'{OUTPUT_FILENAME}.csv', index = False)
            print("\x1b[32mCreated local copy as csv file\x1b[0m")

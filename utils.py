from typing import (
    Any,
    Optional,
    Tuple,
    Dict,
    List
)
from .helpers import (
    list_obj_s3,
    _get_s3_client
)
import json
import base64
import os
import sys
import time
import boto3
import random
import re
import requests
from requests.adapters import HTTPAdapter
from botocore.config import Config
from PIL import Image
from io import BytesIO
import pandas as pd
from datasets import Dataset
import concurrent.futures
from threading import Semaphore
import logging

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
                self.s3_client.upload_fileobj(Bucket=self.bucket_name,
                                    Key='input.jsonl',
                                    Fileobj=f)
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

class FineTuning():
    def __init__(self, model: Any,
                 processor: Optional[Any],
                 dataset: Any,
                 batch_size: int,
                 ):
        """
        Tool to perform fine tuning. Fine tuning consists of the following stages.
        1. Data ingestion (Loading the data)
        2. Data preprocessing (Any preprocessing, formatting of dataset, splitting)
        3. Fine tuning configuration
        4. Fine tuning

        Attributes:
            model (Any): The model that will be used to fine tune. Define the object and pass it here.
            processor (Optional[Any]): A processor function. Used to preprocess data into proper format.
            dataset (Any): The dataset of class Dataset or IterableDataset.
            batch_size (int): Batch size if preprocessing dataset.
            s3_client (Any): S3 client object used in preprocessing.
            bucket_name (str): S3 bucket name where data is present.
            folder_name (str): Folder name used in preprocessing the data.

        """
        self.model = model
        self.processor = processor
        self.dataset = dataset
        self.batch_size = batch_size
    
    def split(self, 
              train_size: float=0.8,
              ) -> Tuple[Any, Any]:
        """
        Function to split dataset into train and test. Based on datatype of dataset (Dataset) 
        train_test_split. By default, shuffle is enabled.
        """
        if isinstance(self.dataset, Dataset):
            print("Standard train_test_split function being employed")
            split = self.dataset.train_test_split(train_size=train_size)
            return split["train"], split["test"]
        else:
            raise ValueError("Acceptable datatypes of dataset of datasets.Dataset")

class StreamingCLIPDataset:
    def __init__(self, 
                 dataset_stream: Any,
                 processor: Any,
                 bucket_name: str,
                 folder_name: str,
                 aws_access_key: Optional[str],
                 aws_secret_key: Optional[str],
                 region: str='us-east-1',
                 train_size=0.8, 
                 seed=42, 
                 is_train=True,
                 ):
        self.dataset_stream = dataset_stream
        self.processor = processor
        self.train_size = train_size
        self.seed = seed
        self.is_train = is_train
        self.bucket_name = bucket_name
        self.folder_name = folder_name
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.region = region

    def _get_s3_client(self):
        session = boto3.Session(aws_access_key_id=self.aws_access_key,
                                aws_secret_access_key=self.aws_secret_key,
                                region_name=self.region)
        
        return session.client('s3')
    
    def _load_image_from_s3(self, filename):
        s3_client = self._get_s3_client()

        response = s3_client.get_object(Bucket=self.bucket_name,
                                            Key=f"{self.folder_name}/{filename}")
        image_bytes = response["Body"].read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return image
    
    def _preprocess_sample(self, sample):
        description_template = f"A {sample['year']} {sample['car_type']} {sample['color']} {sample['make']} {sample['model']} with license plate number {sample['license_plate']} has the following unique identifiers {sample['unique_identifiers']}"

        images = [self._load_image_from_s3(image) for image in sample["s3uri"]]

        if self.processor is None:
            raise ValueError("Processor is None. Please provide a valid processor before calling preprocess.")

        preprocessed = self.processor(text=description_template,
                                images=images,
                                padding='max_length',
                                return_tensors='pt',
                                truncation=True)

        return {"input_ids": preprocessed['input_ids'],
                "attention_mask": preprocessed['attention_mask'],
                "pixel_values": preprocessed['pixel_values']
            }
    def __iter__(self):
        random.seed(self.seed)
        for sample in self.dataset_stream:
            is_train_sample = random.random() < self.train_size
            if (self.is_train and is_train_sample) or (not self.is_train and not is_train_sample):
                try:
                    preprocessed_sample = self._preprocess_sample(sample)
                    yield preprocessed_sample
                except Exception as e:
                    print(f"Error preprocessing sample: {e}")
                    continue 

class VehicleProcessor:
    def __init__(self,
                 evox_api_key: str, 
                 bucket_name: str = 'signal-8-evox',
                 productID: int = 3,
                 productTypeID: int = 67,
                 image_workers: int = 10,
                 max_api_concurrency: int = 400):
        """
        Data ingestion tool created to concurrently download images via an API call, extract relevant information 
        and create a dataset as a JSON. This was made very specifically for Signal 8 project to download vehicle 
        information via their API call.
        
        Args:
            bucket_name: S3 bucket name (default: signal-8-evox)
            max_api_concurrency: Maximum concurrent API calls (default 400)
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        # logging.basicConfig(level=logging.INFO, format='%(filename)s:%(funcName)s:%(lineno)d% - (levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.bucket_name = bucket_name
        self.productID = productID
        self.productTypeID = productTypeID
        self.evox_api_key = evox_api_key
        self.session = requests.Session()
        # Create HTTPAdapter with increased pool size
        adapter = HTTPAdapter(
            pool_connections=50,    # Number of different hosts (default 10 is fine)
            pool_maxsize=500        # Max connections per host (increase from 10 to 100)
        )
        
        # Mount adapter for both HTTP and HTTPS
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        self.image_workers = image_workers
        self.max_api_concurrency = max_api_concurrency
        client_config = Config(
            max_pool_connections=500  # Increase from default 10 to 50
        )
        self.s3_client = _get_s3_client(config=client_config)
        
        # Semaphore to limit concurrent API calls to 400
        self.api_semaphore = Semaphore(self.max_api_concurrency)
        
        # Results storage - now using array format
        self.processed_vehicles = []
        self.processed_count = 0
        self.failed_vehicles = []

    def load_vehicle_ids(self, excel_file_path: str, column_name: str = 'VIF #') -> List[str]:
        """
        Load vehicle IDs from Excel file
        
        Args:
            excel_file_path: Path to Excel file
            column_name: Column name containing vehicle IDs
        
        Returns:
            List of vehicle IDs
        """
        df = pd.read_excel(excel_file_path, sheet_name='Sheet1')
        self.logger.info(f"Loaded a total of {df.shape} from EXCEL")
        df = df[df['Exterior'] == 1]
        # df = df.iloc[:1000, :].copy()
        vehicle_ids = df[column_name].dropna().tolist()
        self.logger.info(f"Loaded {len(vehicle_ids)} vehicle IDs from Excel")
        return vehicle_ids

    def clean_filename_part(self, text: str) -> str:
        """
        Clean text for use in S3 key/filename
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text safe for filenames
        """
        if not text:
            return "Unknown"
        # Remove or replace characters that aren't filename-safe
        cleaned = re.sub(r'[<>:"/\\|?*]', '', str(text))
        cleaned = cleaned.replace(' ', '-')  # Keep spaces as spaces
        return cleaned.strip()

    def fetch_vehicle_data(self, vehicle_id: str) -> Dict:
        """
        Fetch vehicle details from API
        
        Args:
            vehicle_id: Unique vehicle identification number
            
        Returns:
            Dictionary containing vehicle details and image URLs
        """
        # Acquire semaphore to limit concurrent API calls
        self.api_semaphore.acquire()
        
        try:
            # Replace with your actual API endpoint
            url = f"https://api.evoximages.com/api/v1/vehicles/{vehicle_id}/products/{self.productID}/{self.productTypeID}?api_key={self.evox_api_key}"
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API call failed for vehicle {vehicle_id}: {e}")
            raise
        finally:
            # Always release the semaphore
            self.api_semaphore.release()

    def download_and_upload_image(self, image_url: str, vehicle_data: Dict, image_index: int) -> str:
        """
        Download image from URL and upload to S3 with specific naming convention
        
        Args:
            image_url: URL of the image to download
            vehicle_data: Vehicle details dictionary from API
            image_index: Index number of the image (1-36)
            
        Returns:
            S3 URI of uploaded image
        """
        try:
            # Download image with timeout
            response = self.session.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Extract vehicle details for filename
            vifnum = vehicle_data.get('vifnum', 'Unknown')
            make = self.clean_filename_part(vehicle_data.get('make', 'Unknown'))
            model = self.clean_filename_part(vehicle_data.get('model', 'Unknown'))
            year = vehicle_data.get('year', 'Unknown')
            color = self.clean_filename_part(vehicle_data.get('color_simpletitle', 'Unknown'))
            
            # Generate S3 key with specified format: {vifnum}-{make}-{model}-{year}-{color}-{image_number}.jpeg
            s3_key = f"Images/{vifnum}-{make}-{model}-{year}-{color}-{image_index}.jpeg"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=response.content,
                ContentType='image/jpeg'
            )
            
            # Return S3 URI
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            self.logger.debug(f"Uploaded image {image_index} for vehicle {vifnum}")
            
            return s3_uri
            
        except Exception as e:
            self.logger.error(f"Failed to process image {image_index} for vehicle {vehicle_data.get('vifnum', 'Unknown')}: {e}")
            # Return empty string for failed uploads
            return ""

    def process_single_vehicle(self, vehicle_id: str) -> Optional[Dict]:
        """
        Process a single vehicle: fetch data, download/upload images, create JSON
        
        Args:
            vehicle_id: Vehicle identification number
            
        Returns:
            Processed vehicle data in the required format
        """
        try:
            # Step 1: Fetch vehicle data from API
            api_response = self.fetch_vehicle_data(vehicle_id)
            
            # Check if API response is successful
            if api_response.get('status') != 'success':
                self.logger.error(f"API returned error status for vehicle {vehicle_id}")
                return None
            
            # Extract vehicle details from the nested structure
            vehicle_data = api_response.get('vehicle', {})
            image_urls = api_response.get('urls', [])
            
            if not vehicle_data:
                self.logger.error(f"No vehicle data found for vehicle {vehicle_id}")
                return None
                
            if not image_urls:
                self.logger.warning(f"No images found for vehicle {vehicle_id}")
            
            # Step 2: Download and upload images concurrently (max 10 per vehicle)
            s3_uris = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.image_workers) as image_executor:
                # Submit all image download/upload tasks
                image_futures = {
                    image_executor.submit(self.download_and_upload_image, url, vehicle_data, idx + 1): (url, idx + 1)
                    for idx, url in enumerate(image_urls)
                }
                
                # Collect results as they complete and maintain order
                results = []
                for future in concurrent.futures.as_completed(image_futures):
                    url, image_index = image_futures[future]
                    try:
                        s3_uri = future.result()
                        if s3_uri:  # Only add successful uploads
                            results.append(s3_uri)
                    except Exception as e:
                        self.logger.error(f"Image processing failed for vehicle {vehicle_id}, image {image_index}: {e}")
                
                # Filter out None values and maintain order
                s3_uris = [s3uri for s3uri in results if s3uri is not None]
            
            # Step 3: Create vehicle object in required format
            vehicle_json = {
                "vifid": vehicle_data.get('vifnum'),
                "year": vehicle_data.get('year'),
                "make": vehicle_data.get('make'),
                "model": vehicle_data.get('model'),
                "trim": vehicle_data.get('trim'),
                "color": vehicle_data.get('color_simpletitle'),
                "car_type": vehicle_data.get('body'),
                "s3uris": s3_uris
            }

            # logger.info(f"Successfully processed vehicle {vehicle_data.get('vifnum')} with {len(s3_uris)} images")
            return vehicle_json
            
        except Exception as e:
            self.logger.error(f"Failed to process vehicle {vehicle_id}: {e}")
            self.failed_vehicles.append(vehicle_id)
            return None

    def process_all_vehicles(self, vehicle_ids: List[str]):
        """
        Process all vehicles concurrently with proper rate limiting
        
        Args:
            vehicle_ids: List of vehicle IDs to process
        """
        total_vehicles = len(vehicle_ids)
        self.logger.info(f"Starting to process {total_vehicles} vehicles with max {self.max_api_concurrency} concurrent API calls and {self.image_workers} image workers.")
        
        # Use ThreadPoolExecutor for concurrent processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_api_concurrency) as executor:
            # Submit all vehicle processing tasks
            future_to_vehicle = {
                executor.submit(self.process_single_vehicle, vehicle_id): vehicle_id 
                for vehicle_id in vehicle_ids
            }
            
            # Process completed tasks as they finish
            for future in concurrent.futures.as_completed(future_to_vehicle):
                vehicle_id = future_to_vehicle[future]
                
                try:
                    vehicle_data = future.result()
                    
                    if vehicle_data:
                        # Add to processed vehicles array
                        self.processed_vehicles.append(vehicle_data)
                        self.processed_count += 1
                        
                        # Log progress every 100 vehicles
                        if total_vehicles < 100:
                            self.logger.info(f"Progress: {self.processed_count}/{total_vehicles} vehicles processed")
                        elif self.processed_count % 100 == 0:
                            self.logger.info(f"Progress: {self.processed_count}/{total_vehicles} vehicles processed")
                    
                except Exception as e:
                    self.logger.error(f"Unexpected error processing vehicle {vehicle_id}: {e}")
                    self.failed_vehicles.append(vehicle_id)

    def save_results(self, output_file: str = 'dataset.json'):
        """
        Save processed data to JSON file in the required format
        
        Args:
            output_file: Output JSON file path
        """
        # Create the final JSON structure with "output" array
        final_json = {
            "output": self.processed_vehicles
        }
        
        # Save main results
        with open(output_file, 'w') as f:
            json.dump(final_json, f, indent=2)

        self.s3_client.put_object(Bucket=self.bucket_name,
                                  Key=output_file,
                                  Body=json.dumps(final_json, indent=2),
                                  ContentType='application/json')
        
        # Save failed vehicles list for retry
        if self.failed_vehicles:
            with open('failed_vehicles.json', 'w') as f:
                json.dump(self.failed_vehicles, f, indent=2)
        
        # Log summary
        total_processed = len(self.processed_vehicles)
        total_failed = len(self.failed_vehicles)
        total_images = sum(len(vehicle.get('s3uris', [])) for vehicle in self.processed_vehicles)
        
        self.logger.info(f"""
        Processing Summary:
        ==================
        Total vehicles processed: {total_processed}
        Total vehicles failed: {total_failed}
        Total images uploaded: {total_images}
        Results saved to: {output_file}
        Failed vehicles saved to: failed_vehicles.json
        """)
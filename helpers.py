import os
from typing import (
    Optional, 
    Dict, 
    Any, 
    List, 
    Tuple)
import logging
import boto3
import sys
import traceback
from dotenv import load_dotenv
load_dotenv()

def _list_foundational_models(byOutputModality: Optional[str] = None,
                 byProvider: Optional[str] = None) -> None:
    """
    Function to list all models available in the session. This requires a AWS access and secret key of a user to be stored 
    as enivronment variables 'AWS_ACCESS_KEY' and 'AWS_SECRET_KEY' respectively. Default region is us-east-1. Prints the 
    Provider name, Model name, Model ARN, Input Modalities and Output Modalities of foundational models available to 
    respective user.
    
    Parameters:
        byOutputModality (Optional[str]): Filter models by output modality, 'TEXT'|'IMAGE'|'EMBEDDING'.
        byProvider (Optional[str]): Filter models by provider.
    
    Returns:
        None
    """

    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY", None)
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY", None)
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        raise ValueError("AWS_ACCESS_KEY and AWS_SECRET_KEY must be set in the environment variables.")
    
    session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY,
                            aws_secret_access_key=AWS_SECRET_KEY,
                            region_name='us-east-1')
    if not session:
        raise ValueError("Failed to create a Boto3 session. Check your AWS credentials and region.")
    
    bedrock = session.client('bedrock')
    if byOutputModality and byProvider:
        response = bedrock.list_foundation_models(
            byOutputModality=byOutputModality,
            byProvider=byProvider
        )
    elif byOutputModality or byProvider:
        if byOutputModality:
            response = bedrock.list_foundation_models(
                byOutputModality=byOutputModality
            )
        else:
            response = bedrock.list_foundation_models(
                byProvider=byProvider
            )
    else:
        response = bedrock.list_foundation_models()
    
    if 'modelSummaries' not in response:
        raise ValueError("No models found in the response. Check your AWS credentials and permissions.")
    
    for model in response['modelSummaries']:
        print(f"Provider name: {model['providerName']}\nModel Name: {model['modelName']}\nModel ARN: {model['modelArn']}")
        print(f"Input Modalities: {model['inputModalities']}\nOutput Modalities: {model['outputModalities']}")
        print("-" * 30)

def _list_inference_profiles() -> None:
    """
    Function to list all inference profiles available in the session. This requires a AWS access and secret key of a user to be stored 
    as enivronment variables 'AWS_ACCESS_KEY' and 'AWS_SECRET_KEY' respectively. Default region is us-east-1. Prints the Profile name 
    and Profile ID of all inference models available for the respective user.
    
    Returns:
        None
    """
    
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        raise ValueError("AWS_ACCESS_KEY and AWS_SECRET_KEY must be set in the environment variables.")
    session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY,
                            aws_secret_access_key=AWS_SECRET_KEY,
                            region_name='us-east-1')
    if not session:
        raise ValueError("Failed to create a Boto3 session. Check your AWS credentials and region.")
    
    bedrock = session.client('bedrock')
    response = bedrock.list_inference_profiles()
    for profile in response.get('inferenceProfileSummaries', []):
        print(f"Profile Name: {profile['inferenceProfileName']}\nProfile ID: {profile['inferenceProfileId']}")
        print("-" * 30)

def _parse_arn(arn: str) -> Dict:
    """
    Function to parse ARN to extract components. Following are the
    different ARN formats.

    arn:partition:service:region:account_id:resource_id\n
    arn:partition:service:region:account_id:resource_type/resource_id\n
    arn:partition:service:region:account_id:resource_type/resource_id\n

    For more documentation, read this;
    https://docs.aws.amazon.com/IAM/latest/UserGuide/reference-arns.html

    Parameters:
        arn (str): The ARN to be parsed.

    Returns:
        result (dict): Returns a dictionary of different components of ARN.
        {\n
        'arn': elements[0],\n
        'partition': elements[1],\n
        'service': elements[2],\n
        'region': elements[3],\n
        'account': elements[4],\n
        'resource': elements[5],\n
        'resource_type': None\n
        }
    """

    try:
        elements = arn.split(':', 5)
        result = {
        'arn': elements[0],
        'partition': elements[1],
        'service': elements[2],
        'region': elements[3],
        'account': elements[4],
        'resource': elements[5],
        'resource_type': None
        }
        if '/' in result['resource']:
            result['resource_type'], result['resource'] = result['resource'].split('/',1)
        elif ':' in result['resource']:
            result['resource_type'], result['resource'] = result['resource'].split(':',1)
        return result
    except:
        print(f"ARN {arn} could not be parsed")
        sys.exit(0)

def _local_or_sagemaker() -> bool:
    """
    Checks if the current Python script is running within an Amazon SageMaker environment or locally.
    Returns True if running on SageMaker, False if running locally.
    """
    # As a fallback, check for other common SageMaker environment variables
    sagemaker_env_vars = ['SM_CHANNEL_TRAIN', 'SM_MODEL_DIR', 'SAGEMAKER_PROGRAM']
    for var in sagemaker_env_vars:
        if var in os.environ:
            print(var)
            return True

    return False

def _get_s3_client(aws_access_key: Optional[str]=None,
                   aws_secret_key: Optional[str]=None,
                   config: Optional[Any]=None,
                   region_name: str='us-east-1'):
    """
    Function to generate S3 client object. Access keys are retrieved from .env by default.
    If alternate keys can be provisioned via parameters. Default region name is 'us-east-1'.

    Parameters:
        aws_access_key (Optional[str]): AWS Access key ID.
        aws_secret_key (Optional[str]): AWS Secret key ID.
        region_name (str): Region name.

    Returns:
        S3 client object.
    """
    
    if aws_access_key and aws_secret_key is None:
        aws_access_key = os.getenv("AWS_ACCESS_KEY")
        aws_secret_key = os.getenv("AWS_SECRET_KEY")
        if aws_access_key and aws_secret_key is None:
            raise ValueError("AWS credentials not set in environment.")
        else:
            session = boto3.Session(aws_access_key_id=aws_access_key,
                                    aws_secret_access_key=aws_secret_key,
                                    region_name=region_name)
        
            s3_client = session.client("s3", config=config)

            return s3_client
    else:
        session = boto3.Session(aws_access_key_id=aws_access_key,
                                    aws_secret_access_key=aws_secret_key,
                                    region_name=region_name)
            
        s3_client = session.client("s3", config=config)

        return s3_client
    
def _setup_logger(name: str, level: int, handler_type: str='stream', filename: Optional[str]=None):
    """
    Set up and return a logger with a given name and level.
    
    Parameters:
        name (str): Name of logger
        level (int): Level of logger like logging.DEBUG=10, logging.INFO=20, logging.WARNING=30, logging.ERROR=40, logging.CRITICAL=50.
        handler_type (str): 'stream' for console output, 'file' for file output.
        filename (Optional[str]): required if handler_type is 'file'.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handler based on type
    if handler_type == 'stream':
        handler = logging.StreamHandler()
    elif handler_type == 'file':
        if not filename:
            raise ValueError("Filename must be provided for file handler")
        handler = logging.FileHandler(filename)
    else:
        raise ValueError("Unknown handler_type. Use 'stream' or 'file'.")
    
    # Define formatter
    formatter = logging.Formatter('%(filename)s:%(funcName)s:%(lineno)d - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Avoid adding handlers multiple times if logger is configured more than once
    if not logger.hasHandlers():
        logger.addHandler(handler)
    
    return logger

def list_obj_s3(s3_client: Any,
                bucket_name: Optional[str],
                folder_name: Optional[str],
                delimiter: Optional[str] = '')-> List[str]:
    """
    Function to return list of objects present in bucket. There is an optional
    delimiter parameter to toggle between folder and file names. If delimiter is empty, 
    it will return all files in the bucket.

    Parameters:
        s3_client (Any): S3 client object
        bucket_name (str): Name of S3 bucket where concerned objects are present.
        foldername (str): Name of folder in which objects are present.
        delimiter (str): Delimiter to toggle between folder and file names. Default is '/'.

    Returns:
        pdf_list (list[str]): List of object names with folder path included.
    """

    obj_list = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name,
                                   Prefix=folder_name,
                                   Delimiter=delimiter):
        if delimiter:
            if 'CommonPrefixes' in page:
                obj_list = [obj["Prefix"] for obj in page.get('CommonPrefixes', [])]
        else:
            for obj in page.get('Contents', []):
                key = obj['Key']
                obj_list.append(key)

    return obj_list

def create_sns_topic(sns_client: Any) -> str:
    """
    Function to create a SNS topic. Can be generalized to create any SNS topic, 
    just need to alter the logic for the SNS topic name. Right now it's just 
    pulling from the .env file as SNS_TOPIC_NAME.

    Parameters:
        sns_client (Any): SNS client object.

    Returns:
        str: Returns the SNS ARN of the concerned topic.
    """
    # First check if any SNS topics present.
    SNS_TOPIC_NAME = os.getenv('SNS_TOPIC_NAME')
    topics = []

    response = sns_client.list_topics()
    if "Topics" in response:
        if response['Topics']:
            topics.extend(response['Topics'])
    if "NextToken" in response:
        while True:
            next_token = response['NextToken']
            contd_response = sns_client.list_topics(NextToken=next_token)
            topics.extend(contd_response["Topics"])
            if "NextToken" in contd_response:
                next_token = contd_response['NextToken']
            else:
                break
    
    for topic in topics:
        topic_name = {}
        topic_name = _parse_arn(topic['TopicArn'])
        # If Topic already present, pass back the ARN.
        if topic_name:
            if SNS_TOPIC_NAME == topic_name.get('resource'):
                print("\x1b[32mSNS topic already created.\x1b[0m")
                return topic['TopicArn']
    # If Topic not present, create and pass ARN.
    sns_response = sns_client.create_topic(Name = SNS_TOPIC_NAME)
    print("\x1b[32mCreated a SNS topic\x1b[0m")
    return sns_response['TopicArn']


def create_sqs_queue(sqs_client: Any) -> Tuple[str, str]:
    """
    Function to create a SQS queue if it doesn't exist. Right now it's just pulling from 
    the .env file for SQS queue nameas SQS_QUEUE_NAME.

    Parameters:
        sqs_client (Any): SQS client object.

    Returns:
        sqs_url (str): SQS URL.
        sqs_arn (str): SQS ARN.
    """
    SQS_QUEUE_NAME = os.getenv('SQS_QUEUE_NAME')
    urls = []

    response = sqs_client.list_queues()
    if "QueueUrls" in response:
        if response['QueueUrls']:
            urls.extend(response['QueueUrls'])
    if "NextToken" in response:
        while True:
            next_token = response['NextToken']
            contd_response = sqs_client.list_queues(NextToken=next_token)
            urls.extend(contd_response['QueueUrls'])
            if "NextToken" in contd_response:
                next_token = contd_response['NextToken']
            else:
                break

    for url in urls:
        queue_name = url.split('/')[-1]
        if queue_name:
            if SQS_QUEUE_NAME == queue_name:
                print("\x1b[32mSQS queue already created.\x1b[0m")
                try:
                    sqs_attrs = sqs_client.get_queue_attributes(QueueUrl = url, AttributeNames = ['QueueArn'])
                    sqs_arn = sqs_attrs['Attributes']['QueueArn']
                    return url, sqs_arn
                except Exception as e:
                    print(f"ERROR: {e}\n\nCouldn't retrieve SQS queue ARN.")
                    traceback.print_exc()
                    sys.exit(0)

    try:
        sqs_response = sqs_client.create_queue(QueueName = SQS_QUEUE_NAME)
        print("\x1b[32mCreated SQS Queue\x1b[0m")
    except Exception as e:
        print(f"ERROR: {e}\n\nCouldn't create SQS queue.")
        traceback.print_exc()
        sys.exit(0)
    try:
        sqs_attrs = sqs_client.get_queue_attributes(QueueUrl = sqs_response['QueueUrl'], AttributeNames = ['QueueArn'])
        sqs_arn = sqs_attrs['Attributes']['QueueArn']
        return sqs_response['QueueUrl'], sqs_arn
    except Exception as e:
        print(f"ERROR: {e}\n\nCouldn't retrieve SQS queue ARN")
        traceback.print_exc()
        sys.exit(0)
    
def subscribe(sns_arn: str, sqs_arn: str, sns_client: Any) -> None:
    """
    Function that subscribes SQS queue to SNS notification topic. Pulling SQS queue name from .env file 
    as SQS_QUEUE_NAME.

    Parameters:
        sns_arn (str): SNS ARN.
        sqs_arn (str): SQS ARN.
        sqs_url (str): SQS URL to retrieve queue attributes.
        sns_client (Any): SNS client object.

    Returns:
        None
    """
    # First let's check if there is a subscription already present.
    paginator = sns_client.get_paginator('list_subscriptions_by_topic')
    for page in paginator.paginate(TopicArn=sns_arn):
        if page["Subscriptions"]:
            for subscription in page['Subscriptions']:
                endpoint = _parse_arn(subscription["Endpoint"]).get("resource")
                if os.getenv('SQS_QUEUE_NAME') == endpoint:
                    print(f"\x1b[32mSubscription is already present for queue {endpoint}.\x1b[0m")
                    return None

    sns_client.subscribe(TopicArn = sns_arn, Protocol = 'sqs', Endpoint = sqs_arn)
    print("\x1b[32mSubscribed SQS Queue to SNS Topic\x1b[0m")
from time import sleep
from prefect_aws import S3Bucket, AwsCredentials
from dotenv import load_dotenv
load_dotenv()
import os

def create_aws_creds_block():
    my_aws_creds_obj = AwsCredentials(
        aws_access_key_id=os.environ['aws_access_key_id'], aws_secret_access_key=os.environ['aws_secret_access_key']
    )
    my_aws_creds_obj.save(name="my-aws-creds", overwrite=True)


def create_s3_bucket_block():
    aws_creds = AwsCredentials.load("my-aws-creds")
    my_s3_bucket_obj = S3Bucket(
        bucket_name="apple-quality", credentials=aws_creds
    )
    my_s3_bucket_obj.save(name="apple-quality-bucket", overwrite=True)  #remember to create a name with smallcase and dashes only


if __name__ == "__main__":
    create_aws_creds_block()
    sleep(10)
    create_s3_bucket_block()

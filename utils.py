import psycopg2
import os
import boto3
from botocore.exceptions import ClientError

AWS_REGION = 'us-west-2'
CHARSET = "utf-8"
SENDER = 'do-not-reply@emotechnologies.com'


def get_credentials():
    name = os.environ['DATABASE_NAME']
    host = os.environ['DATABASE_HOST']
    username = os.environ['DATABASE_USERNAME']
    password = os.environ['DATABASE_PASSWORD']
    return name, host, username, password


def create_conn():
    db_database, db_host, db_user, db_password = get_credentials()
    db_port = 5432
    conn = psycopg2.connect(user=db_user, password=db_password, host=db_host, port=db_port, database=db_database)
    conn.autocommit = False
    return conn


def get_ip_address(company_uuid, chatbot_name):
    select_ip = 'SELECT address FROM chatbots WHERE company_uuid = %s AND name = %s'
    conn = create_conn()
    cursor = conn.cursor()
    cursor.execute(select_ip, (company_uuid, chatbot_name))
    ip_addresss = cursor.fetchone()
    print(ip_addresss[0])  # return tuple
    return ip_addresss[0]


def add_billing(company_uuid, question, answer):
    add_billing = ("INSERT INTO billing (company_uuid, notes, app_name, billing_type, value) "
                   "VALUES (%s, %s, 'NICE', 'SENTENCES_COUNT', 1)")
    input_json = {'question': question, 'answer': answer}
    conn = create_conn()
    cursor = conn.cursor()
    cursor.execute(add_billing, (company_uuid, json.dumps(input_json)))
    conn.commit()


def update_status(company_uuid, chatbot_name, status):
    update_status = ("UPDATE chatbots SET status = %s WHERE company_uuid = %s and name = %s")
    conn = create_conn()
    cursor = conn.cursor()
    cursor.execute(update_status, (status, company_uuid, chatbot_name))
    conn.commit()
    # closing database connection.
    if (conn):
        cursor.close()
        conn.close()
        print("PostgreSQL connection is closed")


def send_email(recipient, subject, body):
    try:
        client = boto3.client('ses', region_name=AWS_REGION)
        response = client.send_email(
            Destination={
                'ToAddresses': recipient,
                'BccAddresses': [
                    'xuyuan.kee@emotechnologies.com',
                    'desmond.lim@emotechnologies.com',
                ]
            },
            Message={
                'Body': {
                    'Text': {
                        'Charset': CHARSET,
                        'Data': body,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': subject,
                },
            },
            Source=SENDER,
        )
    except ClientError as err:
        error = err.response['Error']['Message']
        print(f'{error}')
    else:
        resp = response['MessageId']
        print(f'Email sent. Message ID: {resp}')

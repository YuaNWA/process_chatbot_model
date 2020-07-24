import os
from configparser import ConfigParser

import boto3


class Config(object):
    SECRET_KEY = "3hHTye0uEYYUqYQwGWAjljCjHiZNMTrS"


class ProductionConfig(Config):
    pass


def config(filename='database.ini', section='postgresql_dev'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    print(db)
    return db


def get_credentials():
    server = os.environ.get('SERVER')

    if server:
        ssm = boto3.client('ssm', region_name='ap-southeast-1')

        parameter = ssm.get_parameter(Name='/{}/DATABASE_NAME'.format(server), WithDecryption=True)
        name = parameter.get('Parameter').get('Value')

        parameter = ssm.get_parameter(Name='/{}/DATABASE_HOST'.format(server), WithDecryption=True)
        host = parameter.get('Parameter').get('Value')

        parameter = ssm.get_parameter(Name='/{}/DATABASE_USERNAME'.format(server), WithDecryption=True)
        username = parameter.get('Parameter').get('Value')

        parameter = ssm.get_parameter(Name='/{}/DATABASE_PASSWORD'.format(server), WithDecryption=True)
        password = parameter.get('Parameter').get('Value')

    else:
        name = os.environ.get('DATABASE_NAME')
        host = os.environ.get('DATABASE_HOST')
        username = os.environ.get('DATABASE_USERNAME')
        password = os.environ.get('DATABASE_PASSWORD')

    # print('{} {} {} {}'.format(name, host, username, password))
    return {'database': name, 'host': host, 'user': username, 'password': password}

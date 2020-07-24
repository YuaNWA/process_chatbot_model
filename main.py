# -*- coding: utf-8 -*-
import glob
import json
import logging
import os
import shutil
import zipfile
from logging.handlers import RotatingFileHandler
from time import sleep
import requests

import nltk
import paramiko
from deeppavlov import train_model
from deeppavlov.core.common.file import read_json
from termcolor import colored

from reaction_tools import prep_synonyms
from utils import get_ip_address
from utils import send_email

import pandas as pd

nltk.data.path.append("/nltk/")

# logging init
formatter = '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'
formatter_entry = '%(message)s'


def setup_logger(logger_name, log_file, fmt, level=logging.DEBUG):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(fmt)
    fileHandler = RotatingFileHandler(
        log_file, maxBytes=1000000, backupCount=10)
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


setup_logger('general_logger', r'./logs/chatbot.log',
             formatter, logging.DEBUG)
logger = logging.getLogger('general_logger')


def train_custom_model(data_path, x_col_name, y_col_name, save_load_path):
    model_config = read_json("./configs/tfidf_logreg_en_faq.json")
    model_config['dataset_reader']['x_col_name'] = x_col_name
    model_config['dataset_reader']['y_col_name'] = y_col_name
    model_config["dataset_reader"]["data_path"] = data_path
    model_config["dataset_reader"]["data_url"] = None
    model_config['metadata']['variables']['MODELS_PATH'] = save_load_path
    if data_path in model_config['dataset_reader']:
        del model_config['dataset_reader']['data_url']
    model_config['dataset_reader']['data_path'] = data_path

    custom_model = train_model(model_config)
    return custom_model


MAX_FILES = 5


def get_data_from_file_name(file_path):
    file_path_split = file_path.split('/')
    user_file_split = file_path_split[2].split('--')  # 0: company_uuid, 1: chatbot_name, 2: email name, 3: email domain, 4: file_name

    # 0: user_uuid, 1: file name
    return user_file_split[0], user_file_split[1], user_file_split[2], user_file_split[3], user_file_split[4]


def process_csv_files():
    logger.info("processing files start")
    # check output_files in s3
    files_list = glob.glob("./input_files/*.csv")

    for file in files_list:
        company_uuid, chatbot_name, email_name, email_domain, file_name = get_data_from_file_name(file)
        email = email_name + '@' + email_domain
        out_file_path = './output_files/' + company_uuid + "/" + chatbot_name + "/"
        out_model_path = './output_files/' + company_uuid + "/" + chatbot_name + "/model/"

        # create directories if they don't exist
        try:
            # Create target Directory
            os.makedirs(out_file_path)
            logger.info("Directory " + out_file_path + " created ")
        except FileExistsError:
            logger.info("Directory " + out_file_path + " already exists")
        try:
            # Create target Directory
            os.makedirs(out_model_path)
            logger.info("Directory " + out_model_path + " created ")
        except FileExistsError:
            logger.info("Directory " + out_model_path + " already exists")

        out_data_file_path = out_file_path + file_name

        # copy data to output dir
        shutil.copy(file, out_data_file_path)

        if not file_name.startswith('__'):
            # train model
            logger.info(colored("Preparing Reactions...", "blue"))
            prep_synonyms(out_data_file_path)
            logger.info(colored("Reactions prepared...", "blue"))
            logger.info(colored("Training Model...", "blue"))
            faq = train_custom_model(
                out_data_file_path, 'Sentence', 'Response', out_model_path)
            faq.save()
            logger.info(colored("Training Model Completed", "blue"))

            os.remove(file)

            sleep(5)

        # start with the uploading to AWS instance
        logger.info("Uploading to AWS start")
        ip_address = get_ip_address(company_uuid, chatbot_name)
        key = paramiko.RSAKey.from_private_key_file('./configs/ChatbotSvr.pem')
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(ip_address, username='ec2-user', pkey=key)
        sftp = client.open_sftp()
        sftp.put(out_data_file_path, './data/data.csv')
        for dirpath, dirnames, filenames in os.walk(out_model_path):
            remote_path = './model/' + dirpath.replace(out_model_path, '')
            logger.info("remote_path: " + remote_path)
            # make remote directory ...
            for filename in filenames:
                local_path = os.path.join(dirpath, filename)
                logger.info("local_path: " + local_path)
                remote_filepath = os.path.join(remote_path, filename)
                logger.info("remote_filepath: " + remote_filepath)
                # put file
                sftp.put(local_path, remote_filepath)

                logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            logger.info("##############################")
        # call api to reload model
        url = "https://prodapi.emotechnologies.com/" + company_uuid +"/" + chatbot_name + "/chatbot/reload"

        payload = {}
        headers = {}

        response = requests.request("GET", url, headers=headers, data=payload)

        # send email
        logger.info("Sending email")
        if response.status_code==201:
            send_email([email], 'Model training has been completed',
                       'Model training has been completed for bot [' + chatbot_name + '] using file [' + file_name + ']')
        else:
            send_email([email], 'Error with model training',
                       'There was an error while training model for bot [' + chatbot_name + '] using file [' + file_name + ']. ' + json.dumps(response.json()))
        client.close()
        logger.info("Uploading to AWS end")
    logger.info("processing files end")


def process_zip_files():
    logger.info("processing files start")
    files_list = glob.glob("./input_files/*.zip")

    for file in files_list:
        company_uuid, chatbot_name, email_name, email_domain, file_name = get_data_from_file_name(file)
        email = email_name + '@' + email_domain
        out_file_path = './output_files/' + company_uuid + "/" + chatbot_name + "/" + file_name + "/"
        out_model_path = './output_files/' + company_uuid + "/" + chatbot_name + "/model/"

        # create directories if they don't exist
        try:
            # Create target Directory
            os.makedirs(out_file_path)
            logger.info("Directory " + out_file_path + " created ")
        except FileExistsError:
            logger.info("Directory " + out_file_path + " already exists")
        try:
            # Create target Directory
            os.makedirs(out_model_path)
            logger.info("Directory " + out_model_path + " created ")
        except FileExistsError:
            logger.info("Directory " + out_model_path + " already exists")

        # unzip data to output dir
        with zipfile.ZipFile(file, "r") as zip_ref:
            zip_ref.extractall(out_file_path)

        if not file_name.startswith('__'):
            # train model
            logger.info(colored("Preparing Reactions...", "blue"))
            data_files = glob.glob(out_file_path + "data-*.csv")
            convo_files = glob.glob(out_file_path + "convo-*.csv")
            if len(data_files) == 0:
                logger.error("No data files found")
                send_email([email], 'Error with model training',
                           'There was an error while training model for bot [' + chatbot_name + '] using file [' + file_name + ']. No data file found.')
                # remove the uploaded file
                os.remove(file)
                break
            else:
                prep_synonyms(data_files[0])
            if len(convo_files) == 0:
                logger.warning("No convo files found")
            else:
                prep_synonyms(convo_files[0])
                # join the convo to data (if convo exists)
                data_df = pd.read_csv(data_files[0])
                convo_df = pd.read_csv(convo_files[0])
                result_df = pd.concat([data_df, convo_df])
                result_df.to_csv(data_files[0])
                # write to DB


            logger.info(colored("Reactions prepared...", "blue"))
            logger.info(colored("Training Model...", "blue"))
            faq = train_custom_model(
                data_files[0], 'Sentence', 'Response', out_model_path)
            faq.save()
            logger.info(colored("Training Model Completed", "blue"))

            os.remove(file)

            sleep(5)

        # start with the uploading to AWS instance
        logger.info("Uploading to AWS start")
        ip_address = get_ip_address(company_uuid, chatbot_name)
        key = paramiko.RSAKey.from_private_key_file('./configs/ChatbotSvr.pem')
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(ip_address, username='ec2-user', pkey=key)
        sftp = client.open_sftp()
        sftp.put(data_files[0], './data/data.csv')
        if len(convo_files) > 0:
            sftp.put(convo_files[0], './data/convo.csv')
        for dirpath, dirnames, filenames in os.walk(out_model_path):
            remote_path = './model/' + dirpath.replace(out_model_path, '')
            logger.info("remote_path: " + remote_path)
            # make remote directory ...
            for filename in filenames:
                local_path = os.path.join(dirpath, filename)
                logger.info("local_path: " + local_path)
                remote_filepath = os.path.join(remote_path, filename)
                logger.info("remote_filepath: " + remote_filepath)
                # put file
                sftp.put(local_path, remote_filepath)

                logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            logger.info("##############################")
        # call api to reload model
        url = "https://prodapi.emotechnologies.com/" + company_uuid +"/" + chatbot_name + "/chatbot/reload"

        payload = {}
        headers = {}

        response = requests.request("GET", url, headers=headers, data=payload)

        # send email
        logger.info("Sending email")
        if response.status_code==201:
            send_email([email], 'Model training has been completed',
                       'Model training has been completed for bot [' + chatbot_name + '] using file [' + file_name + '].')
        else:
            send_email([email], 'Error with model training',
                       'There was an error while training model for bot [' + chatbot_name + '] using file [' + file_name + ']. ' + json.dumps(response.json()))
        client.close()
        logger.info("Uploading to AWS end")
    logger.info("processing files end")


if __name__ == '__main__':
    continue_process = True
    while continue_process:
        try:
            process_csv_files()
            process_zip_files()
            sleep(5*60)
        except KeyboardInterrupt as key_int:
            continue_process = False
            exit(0)


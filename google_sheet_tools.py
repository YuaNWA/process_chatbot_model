from __future__ import print_function

import os.path
import pickle

import httplib2shim
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from termcolor import colored


def init():
    httplib2shim.patch()
    # If modifying these scopes, delete the file token.pickle.
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

    # The ID and range of a sample spreadsheet.
    global SPREADSHEET_ID
    global SAMPLE_RANGE_NAME
    SPREADSHEET_ID = '17OF43aYLapmC25ngYOswO_J3wpWJbnQoAUiE8a-vsIQ'
    SAMPLE_RANGE_NAME = 'Response'

    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('./configs/token.pickle'):
        with open('./configs/token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                './configs/credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('./configs/token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    global service
    service = build('sheets', 'v4', credentials=creds)
    print(colored("google sheets finished init", "blue"))


def append_data(data):
    print(data)
    values = [
        [
            data["date"], data["question"], data["person"], data["response"], data["sentence"], data["intents"],
            data["objects"], data["question_intents"], data["question_objects"], data["matched_terms"]
        ]
    ]
    print(values)
    body = {
        'values': values
    }
    print(body)
    result = service.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID, range="Response",
        valueInputOption="USER_ENTERED", body=body).execute()
    print('{0} cells appended.'.format(result \
                                       .get('updates') \
                                       .get('updatedCells')))

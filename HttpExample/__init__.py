import logging

import azure.functions as func

import json
import requests
from dotenv import load_dotenv
import os

load_dotenv()

REPO_URL = "jeannefukumaru/db-registry-to-gh-actions"
GH_TOKEN = os.getenv("GH_TOKEN")   # eg. org/repo
GH_TOKEN = "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"    #github token
EVENT_NAME = "comment" #mandatory name for the event. can be dynamically provided  


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    #HTTP url
    url = "https://api.github.com/repos/" + REPO_URL + "/dispatches"

    #HTTP header
    headers = {
        'Authorization': 'Bearer ' + GH_TOKEN,
        'Content-Type': 'text/plain'
        }

    #HTTP body
    body = dict()
    body['event_type'] = EVENT_NAME
    body['client_payload'] = req.get_json()
    json_body = (json.dumps(body)).encode('utf-8')

    requests.request("POST", url, headers=headers, data=json_body)

    return func.HttpResponse(
            status_code=200
        )

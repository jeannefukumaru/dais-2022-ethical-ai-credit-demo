import requests
import json
import os

TOKEN = os.environ.get("TOKEN")
base_url = os.environ.get("BASE_URL")
event_message = json.loads(os.environ.get("event_message"))

headers = {
  'Content-Type': 'application/json',
  'Authorization': 'Bearer ' + TOKEN
}

url = base_url + "/api/2.0/mlflow/comments/create"

payload = json.dumps({
  "comment": "this version is great",
  "name": event_message['model_name'],
  "version":  event_message['version']
})

requests.request("POST", url, headers=headers, data=payload)

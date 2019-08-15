# Copyright (c) 2019 Microsoft
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json
import os

import click
import requests
from azureml.core import Model, Workspace
from azureml.core.authentication import AzureCliAuthentication


@click.command()
@click.option("-n", "--service-name", type=str,
              help="The name of the service to be tested")
@click.option("--data-path", '-d', type=str,
              default="./code/score/sample_data.json",
              help="The path to the data file to send to be tested.")
@click.option("--auth-enabled", "-a", is_flag=True)
def test_response(service_name, data_path, auth_enabled):
    cli_auth = AzureCliAuthentication()
    ws = Workspace.from_config(auth=cli_auth)

    with open(data_path, "r") as file_obj:
        data = json.load(file_obj)

    service = ws.webservices[service_name]
    headers = {}
    if auth_enabled:
        auth_key = service.get_keys()[0]
        headers = {"Authorization": "Bearer {0}".format(auth_key)}

    response = requests.post(service.scoring_uri, json=data, headers=headers)

    response_data = json.loads(response.content)
    
    assert response.status_code == 200

if __name__ == "__main__":
    test_response()

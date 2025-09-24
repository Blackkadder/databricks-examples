import requests
import os

# Set variables
dashboard_id = "xxxxxxx"
warehouse_id = "xxxxxx"

databricks_instance = "xxxxx.databricks.com"  # e.g., "adb-1234567890123456.7.azuredatabricks.net"

# OAuth variables
client_id = dbutils.secrets.get(scope="test-sp", key="client_id")
client_secret = dbutils.secrets.get(scope="test-sp", key="client_secret")
token_url = f"https://{databricks_instance}/oidc/v1/token"
url = f"https://{databricks_instance}/api/2.0/lakeview/dashboards/{dashboard_id}/published"

# Get OAuth token
data = {
    "grant_type": "client_credentials",
    "scope": "all-apis"
}
response = requests.post(token_url, data=data, auth=(client_id, client_secret))
response.raise_for_status()
service_principal_token = response.json()["access_token"]
# service_principal_token = dbutils.secrets.get(scope="test-sp", key="token")

# Publish dashboard using service principal
headers = {
    "Authorization": f"Bearer {service_principal_token}",
    "Content-Type": "application/json"
}
payload = {
    "embed_credentials": True,
    "warehouse_id": warehouse_id
}

response = requests.post(url, headers=headers, json=payload)
response.raise_for_status()
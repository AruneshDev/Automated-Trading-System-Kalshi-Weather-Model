import requests
from datetime import datetime, timedelta
import time
from typing import Any, Dict, Optional


class HttpError(Exception):
    """Represents an HTTP error with reason and status code."""

    def __init__(self, reason: str, status: int):
        super().__init__(reason)
        self.reason = reason
        self.status = status

    def __str__(self) -> str:
        return f"HttpError({self.status} {self.reason})"


class KalshiClient:
    """A client for calling authenticated Kalshi API endpoints with OAuth token."""

    def __init__(self, host: str, token: str):
        self.host = host
        self.token = token
        self.last_api_call = datetime.now()

    def rate_limit(self) -> None:
        THRESHOLD_IN_MILLISECONDS = 100
        now = datetime.now()
        threshold_in_seconds = THRESHOLD_IN_MILLISECONDS / 1000
        if (now - self.last_api_call).total_seconds() < threshold_in_seconds:
            time.sleep(threshold_in_seconds -
                       (now - self.last_api_call).total_seconds())
        self.last_api_call = datetime.now()

    def post(self, path: str, body: Dict[str, Any]) -> Any:
        self.rate_limit()
        response = requests.post(
            f"{self.host}{path}", json=body, headers=self.request_headers())
        self.raise_if_bad_response(response)
        return response.json()

    def get(self, path: str, params: Dict[str, Any] = {}) -> Any:
        self.rate_limit()
        response = requests.get(
            f"{self.host}{path}", headers=self.request_headers(), params=params)
        self.raise_if_bad_response(response)
        return response.json()

    def delete(self, path: str, params: Dict[str, Any] = {}) -> Any:
        self.rate_limit()
        response = requests.delete(
            f"{self.host}{path}", headers=self.request_headers(), params=params)
        self.raise_if_bad_response(response)
        return response.json()

    def request_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def raise_if_bad_response(self, response: requests.Response) -> None:
        if not 200 <= response.status_code < 300:
            raise HttpError(response.reason, response.status_code)

# Assuming ExchangeClient inherits from KalshiClient and overrides or extends its functionality as needed.
# Remember to replace the methods in ExchangeClient with ones specific to your application's needs,
# focusing on using the `self.token` for OAuth 2.0 authentication.


def get_google_access_token(authorization_code: str, client_id: str, client_secret: str, redirect_uri: str) -> str:
    """Exchange authorization code for access token."""
    data = {
        "code": authorization_code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    response = requests.post('https://oauth2.googleapis.com/token', data=data)
    response_data = response.json()
    return response_data['access_token']


# Example usage:
client_id = "aklal@bu.edu"
client_secret = "Sachin@007"
redirect_uri = "https://accounts.google.com/o/oauth2/auth/oauthchooseaccount?client_id=798145346451-8ju0pr7kp30vt5fokerbsnnq4sl09r38.apps.googleusercontent.com&redirect_uri=https%3A%2F%2Fdemo.kalshi.co%2Foauth%2Fgoogle%2Fcallback&response_type=code&scope=openid%20profile%20email&state=%7B%22request_mfa%22%3A%22false%22%2C%22security_token%22%3A%228v5nlafky04kovzmf6qnn%22%2C%22utm_landing%22%3A%22%2F%22%7D&service=lso&o2v=1&theme=mn&ddm=0&flowName=GeneralOAuthFlow"
# authorization_code = "OBTAINED_AUTHORIZATION_CODE_FROM_GOOGLE"
# access_token = get_google_access_token(authorization_code, client_id, client_secret, redirect_uri)

client = KalshiClient(host="https://api.kalshi.com", token=access_token)
# Do something with client, e.g., client.get("/path/to/resource")

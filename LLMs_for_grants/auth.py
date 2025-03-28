"""Authentication for the chatbot"""

import os
from dotenv import load_dotenv

def check_auth(username, password):
    """Check if the username and password are correct"""

    auth_str = map(lambda x: x.split(":"), os.getenv("AUTH_USERS").split(","))
    auth = {username: password for username, password in auth_str}
    return username in auth and password == auth[username]

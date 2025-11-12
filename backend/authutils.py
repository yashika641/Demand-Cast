# firebase_auth_utils.py

import firebase_admin
from firebase_admin import auth, credentials

# Initialize Firebase Admin app (call once at startup)
cred = credentials.Certificate(r"C:\Users\palya\Desktop\Demand-Cast\service_account_key.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)


def create_user(email: str, password: str):
    """
    Create a new Firebase user with the given email and password.
    Returns the user's UID on success.
    """
    user = auth.create_user(
        email=email,
        password=password
    )
    return user.uid


def get_user_by_email(email: str):
    """
    Fetch an existing user by their email.
    Returns a user object or raises an error if not found.
    """
    user = auth.get_user_by_email(email)
    return user


def delete_user_by_uid(uid: str):
    """
    Delete an existing user by their UID.
    """
    auth.delete_user(uid)


def verify_id_token(id_token: str):
    """
    Verify the Firebase ID token sent from the frontend.
    Returns decoded token/user info if valid, raises error if invalid.
    """
    decoded_token = auth.verify_id_token(id_token)
    return decoded_token

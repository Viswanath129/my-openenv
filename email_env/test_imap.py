import imaplib
import os
from dotenv import load_dotenv

load_dotenv()

def test_login(username, password):
    if not password:
        print(f"FAILED: {username} - No password provided")
        return
    print(f"Testing {username} with password '{password}'...")
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(username, password.replace(" ", ""))
        print(f"SUCCESS: {username}")
        mail.logout()
    except Exception as e:
        print(f"FAILED: {username} - {e}")

if __name__ == "__main__":
    u1 = os.getenv("EMAIL_1_USER")
    p1 = os.getenv("EMAIL_1_PASS")
    u2 = os.getenv("EMAIL_2_USER")
    p2 = os.getenv("EMAIL_2_PASS")
    
    test_login(u1, p1)
    test_login(u2, p2)

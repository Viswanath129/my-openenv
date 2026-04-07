"""
Live Gmail IMAP Client
- Fetches emails from the last 24 hours via IMAP4_SSL
- Returns raw email metadata (classification is done by the ML pipeline in app.py)
"""

import imaplib
import email
from email.header import decode_header
from email.utils import parsedate_to_datetime
import time
import datetime


def validate_credentials(username: str, password: str) -> tuple[bool, str]:
    """Validate IMAP credentials without fetching emails."""
    password = password.replace(" ", "")
    if not password or len(password) < 8:
        return False, "Password too short."

    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com", 993)
        mail.login(username, password)
        mail.logout()
        return True, "Success"
    except imaplib.IMAP4.error as e:
        return False, f"Auth Failed: {str(e)}"
    except Exception as e:
        return False, f"Connection error: {str(e)}"


def fetch_live_emails(username: str, password: str) -> list:
    """Fetch recent emails from a Gmail account via IMAP."""
    password = password.replace(" ", "")

    if not password or len(password) < 8:
        print(f"[IMAP] Skipping {username}: Incomplete password.")
        return []

    try:
        print(f"[IMAP] Fetching (24hr) for {username}...")
        mail = imaplib.IMAP4_SSL("imap.gmail.com", 993)
        mail.login(username, password)
        mail.select("inbox")

        lookback = datetime.timedelta(hours=24)
        threshold = datetime.datetime.now(datetime.timezone.utc) - lookback
        date_str = threshold.strftime("%d-%b-%Y")
        status, messages = mail.search(None, f'SINCE "{date_str}"')

        if status != "OK":
            mail.logout()
            return []

        email_ids = messages[0].split()
        parsed_emails = []

        for e_id in reversed(email_ids):
            if len(parsed_emails) >= 50:
                break

            res, msg = mail.fetch(e_id, "(RFC822)")
            for response in msg:
                if not isinstance(response, tuple):
                    continue

                msg_obj = email.message_from_bytes(response[1])

                # Parse date
                msg_date = None
                if msg_obj["Date"]:
                    try:
                        msg_date = parsedate_to_datetime(msg_obj["Date"])
                    except Exception:
                        pass

                if msg_date and msg_date < threshold:
                    continue

                # Decode subject safely
                subject = "No Subject"
                if msg_obj["Subject"]:
                    try:
                        subj_header = decode_header(msg_obj["Subject"])[0]
                        if isinstance(subj_header[0], bytes):
                            subject = subj_header[0].decode(
                                subj_header[1] or "utf-8", errors="ignore"
                            )
                        else:
                            subject = str(subj_header[0])
                    except Exception:
                        subject = "No Subject"

                sender = msg_obj.get("From", "Unknown")
                if len(sender) > 40:
                    sender = sender[:37] + "..."

                parsed_emails.append(
                    {
                        "id": f"{username.split('@')[0]}-{e_id.decode()}",
                        "sender": sender,
                        "subject": subject,
                        "type": "WORK",
                        "urgency": "MEDIUM",
                        "createdAt": int(msg_date.timestamp() * 1000)
                        if msg_date
                        else int(time.time() * 1000),
                        "waitingTime": 0,
                        "account": username,
                    }
                )

        mail.logout()
        print(f"[IMAP] Found {len(parsed_emails)} recent emails for {username}")
        return parsed_emails

    except Exception as e:
        print(f"[IMAP ERROR - {username}]: {e}")
        return []

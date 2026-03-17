import smtplib
import pytz
import socket
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


def send_email(email, password, smtp_host, msg):
    current_time = datetime.now(pytz.timezone("Asia/Shanghai")).strftime(
        "%Y-%m-%d-%H:%M:%S"
    )
    subject = "Alerting email"
    body = f"Current Time: {current_time}\n{msg}\nSent by: {socket.gethostname()}"

    msg = MIMEMultipart()
    msg["From"] = email
    msg["To"] = email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_host, 587) as server:
            server.starttls()
            server.login(email, password)
            server.sendmail(email, email, msg.as_string())
    except Exception as _:
        print("Error: unable to send email")


def send_emails(email, password, smtp_host, msg, send_to: list[str]):
    current_time = datetime.now(pytz.timezone("Asia/Shanghai")).strftime(
        "%Y-%m-%d-%H:%M:%S"
    )
    subject = "Alerting email"
    body = f"Current Time: {current_time}\n{msg}\nSent by: {socket.gethostname()}"

    for recipient in send_to:
        try:
            msg = MIMEMultipart()
            msg["From"] = email
            msg["To"] = recipient
            msg["Subject"] = subject

            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(smtp_host, 587) as server:
                server.starttls()
                server.login(email, password)
                server.sendmail(email, recipient, msg.as_string())
        except Exception as _:
            print(f"Error: unable to send email to {recipient}")

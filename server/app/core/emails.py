from pathlib import Path
from smtplib import SMTP
from email.header import Header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from app.config import smtp
from app.config import SMTP_USER, SMTP_PASS, SMTP_FROM_NAME, SMTP_HOST, SMTP_PORT


async def init_smtp():
    await smtp.connect()
    await smtp.starttls()
    await smtp.login(SMTP_USER, SMTP_PASS)


def read_email_template(filename: str):
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[2]  # navigate 2 levels up

    # construct the url using the pathlib `/` operator
    template_path = project_root / "app" / "templates" / "emails" / filename

    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


async def send_mail_async(to: str, subject: str, html: str):
    """Send an outgoing email with the given parameters.

    :param to: A list of recipient email addresses.
    :type to: list

    :param subject: The subject of the email.
    :type subject: str

    :param html: The text of the email.
    :type html: str
    """
    
    # TODO: this contains a bug that has 2 stages.
    # 1. If use this function as it, then automitaclly upon some hours, the smtp server gets disconnected.
    # 2. IF the SMTP is instanciated on each send_mail_async request, it freezes the server and SMTP never disconnects.

    # encode the html to utf-8 first
    html = html.encode("utf-8")

    # Prepare Message
    msg = MIMEMultipart("alternative")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = Header(SMTP_FROM_NAME, "utf-8")
    msg["To"] = to

    msg.attach(MIMEText(html, "html", "utf-8"))

    await smtp.send_message(msg)


def send_mail_sync(to: str, subject: str, html: str):
    """Send an outgoing email with the given parameters.

    :param to: A list of recipient email addresses.
    :type to: list

    :param subject: The subject of the email.
    :type subject: str

    :param html: The text of the email.
    :type html: str
    """

    sync_smtp = SMTP(SMTP_HOST, SMTP_PORT)
    sync_smtp.set_debuglevel(1) # debug purposes
    sync_smtp.connect(SMTP_HOST, SMTP_PORT)
    sync_smtp.ehlo()
    sync_smtp.starttls()
    sync_smtp.ehlo()
    sync_smtp.login(SMTP_USER, SMTP_PASS)

    html = html.encode("utf-8")

    # Prepare Message
    msg = MIMEMultipart("alternative")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = Header(SMTP_FROM_NAME, "utf-8")
    msg["To"] = to

    msg.attach(MIMEText(html, "html", "utf-8"))

    sync_smtp.send_message(msg)
    sync_smtp.quit()

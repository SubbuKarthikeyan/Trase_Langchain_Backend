"""
email_tool.py
──────────────
AgentMail SDK tool integration for Trase Bus Travel API.

Uses the official `agentmail` Python SDK to send bus route details
to user-provided email addresses.

SDK Reference (from AgentMail quickstart):
    client = AgentMail(api_key=...)
    client.inboxes.messages.send(inbox_id, to=..., subject=..., text=..., html=...)
"""

from agentmail import AgentMail
from app.core.config import settings


def send_bus_details_email(to_email: str, bus_details: str, subject: str = None) -> str:
    """
    Sends structured bus travel details to the recipient email via AgentMail SDK.

    Args:
        to_email: Target email address (e.g., 'user@example.com').
        bus_details: Plain text or formatted bus schedule & travel details.
        subject: Optional email subject line.

    Returns:
        String status report summarizing the tool call execution.
    """
    inbox_id = settings.EMAIL_SENDER
    api_key = settings.AGENTMAIL_API_KEY

    if not api_key:
        print("\n  [TOOL STATUS: ERROR - CONFIG]")
        print("     |- Detail: AGENTMAIL_API_KEY is not set in .env")
        print("=" * 60 + "\n")
        return "ERROR: AGENTMAIL_API_KEY is not configured. Cannot send email."

    if not subject:
        subject = "[Trase Bus Travel] Your Requested Bus Details & Schedule"

    print("\n" + "=" * 60)
    print(" [TOOL EXECUTION: send_bus_details_email]")
    print(f"    |- Inbox ID  : {inbox_id}")
    print(f"    |- Recipient : {to_email}")
    print(f"    |- Subject   : {subject}")
    print(f"    |- Payload   : {len(bus_details)} characters of details")
    print("=" * 60)

    # Build email body content
    text_content = (
        f"Hello,\n\n"
        f"Here are your requested bus travel details from Trase:\n\n"
        f"{bus_details}\n\n"
        f"Safe Travels,\n"
        f"Trase Travel Team"
    )
    html_content = f"""\
    <div style="font-family: Arial, sans-serif; padding: 20px; color: #333; line-height: 1.6;">
        <h2 style="color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 8px;">
            Trase Bus Travel Details
        </h2>
        <p>Hello,</p>
        <p>Thank you for inquiring with <strong>Trase AI Travel Assistant</strong>. Here are the bus details you requested:</p>
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #1a73e8; margin: 15px 0;">
            <pre style="font-family: Arial, sans-serif; white-space: pre-wrap;">{bus_details}</pre>
        </div>
        <p>If you have any questions or need further schedule details, feel free to ask!</p>
        <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;" />
        <p style="font-size: 12px; color: #777;">Sent automatically via Trase AgentMail Tool Service.</p>
    </div>
    """

    try:
        # Initialize the AgentMail SDK client (per quickstart reference)
        print(" [EmailTool] Initializing AgentMail SDK client...")
        client = AgentMail(api_key=api_key)

        # Send email using SDK method (per quickstart reference)
        print(" [EmailTool] Dispatching email via AgentMail SDK...")
        sent_message = client.inboxes.messages.send(
            inbox_id,
            to=to_email,
            subject=subject,
            text=text_content,
            html=html_content,
            labels=["trase", "bus_details"],
        )

        # Extract response details
        message_id = getattr(sent_message, "message_id", "N/A")
        thread_id = getattr(sent_message, "thread_id", "N/A")

        print("\n  [TOOL STATUS: SUCCESS]")
        print(f"     |- Message ID       : {message_id}")
        print(f"     |- Thread ID        : {thread_id}")
        print("=" * 60 + "\n")

        return f"SUCCESS: Email successfully dispatched to {to_email}. Message ID: {message_id}."

    except Exception as err:
        error_msg = str(err)
        print("\n  [TOOL STATUS: ERROR - SDK EXCEPTION]")
        print(f"     |- Exception Type  : {type(err).__name__}")
        print(f"     |- Exception Detail: {error_msg}")
        print("=" * 60 + "\n")
        return f"ERROR: Failed to send email to {to_email}. Detail: {error_msg}"

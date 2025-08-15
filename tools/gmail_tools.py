from langchain.document_loaders import UnstructuredGmailLoader

from langchain_community.tools.gmail import (
    GmailSearch,
    GmailSendMessage,
    GmailGetMessage,
    GmailCreateDraft,
    GmailGetThread,
    get_gmail_credentials,
)


gmail_tools = [
    GmailSearch(name="search_emails", description="Search for emails in Gmail"),
    GmailGetMessage(
        name="get_email", description="Get the content of a specific email"
    ),
    GmailCreateDraft(name="create_draft", description="Create a draft email"),
    GmailSendMessage(name="send_email", description="Send an email"),
    GmailGetThread(name="get_thread", description="Get an email thread"),
]

credentials = get_gmail_credentials()

# ABOUTME: Gmail integration with human-in-the-loop confirmation for sending emails
# ABOUTME: Uses StreamlitApprovalHandler callback for explicit user confirmation

import re
from typing import Any, List

from langchain.tools import tool
from langchain_community.tools.gmail import (
    GmailCreateDraft,
    GmailGetMessage,
    GmailGetThread,
    GmailSearch,
)

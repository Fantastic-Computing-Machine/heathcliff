# ABOUTME: Google People API tools for searching contacts
# ABOUTME: Allows the agent to find email addresses and phone numbers from the user's contact list

from typing import Any, List, Optional

from googleapiclient.discovery import build
from langchain.tools import tool

from logger import logger
from utils.google_auth import PEOPLE_SCOPES, get_google_credentials


def _get_people_service():
    """Get authenticated Google People API service."""
    # We use a separate token file for people to avoid conflicts or just re-use if we want unified auth
    # But the project seems to use "token.pickle" generally, but "drive_token.pickle" in comm_tools.
    # Let's use "people_token.pickle" to be safe and modular, or reuse "token.pickle" if we want one login.
    # The user deleted "token.pickle" earlier to refresh scopes.
    # If we use "token.pickle" here, and it was created without PEOPLE scopes, it will fail/re-auth.
    # Let's use "token.pickle" to encourage a single sign-on experience,
    # but we must catch the "invalid grant" or scope error if possible.
    # Given the recent conversation, the user is rebuilding token.pickle.
    creds = get_google_credentials(PEOPLE_SCOPES, token_file="token.pickle")
    return build("people", "v1", credentials=creds)


@tool
def search_contacts(query: str) -> str:
    """
    Search for a contact in the user's Google Contacts.
    Use this to find email addresses or phone numbers for a person.

    Args:
        query: The name, email, or phone number to search for.

    Returns:
        A string containing the found contacts' details (Name, Emails, Phones).
    """
    try:
        logger.debug(f"Searching contacts for: {query}")
        service = _get_people_service()

        # searchContacts allows searching across all fields
        results = (
            service.people()
            .searchContacts(query=query, readMask="names,emailAddresses,phoneNumbers")
            .execute()
        )

        connections = results.get("results", [])

        if not connections:
            return f"No contacts found matching '{query}'."

        formatted_contacts = []
        for result in connections:
            person = result.get("person", {})
            names = person.get("names", [])
            emails = person.get("emailAddresses", [])
            phones = person.get("phoneNumbers", [])

            name = names[0].get("displayName") if names else "Unknown Name"
            
            email_list = [e.get("value") for e in emails]
            phone_list = [p.get("value") for p in phones]

            contact_info = f"Name: {name}"
            if email_list:
                contact_info += f"\n  Emails: {', '.join(email_list)}"
            if phone_list:
                contact_info += f"\n  Phones: {', '.join(phone_list)}"
            
            formatted_contacts.append(contact_info)

        return "\n---\n".join(formatted_contacts)

    except Exception as e:
        logger.error(f"Error searching contacts: {e}", exc_info=True)
        return f"Error searching contacts: {str(e)}"


def get_people_tools() -> List[Any]:
    """
    Get all people tools as a list for agent registration.

    Returns:
        List of LangChain tools
    """
    return [search_contacts]

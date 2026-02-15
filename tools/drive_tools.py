# ABOUTME: Google Drive integration using LangChain tools for search and read
# ABOUTME: Provides tools for searching and reading files from Google Drive (Read-Only)

import io
from typing import Any, List

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from langchain_core.tools import Tool

from logger import logger
from utils.google_auth import DRIVE_SCOPES, get_google_credentials


def _get_drive_service():
    """Get authenticated Drive API service."""
    creds = get_google_credentials(DRIVE_SCOPES)
    return build("drive", "v3", credentials=creds)


_drive_api_resource = None


def _get_api_resource():
    global _drive_api_resource
    if not _drive_api_resource:
        _drive_api_resource = _get_drive_service()
    return _drive_api_resource


def _search_gdrive_files(query: str) -> str:
    """
    Search for files in Google Drive.

    Args:
        query: The search query string (e.g., "name contains 'report'").

    Returns:
        A formatted string listing the files found.
    """
    try:
        service = _get_api_resource()
        # Simple search implementation
        # Note: The query syntax for Drive API is slightly different than natural language.
        # We'll assume the user provides keywords and we construct a simple name search.
        # Or better, we let the user string pass through but wrap it if it's simple.

        # If the query looks like a raw drive query (contains operators), use it.
        # Otherwise, assume it's a name search.
        if "contains" in query or "=" in query:
            q = query
        else:
            # Escape single quotes to prevent injection/malformed queries
            safe_query = query.replace("'", "\\'")
            q = f"name contains '{safe_query}'"

        q += " and trashed = false"

        logger.debug(f"Searching Google Drive with query: {q}")

        results = (
            service.files()
            .list(q=q, pageSize=10, fields="nextPageToken, files(id, name, mimeType)")
            .execute()
        )
        files = results.get("files", [])

        if not files:
            return "No files found matching the query."

        result_str = "Found the following files:\n"
        for f in files:
            result_str += (
                f"- Name: {f['name']} (ID: {f['id']}, Type: {f['mimeType']})\n"
            )

        return result_str

    except Exception as e:
        logger.error(f"Error searching Google Drive: {e}", exc_info=True)
        return f"Error searching Google Drive: {str(e)}"


def _read_gdrive_file_content(file_id: str) -> str:
    """
    Read a text file from Google Drive by its ID.

    Args:
        file_id: Google Drive file ID

    Returns:
        File content as text, or an error message.
    """
    try:
        logger.debug(f"Reading Google Drive file: {file_id}")
        service = _get_api_resource()

        # Get file metadata to check mime type
        file_metadata = (
            service.files().get(fileId=file_id, fields="name, mimeType").execute()
        )
        file_name = file_metadata.get("name", "Unknown")
        mime_type = file_metadata.get("mimeType", "")
        logger.debug(f"File metadata: name={file_name}, mime_type={mime_type}")

        # Basic check for common text-like mime types
        # This is not exhaustive but covers common cases for readable files.
        if not any(
            t in mime_type
            for t in [
                "text/",
                "application/json",
                "application/xml",
                "application/msword",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ]
        ):
            logger.warning(
                f"File '{file_name}' (ID: {file_id}) is not a recognized text-like file type (type: {mime_type}). Attempting to read anyway, but content might be garbled."
            )
            # We can still attempt to read, but warn that it might not be text.

        # Download file content
        request = service.files().get_media(fileId=file_id)
        file_buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(file_buffer, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                logger.debug(f"Download progress: {int(status.progress() * 100)}%")

        file_buffer.seek(0)
        content = file_buffer.read().decode(
            "utf-8", errors="ignore"
        )  # Use ignore for non-text files
        logger.info(
            f"Successfully read Google Drive file: {file_name} ({len(content)} bytes)"
        )

        return f"File: {file_name}\n\n{content}"

    except Exception as e:
        logger.error(f"Error reading Google Drive file {file_id}: {e}", exc_info=True)
        return f"Error reading Google Drive file: {str(e)}"


def get_drive_tools() -> List[Any]:
    """
    Get all Google Drive tools as a list for agent registration.

    Returns:
        List of LangChain tools
    """
    # Ensure service is initialized
    _get_api_resource()

    tools: List[Any] = []

    # Add custom Google Drive Search Tool
    search_tool = Tool(
        name="google_drive_search",
        description="""Search for files in Google Drive.
        Args: query (str) - The search term or query (e.g. 'project plan').""",
        func=_search_gdrive_files,
    )
    tools.append(search_tool)

    # Add a custom tool for reading file content, as GoogleDriveToolkit doesn't directly expose this for arbitrary file IDs.
    read_file_tool = Tool(
        name="read_gdrive_file_content",
        description="""Read the content of a specific Google Drive file by its file ID.
        Use this after searching for a file to get its content.
        Args: file_id (str) - The ID of the file to read.""",
        func=_read_gdrive_file_content,
    )
    tools.append(read_file_tool)

    return tools

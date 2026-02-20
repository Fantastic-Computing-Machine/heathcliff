# ABOUTME: Skill management tools — load_skill for progressive disclosure,
# ABOUTME: update_master_info for live profile updates during conversation.
#
# Reference: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant

from typing import Any, List

from langchain.tools import tool

from logger import logger
from skills.skills import SKILLS, SKILLS_BY_NAME
from skills.master_info import (
    get_skill_content as _master_info_live,
    update_master_info,
)


@tool
def load_skill(skill_name: str) -> str:
    """Load a specialised skill into context for detailed guidance.

    Call this when you need detailed instructions for a specific domain.
    Only load a skill when it is actually needed — not pre-emptively.

    Available skills:
    - master_info: Adi's live personal profile, schedule, location, and preferences
    - british_persona: Tone, British English rules, and wit examples
    - email_safety: Email address rules and confirmation workflow

    Args:
        skill_name: Name of the skill to load (e.g. "master_info")

    Returns:
        Full skill content with detailed instructions, or an error message.
    """
    if skill_name == "master_info":
        # Always fetch the current live profile (not the snapshot from import time)
        content = _master_info_live()
        logger.info("[skill] Loaded live master_info")
        return f"# Skill: master_info\n\n{content}"

    skill = SKILLS_BY_NAME.get(skill_name)
    if skill:
        logger.info(f"[skill] Loaded: {skill_name}")
        return f"# Skill: {skill['name']}\n\n{skill['content']}"

    available = ", ".join(SKILLS_BY_NAME.keys())
    logger.warning(f"[skill] Not found: {skill_name!r}. Available: {available}")
    return f"Skill '{skill_name}' not found. Available skills: {available}"


def get_skill_tools() -> List[Any]:
    """Return all skill management tools for agent registration."""
    return [load_skill, update_master_info]

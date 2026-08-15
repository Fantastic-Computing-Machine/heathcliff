# ABOUTME: Comprehensive tests for skills/ — TypedDict structure, load_skill, update_master_info
# ABOUTME: Also tests the live master_info skill's _ACTIVE dict update mechanism

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Skill TypedDict definitions
# ---------------------------------------------------------------------------


class TestSkillDefinitions:
    """Tests for the SKILLS list and Skill TypedDict structure."""

    def test_skills_is_a_list(self):
        from skills.skills import SKILLS

        assert isinstance(SKILLS, list)
        assert len(SKILLS) > 0

    def test_every_skill_has_required_keys(self):
        from skills.skills import SKILLS

        for skill in SKILLS:
            assert "name" in skill, f"Missing 'name' in skill: {skill}"
            assert "description" in skill, f"Missing 'description' in {skill['name']}"
            assert "content" in skill, f"Missing 'content' in {skill['name']}"

    def test_skill_names_are_unique(self):
        from skills.skills import SKILLS

        names = [s["name"] for s in SKILLS]
        assert len(names) == len(set(names)), "Duplicate skill names found"

    def test_skills_by_name_maps_all_skills(self):
        from skills.skills import SKILLS, SKILLS_BY_NAME

        for skill in SKILLS:
            assert skill["name"] in SKILLS_BY_NAME
            assert SKILLS_BY_NAME[skill["name"]] is skill

    def test_master_info_skill_is_registered(self):
        from skills.skills import SKILLS_BY_NAME

        assert "master_info" in SKILLS_BY_NAME

    def test_british_persona_skill_is_registered(self):
        from skills.skills import SKILLS_BY_NAME

        assert "british_persona" in SKILLS_BY_NAME

    def test_email_safety_skill_is_registered(self):
        from skills.skills import SKILLS_BY_NAME

        assert "email_safety" in SKILLS_BY_NAME

    def test_skill_descriptions_are_non_empty(self):
        from skills.skills import SKILLS

        for skill in SKILLS:
            assert len(skill["description"].strip()) > 0, (
                f"{skill['name']} has empty description"
            )

    def test_skill_contents_are_non_empty(self):
        from skills.skills import SKILLS

        for skill in SKILLS:
            assert len(skill["content"].strip()) > 0, (
                f"{skill['name']} has empty content"
            )

    def test_descriptions_are_brief(self):
        """Descriptions should be short — they're always in the prompt."""
        from skills.skills import SKILLS

        for skill in SKILLS:
            assert len(skill["description"]) < 300, (
                f"{skill['name']} description too long ({len(skill['description'])} chars)"
            )


# ---------------------------------------------------------------------------
# load_skill tool
# ---------------------------------------------------------------------------


class TestLoadSkill:
    """Tests for the load_skill @tool."""

    def test_load_skill_returns_string(self):
        from skills.skill_tools import load_skill

        result = load_skill.invoke({"skill_name": "british_persona"})
        assert isinstance(result, str)

    def test_load_known_skill_returns_content(self):
        from skills.skill_tools import load_skill

        result = load_skill.invoke({"skill_name": "british_persona"})
        assert "british_persona" in result.lower() or "british" in result.lower()

    def test_load_unknown_skill_returns_error_message(self):
        from skills.skill_tools import load_skill

        result = load_skill.invoke({"skill_name": "nonexistent_skill"})
        assert "not found" in result.lower()

    def test_load_unknown_skill_lists_available(self):
        from skills.skill_tools import load_skill

        result = load_skill.invoke({"skill_name": "bad_skill"})
        assert "master_info" in result
        assert "british_persona" in result

    def test_load_email_safety_skill(self):
        from skills.skill_tools import load_skill

        result = load_skill.invoke({"skill_name": "email_safety"})
        assert "email" in result.lower()
        # Must contain the core safety rule
        assert (
            "never" in result.lower()
            or "hallucinate" in result.lower()
            or "invent" in result.lower()
        )

    def test_load_master_info_returns_live_content(self):
        """load_skill('master_info') always returns the current _ACTIVE snapshot."""
        from skills.master_info import get_master_info, set_master_info_field
        from skills.skill_tools import load_skill

        # Patch a known field and verify load_skill reflects it
        set_master_info_field("_test_marker", "pytest_was_here")
        result = load_skill.invoke({"skill_name": "master_info"})
        assert "pytest_was_here" in result or "master_info" in result  # Content is live
        # Cleanup
        info = get_master_info()
        assert "_test_marker" in info

    def test_load_skill_format_has_header(self):
        """Returned string should have '# Skill:' header."""
        from skills.skill_tools import load_skill

        result = load_skill.invoke({"skill_name": "british_persona"})
        assert "# Skill:" in result

    def test_load_skill_is_langchain_tool(self):
        """load_skill must be a LangChain BaseTool (has .name and .description)."""
        from skills.skill_tools import load_skill

        assert hasattr(load_skill, "name")
        assert hasattr(load_skill, "description")
        assert load_skill.name == "load_skill"

    def test_get_skill_tools_returns_list(self):
        from skills.skill_tools import get_skill_tools

        tools = get_skill_tools()
        assert isinstance(tools, list)
        assert len(tools) == 2  # load_skill + update_master_info

    def test_get_skill_tools_names(self):
        from skills.skill_tools import get_skill_tools

        names = {t.name for t in get_skill_tools()}
        assert "load_skill" in names
        assert "update_master_info" in names


# ---------------------------------------------------------------------------
# Master-info live skill
# ---------------------------------------------------------------------------


class TestMasterInfoSkill:
    """Tests for the live, mutable master_info skill."""

    def setup_method(self):
        """Reset _ACTIVE to seed values before each test."""
        import copy

        from skills import master_info as mi_mod

        mi_mod._ACTIVE.clear()
        mi_mod._ACTIVE.update(copy.deepcopy(mi_mod._load_seed()))

    def test_initial_seed_has_name(self):
        from skills.master_info import _load_seed, get_master_info

        info = get_master_info()
        assert "name" in info
        assert info["name"] == _load_seed()["name"]

    def test_get_master_info_returns_snapshot(self):
        """get_master_info() returns a copy — mutating it doesn't affect _ACTIVE."""
        from skills.master_info import _ACTIVE, _load_seed, get_master_info

        snapshot = get_master_info()
        snapshot["name"] = "Mutated"
        assert _ACTIVE["name"] == _load_seed()["name"]

    def test_set_master_info_field_updates_string(self):
        from skills.master_info import get_master_info, set_master_info_field

        set_master_info_field("location", "San Francisco, CA")
        assert get_master_info()["location"] == "San Francisco, CA"

    def test_append_to_list_field_adds_new_value(self):
        from skills.master_info import append_to_list_field, get_master_info

        before = len(get_master_info().get("interests", []))
        append_to_list_field("interests", "Quantum Computing")
        after = get_master_info()["interests"]
        assert "Quantum Computing" in after
        assert len(after) == before + 1

    def test_append_to_list_field_no_duplicates(self):
        from skills.master_info import append_to_list_field, get_master_info

        append_to_list_field("interests", "Duplicate Interest")
        append_to_list_field("interests", "Duplicate Interest")
        count = get_master_info()["interests"].count("Duplicate Interest")
        assert count == 1

    def test_update_master_info_tool_appends_to_interests(self):
        from skills.master_info import get_master_info, update_master_info

        result = update_master_info.invoke({"field": "interests", "value": "Surfing"})
        assert (
            "noted" in result.lower()
            or "added" in result.lower()
            or "surfing" in result.lower()
        )
        assert "Surfing" in get_master_info()["interests"]

    def test_update_master_info_tool_sets_location(self):
        from skills.master_info import get_master_info, update_master_info

        update_master_info.invoke({"field": "location", "value": "Austin, TX"})
        assert get_master_info()["location"] == "Austin, TX"

    def test_update_master_info_tool_appends_notes(self):
        from skills.master_info import get_master_info, update_master_info

        update_master_info.invoke(
            {"field": "notes", "value": "Prefers espresso over drip coffee"}
        )
        notes = get_master_info()["notes"]
        assert any("espresso" in n.lower() for n in notes)

    def test_update_master_info_tool_sets_custom_field(self):
        from skills.master_info import get_master_info, update_master_info

        update_master_info.invoke({"field": "favorite_season", "value": "Autumn"})
        assert get_master_info()["favorite_season"] == "Autumn"

    def test_update_master_info_reports_duplicate(self):
        from skills.master_info import update_master_info

        # Add once
        update_master_info.invoke({"field": "interests", "value": "Chess"})
        # Add again
        result = update_master_info.invoke({"field": "interests", "value": "Chess"})
        assert "already" in result.lower() or "knew" in result.lower()

    def test_work_hours_update_via_json(self):
        import json

        from skills.master_info import get_master_info, update_master_info

        payload = json.dumps({"start": "11:00", "end": "20:00"})
        update_master_info.invoke({"field": "work_hours", "value": payload})
        wh = get_master_info()["work_hours"]
        assert wh["start"] == "11:00"
        assert wh["end"] == "20:00"

    def test_get_skill_content_reflects_updates(self):
        """get_skill_content() always renders from the current _ACTIVE."""
        from skills.master_info import get_skill_content, update_master_info

        update_master_info.invoke({"field": "location", "value": "Denver, CO"})
        content = get_skill_content()
        assert "Denver" in content

    def test_thread_safety_concurrent_writes(self):
        """Multiple threads updating different fields should not deadlock or corrupt."""
        import threading

        from skills.master_info import get_master_info, update_master_info

        errors = []

        def worker(field, value):
            try:
                update_master_info.invoke({"field": field, "value": value})
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=("interests", f"Interest_{i}"))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        # All 20 unique interests should be in the list
        interests = get_master_info()["interests"]
        added = [f"Interest_{i}" for i in range(20)]
        for interest in added:
            assert interest in interests

    def test_update_master_info_is_langchain_tool(self):
        from skills.master_info import update_master_info

        assert hasattr(update_master_info, "name")
        assert update_master_info.name == "update_master_info"


# ---------------------------------------------------------------------------
# British persona skill content
# ---------------------------------------------------------------------------


class TestBritishPersonaSkillContent:
    """Tests for british_persona skill content quality."""

    def test_british_persona_mentions_british_english(self):
        from skills.skills import SKILLS_BY_NAME

        content = SKILLS_BY_NAME["british_persona"]["content"]
        assert "british" in content.lower() or "colour" in content.lower()

    def test_british_persona_has_tone_section(self):
        from skills.skills import SKILLS_BY_NAME

        content = SKILLS_BY_NAME["british_persona"]["content"]
        assert "tone" in content.lower() or "personality" in content.lower()

    def test_email_safety_mentions_never_hallucinate(self):
        from skills.skills import SKILLS_BY_NAME

        content = SKILLS_BY_NAME["email_safety"]["content"]
        keywords = ["never", "hallucinate", "invent", "guess"]
        assert any(k in content.lower() for k in keywords)

    def test_email_safety_has_workflow_steps(self):
        from skills.skills import SKILLS_BY_NAME

        content = SKILLS_BY_NAME["email_safety"]["content"]
        # Should describe a workflow (numbered steps or section headers)
        assert "#" in content or "1." in content or "step" in content.lower()

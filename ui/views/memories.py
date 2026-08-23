# ABOUTME: Searchable, paginated long-term memory management view.

from __future__ import annotations

from typing import Any

import streamlit as st

from ui.components import page_heading

PAGE_SIZE = 50


def _memory_rows(memory: Any, query: str, page: int) -> list[dict[str, Any]]:
    if query:
        raw = memory.recall(query, n=PAGE_SIZE)
        documents = (raw.get("documents") or [[]])[0]
        metadata = (raw.get("metadatas") or [[]])[0]
        ids = (raw.get("ids") or [[]])[0]
    else:
        raw = memory.memories.get(limit=PAGE_SIZE, offset=page * PAGE_SIZE) or {}
        documents = raw.get("documents") or []
        metadata = raw.get("metadatas") or []
        ids = raw.get("ids") or []

    return [
        {
            "id": memory_id,
            "memory": document,
            "category": meta.get("category", "general"),
            "added": str(meta.get("timestamp", "unknown"))[:19],
        }
        for document, meta, memory_id in zip(documents, metadata, ids)
    ]


def _render_pagination(page: int, row_count: int, query: str) -> None:
    if query:
        st.caption(f"Showing up to {PAGE_SIZE} matching memories.")
        return

    previous, label, next_page = st.columns([1, 3, 1])
    if previous.button("Previous 50", disabled=page == 0):
        st.session_state.memory_page = page - 1
        st.rerun()
    label.caption(f"Page {page + 1} · up to {PAGE_SIZE} active memories per page")
    if next_page.button("Next 50", disabled=row_count < PAGE_SIZE):
        st.session_state.memory_page = page + 1
        st.rerun()


def render() -> None:
    runtime = st.session_state["app_runtime"]
    memory = runtime.memory
    page_heading(
        "Knowledge",
        "Memories",
        "Active memories are always visible below. Browse 50 at a time or search them.",
    )

    with st.expander("Add a memory"):
        with st.form("add_memory", clear_on_submit=True):
            content = st.text_area(
                "Memory", height=120, placeholder="A fact or preference to retain"
            )
            category = st.selectbox(
                "Category", ["general", "preference", "fact", "reminder", "important"]
            )
            saved = st.form_submit_button("Save memory", type="primary")
        if saved:
            if content.strip():
                memory.add_memory(content.strip(), category=category)
                st.session_state.memory_page = 0
                st.success("Memory saved.")
            else:
                st.error("Enter a memory before saving.")

    query = st.text_input("Search memories", placeholder="Search by meaning or keyword")
    if query:
        st.session_state.memory_page = 0
    page = st.session_state.setdefault("memory_page", 0)
    rows = _memory_rows(memory, query, page)
    _render_pagination(page, len(rows), query)

    if not rows:
        st.info("No active memories match this page or search.")
        return

    st.dataframe(
        [{key: value for key, value in row.items() if key != "id"} for row in rows],
        hide_index=True,
        width="stretch",
        column_config={"memory": st.column_config.TextColumn("Memory", width="large")},
    )

    selected_id = st.selectbox(
        "Inspect or remove a memory",
        options=[row["id"] for row in rows],
        format_func=lambda memory_id: next(
            row["memory"][:110] for row in rows if row["id"] == memory_id
        ),
    )
    selected = next(row for row in rows if row["id"] == selected_id)
    with st.expander("Selected memory", expanded=True):
        st.write(selected["memory"])
        st.caption(f"{selected['category']} · {selected['added']}")
        if st.button("Prepare deletion", key=f"delete_prepare_{selected_id}"):
            st.session_state["delete_memory_id"] = selected_id
        if st.session_state.get("delete_memory_id") == selected_id:
            st.warning("Deleting a memory cannot be undone.")
            confirm, cancel = st.columns(2)
            if confirm.button(
                "Delete memory", type="primary", key=f"delete_{selected_id}"
            ):
                memory.delete_memory(selected_id)
                st.session_state.pop("delete_memory_id", None)
                st.rerun()
            if cancel.button("Cancel", key=f"cancel_delete_{selected_id}"):
                st.session_state.pop("delete_memory_id", None)
                st.rerun()

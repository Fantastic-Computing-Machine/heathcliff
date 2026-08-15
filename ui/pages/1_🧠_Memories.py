# ABOUTME: Streamlit multipage app - Memories management page
# ABOUTME: View, search, and add long-term memories

import os
import sys

import streamlit as st

# Add parent directory to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from config import Config
from db.memory_manager import MemoryManager

st.set_page_config(page_title="Memories", page_icon="🧠", layout="wide")


# Initialize memory
@st.cache_resource
def get_memory():
    config = Config
    return MemoryManager()


memory = get_memory()

# Header
st.title("🧠 Long-term Memories")
st.markdown(
    "Store and retrieve important information about yourself and your preferences"
)

# Layout
col1, col2 = st.columns([2, 1])

# Left column: Search and view
with col1:
    st.subheader("📚 View Memories")

    # Search memories
    search_query = st.text_input(
        "Search memories",
        placeholder="Enter search query...",
        help="Search using natural language - e.g., 'favorite music', 'work preferences'",
    )

    if search_query:
        with st.spinner("Searching..."):
            results = memory.recall(search_query, n=10)

        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            ids = results["ids"][0] if results.get("ids") else []

            st.success(f"Found {len(docs)} relevant memories")

            for i, (doc, meta, mem_id) in enumerate(zip(docs, metas, ids)):
                with st.expander(
                    f"📝 {doc[:60]}{'...' if len(doc) > 60 else ''}", expanded=(i == 0)
                ):
                    st.write(doc)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.caption(f"**Category**: {meta.get('category', 'general')}")
                    with col_b:
                        timestamp = meta.get("timestamp", "unknown")
                        if timestamp != "unknown":
                            st.caption(f"**Added**: {timestamp[:10]}")
                    if mem_id:
                        if st.button("🗑️ Delete memory", key=f"delete_memory_{mem_id}"):
                            if memory.delete_memory(mem_id):
                                st.success("Memory deleted.")
                                if hasattr(st, "rerun"):
                                    st.rerun()
                                else:
                                    st.rerun()
                            else:
                                st.error("Failed to delete memory.")
        else:
            st.info("💡 No memories found matching your search.")
    else:
        st.info("👆 Enter a search query above to find relevant memories")

        # Show all memories button
        if st.button("📋 Show All Memories"):
            all_memories = memory.memories.get(limit=300)

            if all_memories and all_memories.get("documents"):
                docs = all_memories["documents"]
                metas = (
                    all_memories["metadatas"] if all_memories.get("metadatas") else []
                )
                ids = all_memories["ids"] if all_memories.get("ids") else []
                st.write(f"**Total memories**: {len(docs)}")

                for doc, meta, mem_id in zip(docs, metas, ids):
                    with st.expander(f"📝 {doc[:60]}..."):
                        st.write(doc)
                        st.caption(f"Category: {meta.get('category', 'general')}")
                        if mem_id:
                            if st.button(
                                "🗑️ Delete memory", key=f"delete_memory_all_{mem_id}"
                            ):
                                if memory.delete_memory(mem_id):
                                    st.success("Memory deleted.")
                                    if hasattr(st, "rerun"):
                                        st.rerun()
                                    else:
                                        st.rerun()
                                else:
                                    st.error("Failed to delete memory.")
            else:
                st.info("No memories stored yet")

# Right column: Add new memory
with col2:
    st.subheader("➕ Add Memory")

    with st.form("add_memory_form", clear_on_submit=True):
        memory_text = st.text_area(
            "Memory content",
            height=150,
            placeholder="Enter something you want Heathcliff to remember...\n\nExamples:\n- I prefer coffee over tea\n- My work hours are 9-5 EST\n- I like jazz music",
            help="Add facts, preferences, or important information",
        )

        memory_category = st.selectbox(
            "Category",
            ["general", "preference", "fact", "reminder", "important", "personal"],
            help="Categorize your memory for better organization",
        )

        submit = st.form_submit_button("💾 Save Memory", width="content")

        if submit:
            if memory_text.strip():
                memory_id = memory.add_memory(
                    memory_text.strip(), category=memory_category
                )
                st.success(f"✅ Memory saved! ID: {memory_id[:8]}...")
            else:
                st.error("⚠️ Please enter memory content")

    # Quick actions
    st.markdown("---")
    st.subheader("⚡ Quick Actions")

    if st.button("📊 View Statistics", width="content"):
        stats = memory.get_stats()
        st.metric("Total Memories", stats["memories"])

        # Category breakdown
        all_mems = memory.memories.get(limit=300)
        if all_mems and all_mems.get("metadatas"):
            categories = {}
            for meta in all_mems["metadatas"]:
                cat = meta.get("category", "general")
                categories[cat] = categories.get(cat, 0) + 1

            st.write("**By Category:**")
            for cat, count in sorted(categories.items()):
                st.write(f"- {cat.capitalize()}: {count}")


# Footer
st.markdown("---")
st.caption(
    "💡 **Tip**: Be specific when adding memories - this helps Heathcliff understand you better!"
)

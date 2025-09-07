import streamlit as st
import arxiv
from transformers import pipeline

# -------------------------------
# Load summarizer (lightweight model, CPU)
# -------------------------------
@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1
    )

summarizer = load_summarizer()

# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="Research Agent", page_icon="📄", layout="wide")
st.title("📄 Research Agent - AI Paper Finder & Summarizer")

# Input from user
query = st.text_input("Enter a research topic (e.g. Quantum Computing, Cybersecurity, LLMs):")

if st.button("🔍 Search Papers"):
    if query.strip() == "":
        st.warning("Please enter a topic.")
    else:
        # -------------------------------
        # Fetch papers from Arxiv
        # -------------------------------
        st.info(f"Searching Arxiv for: **{query}** ...")
        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = list(search.results())
        
        if not results:
            st.error("No papers found. Try another topic.")
        else:
            for i, paper in enumerate(results, start=1):
                st.subheader(f"📌 Paper {i}: {paper.title}")
                st.markdown(f"**Authors:** {', '.join(a.name for a in paper.authors)}")
                st.markdown(f"**Published:** {paper.published.date()}")
                st.markdown(f"[Read Full Paper]({paper.entry_id})")

                # -------------------------------
                # Summarize abstract
                # -------------------------------
                with st.spinner("Summarizing abstract..."):
                    summary = summarizer(
                        paper.summary,
                        max_length=120,
                        min_length=40,
                        do_sample=False
                    )[0]['summary_text']

                st.markdown("**AI Summary:**")
                st.success(summary)
                st.markdown("---")

st.caption("⚡ Powered by Arxiv + HuggingFace DistilBART + Streamlit")

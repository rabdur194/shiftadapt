"""
ShiftAdapt Demo — Streamlit UI
Data-efficient & reliable adaptation of foundation models under distribution shift.
"""
import streamlit as st
import json
from pathlib import Path

st.set_page_config(
    page_title="ShiftAdapt",
    page_icon="🔄",
    layout="wide",
)

st.title("🔄 ShiftAdapt")
st.markdown(
    "**Data-efficient & reliable adaptation of pretrained/foundation models "
    "under distribution shift** (RAG + LLM)"
)

st.sidebar.header("About")
st.sidebar.info(
    "This prototype detects distribution shift, adapts a RAG knowledge base "
    "with only a few new examples, and measures reliability before vs after."
)

tab1, tab2, tab3 = st.tabs(["Run Full Pipeline", "Single Prediction", "How it works"])

with tab1:
    st.subheader("Full Adaptation Pipeline")
    st.caption("Old domain → detect shift on new domain → evaluate → adapt with few examples → re-evaluate")

    if st.button("Run Pipeline", type="primary"):
        with st.spinner("Running pipeline (embeddings + retrieval + evaluation)..."):
            try:
                from src.pipeline import run_full_pipeline
                report = run_full_pipeline()

                st.success("Pipeline completed")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Shift Detection")
                    sd = report["shift_detection"]
                    st.metric("Shifted?", "YES" if sd["shifted"] else "NO")
                    st.write(f"Centroid distance: **{sd['centroid_distance']}**")
                    st.write(f"Threshold: {sd['threshold']}")
                    st.info(sd["interpretation"])

                with col2:
                    st.markdown("### Adaptation")
                    ad = report["adaptation"]
                    st.write(ad["message"])
                    st.write(f"Method: `{ad['method']}`")
                    st.write(f"Examples added: **{ad['examples_added']}**")

                st.markdown("### Reliability: Before vs After")
                c = report["comparison"]
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", f"{c['accuracy_after']:.2%}", f"{c['accuracy_delta']:+.2%}")
                m2.metric("Avg Confidence", f"{c['confidence_after']:.2%}", f"{c['confidence_delta']:+.2%}")
                m3.metric("Improved?", "Yes" if c["improved"] else "No")

                with st.expander("Full report (JSON)"):
                    st.json(report)

            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

with tab2:
    st.subheader("Single Message Prediction")
    text = st.text_area("Paste a message to analyze", height=120,
                        placeholder="e.g. Remote job offer: pay $30 activation fee with gift cards...")
    if st.button("Predict"):
        if not text.strip():
            st.warning("Enter a message")
        else:
            with st.spinner("Retrieving + generating..."):
                from src.pipeline import predict_single
                out = predict_single(text.strip())
                ans = out["answer"]
                st.markdown(f"**Predicted label:** `{ans.get('label')}`")
                st.markdown(f"**Confidence:** {ans.get('confidence')}")
                st.markdown(f"**Explanation:** {ans.get('explanation')}")
                st.caption(f"Mode: {ans.get('mode')}")
                with st.expander("Retrieved knowledge"):
                    st.json(out["retrieved"])

with tab3:
    st.markdown("""
### What this project demonstrates

1. **Foundation model** — uses a pretrained sentence embedding model.
2. **RAG** — retrieves relevant examples from a knowledge base to ground decisions.
3. **LLM** — generates a structured label + explanation (or mock LLM if no API key).
4. **Distribution shift detection** — compares embedding centroids of old vs new data.
5. **Data-efficient adaptation** — adds only a few new labeled examples to the knowledge base (no full retraining).
6. **Reliability** — measures accuracy and confidence before vs after adaptation.

### Research alignment

Supports the proposal: *Data-Efficient and Reliable Adaptation of Pretrained/Foundation Models Under Distribution Shift*.

### Room to grow

- Better drift metrics (MMD, energy distance)
- Calibration (ECE)
- RAGAS faithfulness evaluation
- Parameter-efficient fine-tuning (LoRA) as a second adaptation method
""")

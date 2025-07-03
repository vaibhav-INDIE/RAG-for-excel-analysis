import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import google.generativeai as genai
import google.api_core.exceptions
import io
import time

# --- Page Configuration ---
st.set_page_config(page_title="🧾 Advanced Document Q&A", layout="wide")
st.title("💬 Ask Questions from a Knowledge Base of Financial Reports")
st.markdown("Upload, configure, and process multiple Excel and PDF files to build a combined knowledge base you can query.")


# --- Session State Initialization (Runs ONCE per session) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "knowledge_base_df" not in st.session_state:
    st.session_state.knowledge_base_df = pd.DataFrame()
if "api_key_configured" not in st.session_state:
    st.session_state.api_key_configured = False
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()


# --- Utility & State Management Functions ---

def reset_all():
    """
    Resets the knowledge base, processed files list, and chat history.
    This is called by the 'Clear' button.
    """
    st.session_state.knowledge_base_df = pd.DataFrame()
    st.session_state.processed_files = set()
    st.session_state.messages = []
    st.success("Knowledge Base and chat history have been cleared.")


def handle_rate_limiting(e):
    """Provides a user-friendly error message for 429 errors."""
    st.error(
        f"🚨 API Rate Limit Exceeded: {e}\n\n"
        "This is likely due to the free tier's limitations.\n"
        "**Suggestions:**\n"
        "1. **Wait a minute** and try again.\n"
        "2. Switch to a less-frequently limited model like **'gemini-1.5-flash'** in the sidebar.\n"
        "3. **Check your Google AI Platform billing details** to upgrade to a paid plan for higher limits."
    )

# --- Data Extraction & Processing Functions ---

def extract_from_pdf(file_stream, file_name):
    """Extracts text from a PDF file stream and returns a DataFrame."""
    try:
        with fitz.open(stream=file_stream, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        return pd.DataFrame([{'summary': text, 'source': file_name}])
    except Exception as e:
        st.error(f"Error reading PDF '{file_name}': {e}")
        return None

def generate_text_from_df(df, strategy, config, source_name):
    """
    Converts a DataFrame into a text-based format for embedding based on the chosen strategy.
    
    Args:
        df (pd.DataFrame): The input DataFrame from an Excel sheet.
        strategy (str): The processing strategy ('row', 'markdown', 'template').
        config (dict): Configuration for the chosen strategy.
        source_name (str): The name of the source file and sheet.

    Returns:
        pd.DataFrame: A DataFrame with 'summary' and 'source' columns.
    """
    df = df.copy().dropna(how='all') # Drop rows that are entirely empty

    if config.get('ffill', False):
        df = df.ffill()

    summaries = []
    
    if strategy == 'row':
        id_col = config.get('id_col')
        val_cols = config.get('val_cols')
        if not id_col or not val_cols:
            st.error("Error: 'Row' strategy requires an Identifier and at least one Value column.")
            return pd.DataFrame()

        for _, row in df.iterrows():
            values = [f"{col}: {row.get(col, 'N/A')}" for col in val_cols]
            row_id = row.get(id_col, "N/A")
            summary = f"Source: {source_name} | ID: {row_id} | Data: {', '.join(values)}"
            summaries.append({'summary': summary, 'source': source_name})

    elif strategy == 'markdown':
        # Convert the entire DataFrame to a single Markdown string
        summary = f"Source: {source_name}\n\n" + df.to_markdown(index=False)
        summaries.append({'summary': summary, 'source': source_name})

    elif strategy == 'template':
        template = config.get('template')
        if not template:
            st.error("Error: 'Template' strategy requires a template string.")
            return pd.DataFrame()
        
        for _, row in df.iterrows():
            try:
                summary = template.format(**row.to_dict())
                summaries.append({'summary': summary, 'source': source_name})
            except KeyError as e:
                st.warning(f"Skipping a row. Template has a placeholder '{e}' not found in the sheet's columns.")
                continue

    return pd.DataFrame(summaries)


def embed_text(df):
    """Embeds the 'summary' column of a DataFrame using Gemini."""
    model = "models/text-embedding-004"
    if df.empty:
        st.warning("No text to embed.")
        return None
        
    st.info(f"Embedding {len(df)} text chunks... this may take a moment.")
    
    all_embeddings = []
    progress_bar = st.progress(0, text="Embedding in progress...")
    
    try:
        for i, summary in enumerate(df['summary']):
            time.sleep(0.05) # Small delay to avoid hitting per-minute quotas
            embedding = genai.embed_content(model=model, content=summary)['embedding']
            all_embeddings.append(embedding)
            progress_bar.progress((i + 1) / len(df), text=f"Embedding chunk {i+1}/{len(df)}")

        df['embedding'] = all_embeddings
        progress_bar.empty()
        return df

    except google.api_core.exceptions.ResourceExhausted as e:
        handle_rate_limiting(e)
        progress_bar.empty()
        return None
    except Exception as e:
        st.error(f"Failed to embed text. Is your API key valid? Error: {e}")
        progress_bar.empty()
        return None


# --- Sidebar UI ---
with st.sidebar:
    st.header("⚙️ 1. Configuration")
    api_key = st.text_input("🔐 Enter Gemini API Key", type="password")

    if api_key:
        try:
            genai.configure(api_key=api_key)
            if not st.session_state.api_key_configured:
                st.success("API Key configured!", icon="🔑")
            st.session_state.api_key_configured = True
        except Exception as e:
            st.error(f"Invalid API Key: {e}")
            st.session_state.api_key_configured = False

    model_choice = st.selectbox(
        "🤖 Select Generative Model",
        ("gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"),
        index=0,
        disabled=not st.session_state.api_key_configured
    )
    
    st.divider()

    st.header("📄 2. Build Knowledge Base")
    uploaded_files = st.file_uploader(
        "📁 Upload Excel (.xlsx) or PDF files",
        type=['xlsx', 'xls', 'pdf'],
        accept_multiple_files=True,
        disabled=not st.session_state.api_key_configured
    )
    
    # --- New, Enhanced File Processing UI ---
    if uploaded_files:
        for file in uploaded_files:
            with st.expander(f"**Configure: {file.name}**", expanded=True):
                if file.name in st.session_state.processed_files:
                    st.success("✅ Already processed and in the knowledge base.")
                    continue

                # --- PDF Processing ---
                if file.name.lower().endswith('.pdf'):
                    if st.button(f"Process PDF: {file.name}", key=f"process_{file.name}"):
                        with st.spinner(f"Processing {file.name}..."):
                            file_bytes = file.getvalue()
                            pdf_df = extract_from_pdf(file_bytes, file.name)
                            if pdf_df is not None:
                                embedded_df = embed_text(pdf_df)
                                if embedded_df is not None:
                                    st.session_state.knowledge_base_df = pd.concat([st.session_state.knowledge_base_df, embedded_df], ignore_index=True)
                                    st.session_state.processed_files.add(file.name)
                                    st.success(f"Processed {file.name}!")
                                    st.rerun()
                # --- Excel Processing ---
                else:
                    try:
                        file_bytes = file.getvalue()
                        # --- FIX: Explicitly specify the engine for reading .xlsx files ---
                        xls = pd.ExcelFile(io.BytesIO(file_bytes), engine='openpyxl')
                        
                        # For simplicity, we configure based on the first sheet's columns.
                        first_sheet_df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
                        all_cols = first_sheet_df.columns.tolist()

                        st.markdown("##### Processing Options")
                        ffill = st.checkbox("Handle merged cells by forward-filling", value=True, key=f"ffill_{file.name}", help="Propagates the last valid observation forward to fill gaps. Ideal for tables with merged cells.")
                        
                        strategy = st.radio(
                            "Select a processing strategy for this file:",
                            ('row', 'markdown', 'template'),
                            format_func=lambda x: {
                                'row': 'One document per row (Simple Table)',
                                'markdown': 'Entire sheet as Markdown (Preserves Structure)',
                                'template': 'Custom sentence per row (Advanced)'
                            }[x],
                            key=f"strategy_{file.name}"
                        )

                        config = {'ffill': ffill}
                        if strategy == 'row':
                            config['id_col'] = st.selectbox("🆔 Identifier Column", all_cols, index=0, key=f"id_{file.name}")
                            numeric_cols = first_sheet_df.select_dtypes(include=np.number).columns.tolist()
                            config['val_cols'] = st.multiselect("📊 Value Columns", all_cols, default=numeric_cols, key=f"val_{file.name}")
                        
                        elif strategy == 'template':
                            st.write("Define a template using column names in curly braces `{}`.")
                            st.info(f"Available columns: `{', '.join(all_cols)}`")
                            example_template = "For item {ID}, the value of {ValueColumn1} is {ValueColumn1} and status is {StatusColumn}."
                            config['template'] = st.text_area("Row Template:", value=f"In Week {{{all_cols[0]}}}, the skill is {{{all_cols[1]}}} and the mandatory exercise is {{{all_cols[4]}}}.", key=f"template_{file.name}")

                        if st.button(f"Process Excel: {file.name}", key=f"process_{file.name}"):
                            with st.spinner(f"Processing sheets in {file.name}..."):
                                all_sheets_processed_df = []
                                for sheet_name in xls.sheet_names:
                                    df = pd.read_excel(xls, sheet_name=sheet_name)
                                    source_name = f"{file.name} - {sheet_name}"
                                    processed_sheet_df = generate_text_from_df(df, strategy, config, source_name)
                                    if not processed_sheet_df.empty:
                                        all_sheets_processed_df.append(processed_sheet_df)
                                
                                if all_sheets_processed_df:
                                    final_excel_df = pd.concat(all_sheets_processed_df, ignore_index=True)
                                    embedded_df = embed_text(final_excel_df)
                                    if embedded_df is not None:
                                        st.session_state.knowledge_base_df = pd.concat([st.session_state.knowledge_base_df, embedded_df], ignore_index=True)
                                        st.session_state.processed_files.add(file.name)
                                        st.success(f"Processed {file.name}!")
                                        st.rerun()

                    except Exception as e:
                        st.error(f"Error reading or configuring Excel file '{file.name}': {e}")

    st.button("🗑️ Clear Knowledge Base & Chat", on_click=reset_all, use_container_width=True, type="primary")


# --- Main Application Logic & Chat Interface ---
if not st.session_state.api_key_configured:
    st.info("👋 Welcome! Please enter your Gemini API Key in the sidebar to begin.")

if not st.session_state.knowledge_base_df.empty:
    with st.expander("🧠 View Knowledge Base Status", expanded=False):
        st.success(f"Knowledge Base is active with **{len(st.session_state.knowledge_base_df)}** embedded chunks.")
        st.write("Processed Files:")
        st.json(list(st.session_state.processed_files))
        st.dataframe(st.session_state.knowledge_base_df[['source', 'summary']].head())
elif not uploaded_files and st.session_state.api_key_configured:
     st.info("Please upload files in the sidebar to build your knowledge base.")

if not st.session_state.knowledge_base_df.empty:
    st.divider()
    st.header("💬 3. Ask Questions")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    question_emb = genai.embed_content(model="models/text-embedding-004", content=prompt)['embedding']
                    df = st.session_state.knowledge_base_df
                    df['distance'] = df['embedding'].apply(lambda x: np.dot(x, question_emb))
                    # Retrieve a few more chunks to give the model better context
                    top_k = df.sort_values("distance", ascending=False).head(7)
                    context = "\n\n---\n\n".join(top_k['summary'].values)

                    model_prompt = f"""
                    You are a helpful and precise financial analyst. Your task is to answer the user's question based *only* on the provided context.
                    The context contains chunks of text from various documents. Each chunk may start with a 'Source:' tag (e.g., 'Source: report.xlsx - Sheet1') that tells you where the information comes from. When you use information, you MUST cite the specific source.
                    If the answer is not in the context, state that you cannot find the information in the provided documents. Do not invent or infer information beyond the context.

                    CONTEXT:
                    ---
                    {context}
                    ---

                    QUESTION: {prompt}

                    ANSWER (Remember to cite your sources):
                    """
                    model = genai.GenerativeModel(model_name=model_choice)
                    response = model.generate_content(model_prompt)
                    answer = response.text
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except google.api_core.exceptions.ResourceExhausted as e:
                    handle_rate_limiting(e)
                except Exception as e:
                    error_message = f"An error occurred with the Gemini API: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
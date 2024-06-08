import streamlit as st
import google.generativeai as genai
import pandas as pd

st.set_page_config(layout="wide")

st.title("💬 Side-by-side editor")
st.caption("🚀 A streamlit dataset editor for [KoBLIMA(MathAI)](https://github.com/bckim-mathai/KoBLIMA) dataset.")

### Google API key
if "api_key" not in st.session_state:
  try:
    st.session_state.api_key = st.secrets["GOOGLE_API_KEY"]
  except:
    st.session_state.api_key = ""

if st.session_state.api_key:
  genai.configure(api_key=st.session_state.api_key)
else:
  st.warning("Your Google API Key is not provided in `.streamlit/secrets.toml`, but you can input one in the sidebar for temporary use.")
  with st.sidebar:
    st.header("Google API Key")
    st.session_state.api_key = st.text_input("Google API Key", type="password")  

### Flags
if 'is_changed' not in st.session_state:
  st.session_state.is_changed = False
  st.session_state.is_llm_init = False

if 'is_committed' not in st.session_state:
  st.session_state.is_committed = False

### Functions
def _change(is_changed=True):
  st.session_state.is_changed = is_changed
  if not is_changed:
    st.session_state.is_llm_init = False

def _commit(text1, text2, col1, col2, idx):
  st.session_state._data_df.loc[idx, col1] = '[start conversation]\n' + '[sep]'.join(text1) + '\n[end conversation]'
  st.session_state._data_df.loc[idx, col2] = '[start conversation]\n' + '[sep]'.join(text2) + '\n[end conversation]'
  st.session_state.is_committed = True
  _change(False)

def _load_csv(path:str):
  with st.spinner(f'Load `{path}`...'):
    st.session_state._data_df = pd.read_csv(path)
  st.session_state._csv_path = path

def _save_csv(path:str):
  with st.spinner(f'Save `{path}`...'):
    st.session_state._data_df.to_csv(path, index=False)
  st.session_state.is_committed = False

def stream_display(response, placeholder):
  text=''
  for chunk in response:
    if parts:=chunk.parts:
      if parts_text:=parts[0].text:
        text += parts_text
        placeholder.write(text + "▌")
  placeholder.write(text)
  return text

def _generate(model, prompt_parts, placeholder, i):
  res = model.generate_content(prompt_parts, request_options={"timeout": st.session_state.request_timeout}, stream=True)
  text = stream_display(res, placeholder)
  st.session_state.llm_res[i] = text

def _apply(key:str, value:str):
  st.session_state[key] = value
  _change(True)


if '_data_df' not in st.session_state:
  ### if the dataframe is not loaded.
  st.info("Pleas load dataset.")
  with st.sidebar:
    st.header("Column Names")
    col1_name = st.text_input("Language 1", value='en_gemini')
    st.session_state.col1_name = col1_name
    col2_name = st.text_input("Language 2", value='ko_gemini')
    st.session_state.col2_name = col2_name
    source_col_name = st.text_input("Source", value='source')
    st.session_state.source_col_name = source_col_name
    st.divider()

    st.header("Conversation")
    _start = st.text_input("Start mark", value='[start conversation]')
    st.session_state._start = _start
    _end = st.text_input("End mark", value='[end conversation]')
    st.session_state._end = _end
    _sep = st.text_input("Seperation mark", value='[sep]')
    st.session_state._sep = _sep
    st.divider()

    st.header("Data")
    _csv_path = st.text_input("CSV file", "./data/koblima.csv")

    st.button("Load", on_click=_load_csv, args=[_csv_path], type='primary', use_container_width=True)
else:
  ### if the dataframe is loaded.
  data_df = st.session_state._data_df
  col1_name = st.session_state.col1_name
  col2_name = st.session_state.col2_name
  source_col_name = st.session_state.source_col_name
  _start = st.session_state._start
  _end = st.session_state._end
  _sep = st.session_state._sep

  with st.sidebar:
    st.header("Data")
    idx = st.number_input('Index', 0, len(data_df)-1, on_change=_change, args=[False])
    st.text_input("Source", data_df.loc[idx, source_col_name])

    _csv_save_path = st.text_input("CSV file", st.session_state._csv_path)
    st.button('Save to CSV', on_click=_save_csv, args=[_csv_save_path], type='primary' if st.session_state.is_committed is True else 'secondary', use_container_width=True)

  content1 = data_df.loc[idx, col1_name]
  content1 = content1 if type(content1)==str else ''
  content1 = content1.replace(_start, '').replace(_end, '').split(_sep)
  content2 = data_df.loc[idx, col2_name]
  content2 = content2 if type(content2)==str else ''
  content2 = content2.replace(_start, '').replace(_end, '').split(_sep)
  source = data_df.loc[idx, source_col_name]

  if len(content1) != len(content2):
    content2 = ['' for s in content1]

  if not st.session_state.is_llm_init:
    st.session_state.is_llm_init = True
    st.session_state.llm_res = [''] * len(content1)

  ### Gemini
  with st.sidebar:  
    st.divider()
    st.header("Gemini")
    model_name = st.selectbox("model_name", ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"])
    
    generation_config = {
      "temperature": st.slider("temperature", min_value=0.0, max_value=1.0, value=0.2),
      "top_k": st.slider("top_k", min_value=1, value=64),
      "top_p": st.slider("top_p", min_value=0.0, max_value=1.0, value=0.95),
      "max_output_tokens": st.number_input("max_tokens", min_value=1, value=8192),
      "response_mime_type": st.selectbox("response_mime_type", ["text/plain", "application/json"]),
    }

    st.number_input("request_timeout", min_value=1, value=300, key='request_timeout')

    st.divider()
    st.header("Display")
    st.number_input("conversation area height", min_value=1, value=200, key='conv_height')
    st.number_input("instruction area height", min_value=1, value=100, key='inst_height')
    


  safety_settings = [
    {
      "category": "HARM_CATEGORY_HARASSMENT",
      "threshold": "BLOCK_NONE",
    },
    {
      "category": "HARM_CATEGORY_HATE_SPEECH",
      "threshold": "BLOCK_NONE",
    },
    {
      "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
      "threshold": "BLOCK_NONE",
    },
    {
      "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
      "threshold": "BLOCK_NONE",
    },
  ]

  model = genai.GenerativeModel(model_name=model_name,
                                generation_config=generation_config,
                                safety_settings=safety_settings)

  text_col1 = []
  text_col2 = []
  for i, (text1, text2) in enumerate(zip(content1, content2)):
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
      _text1 = st.text_area(f'`{col1_name}`({i})', value=text1.strip(), on_change=_change, height=st.session_state.conv_height, key=f"_text_col1_{i}")
      text_col1.append(_text1)
    with col2:
      _text2 = st.text_area(f'`{col2_name}`({i})', value=text2.strip(), on_change=_change, height=st.session_state.conv_height, key=f"_text_col2_{i}")
      text_col2.append(_text2)

    instruction = st.text_area(
      f"Instruction({i})",
      value=f"""Translate the following English conversation into modern and natural Korean, following rules:
* The input is {"assistant's response" if i%2 else "user's query"} of a multi-turn conversation.
* Translate the input into polite and friendly {'formal Korean (존댓말) ensuring that all original content is retained' if i%2 else 'informal Korean (반말) while maintaining their original tone and style'}.
* You can use the original terms in English.
* Specify the programming language used for long code snippets and use proper quotation mark `code` for short codes.
* Keep the markdown separation if exist.
* Keep the LaTeX equations if exist.
* Keep the references if exist.
* Just rephrase the input only.""",
      height=st.session_state.inst_height,
    )

    prompt_parts1 = [instruction, "input: " + _text1, "output: "]
    prompt_parts2 = [instruction, "input: " + _text2, "output: "]

    with st.container(border=True):
      placeholder = st.empty()
    
    col1, col2 = st.columns(2)
    with col1:
      st.button(f"Generate with `{col1_name}`({i}) as input", use_container_width=True, on_click=_generate, args=[model, prompt_parts1, placeholder, i])
      st.button(f"Apply to `{col1_name}`({i})", use_container_width=True, on_click=_apply, args=[f"_text_col1_{i}", st.session_state.llm_res[i]], disabled=not bool(st.session_state.llm_res[i]))
    with col2:
      st.button(f"Generate with `{col2_name}`({i}) as input", use_container_width=True, on_click=_generate, args=[model, prompt_parts2, placeholder, i])
      st.button(f"Apply to `{col2_name}`({i})", use_container_width=True, on_click=_apply, args=[f"_text_col2_{i}", st.session_state.llm_res[i]], disabled=not bool(st.session_state.llm_res[i]))

    placeholder.write(st.session_state.llm_res[i])

  st.button("Change", on_click=_commit, args=[text_col1, text_col2, col1_name, col2_name, idx], type='primary' if st.session_state.is_changed is True else 'secondary', use_container_width=True)
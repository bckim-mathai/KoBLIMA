from time import sleep
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
import streamlit as st

import google.generativeai as genai

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

dataset = load_dataset("GAIR/lima")

train_dataset_df = pd.DataFrame([{
  'conversations': '[start conversation]\n' + str('\n[sep]\n'.join(data['conversations'])) + '\n[end conversatioin]',
  'source': str(data['source']),
} for data in dataset['train']])

genai.configure(api_key=GOOGLE_API_KEY)

# Set up the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

def _stream_text(response):
  full_text = ''
  for chunk in response:
    try:
      full_text += chunk.text
      print(chunk.text, end='')
    except Exception as e:
      print(f'\n\n===== Exception =====')
      print(f"{chunk.prompt_feedback=}")
      print(f"{chunk.candidates[0].finish_reason=}")
      print(f"{chunk.candidates[0].safety_ratings=}")
      print(f'===== Failed. =====')
      raise e
  print('')
  return full_text

def gemini_lima(
    lima_df:pd.DataFrame,
    file_path:str='./data/lima_r.csv',
    instruction:str='',
    input_col:str='conversations',
    output_col:str='ko',
    resume:bool=False,
    first_sleep:float=10.0,
    second_sleep:float=20.0,
    verbose:bool=False,
  ) -> None:
  if resume:
    lima_df = pd.read_csv(file_path)
    df = lima_df.copy()
  else:
    assert lima_df is not None, "No LIMA dataset."
    df = lima_df.copy()
    df.loc[:, output_col] = None
  
  try:
    idx = df.loc[:, output_col].isna()
  except:
    idx = ~df.loc[:, input_col].isna()

  if verbose:
    print(f"We have {idx.sum()} conversations...\n\n")
    convs = lima_df[idx].iterrows()
  else:
    convs = tqdm(lima_df[idx].iterrows(), total=len(lima_df[idx]))

  for i, s in convs:
    text = s[input_col].strip()
    if verbose:
      print(f"====== Input text ({i}) ======\n\n{text}\n\n------ Output text ({i}) ------\n")
    try:
      prompt_parts = [
        instruction,
        "input: " + text,
        "output: ",
      ]
      if verbose:
        res = model.generate_content(prompt_parts, request_options={"timeout": 300}, stream=True)
        translated_text = _stream_text(res).strip()
      else:
        res = model.generate_content(prompt_parts, request_options={"timeout": 300})
        translated_text = res.text.strip()
    except Exception as e:
      try:
        if verbose:
          print(e)
          print(f"\n\n------ 1st try failed. ({i}) ------\n")
        sleep(second_sleep)
        if verbose:
          res = model.generate_content(prompt_parts, request_options={"timeout": 600}, stream=True)
          translated_text = _stream_text(res)
        else:
          res = model.generate_content(prompt_parts, request_options={"timeout": 600})
          translated_text = res.text.strip()
        assert len(text.split("[sep]")) == len(translated_text.split("[sep]")), "------ # of conversation turns mismatch. ------"
      except Exception as e:
        if verbose:
          print(e)
          print(f"\n\n------ 2nd try failed. Passed ({i}) ------\n")
        translated_text = ''
    if len(text.split("[sep]")) == len(translated_text.split("[sep]")):
      df.loc[i, output_col] = translated_text
      df.to_csv(file_path, index=False)
      if verbose:
        print(f"====== Success. ({i}) ======")
    else:
      if verbose:
        print(f"\n\n({i}) ### Something's wrong!!! ###\n{translated_text}")
    if not verbose:
      convs.set_postfix({'failed': sum(df.loc[:,output_col].iloc[:i].isna())})
    sleep(first_sleep)

  return df.index[df.loc[:, output_col].isna()]

instruction_en = """Rephrase the following multi-turn conversation, given in the input, using modern, natural English, and make sure to follow these rules:
* Conversation starts with `[start conversation]` and ends with `[end conversation]`.
* Each conversation turn is separated by `[sep]`.
* Rephrase the user's query (odd-numbered turns) while maintaining their original tone and style.
* Rephrase the assistant's responses (even-numbered turns) using polite and formal English, ensuring that all original content is retained.
* You can use the original terms.
* Specify the programming language used for long code snippets and use proper quotation mark `code` for short codes.
* Keep the LaTeX equations if exist.
* Keep the references if exist."""

instruction_ko = """Translate the following English multi-turn conversation into modern and natural Korean, following rules:
* Conversation starts with `[start conversation]` and ends with `[end conversation]`.
* Each conversation turn is separated by `[sep]`.
* Translate the user's query (odd-numbered turns) using friendly but polite informal Korean (반말) while maintaining their original tone and style.
* Translate the assistant's responses (even-numbered turns) using polite formal Korean (존댓말) ensuring that all original content is retained.
* You can use the original terms in English.
* Specify the programming language used for long code snippets and use proper quotation mark `code` for short codes.
* Keep the markdown separation if exist.
* Keep the LaTeX equations if exist.
* Keep the references if exist."""


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='Transform LIMA dataset with Gemini 1.5 Flash')
  parser.add_argument('-i', '--input-file', type=str, default=None, help="Input CSV file. If not specified, then use the original LIMA dataset.")
  parser.add_argument('-o', '--output-file', type=str, required=True, help="Output CSV file.")
  parser.add_argument('-m', '--mode', choices=['en', 'ko'], default='en', help="Choose 'rephrase in English' or 'translate into Korean'.")
  parser.add_argument('-ic', '--input-col', type=str, default='conversations', help="Input column name.")
  parser.add_argument('-oc', '--output-col', type=str, required=True, help="Output column name.")
  parser.add_argument('--first-sleep', type=float, default=10.0, help="How long sleep after generation completed.")
  parser.add_argument('--second-sleep', type=float, default=20.0, help="How long sleep after 1st generation failed.")
  parser.add_argument('-r', '--resume', action='store_true', help="Resume generations.")
  parser.add_argument('-v', '--verbose', action='store_true', help="Print LLM's output.")

  _args = parser.parse_args()

  args = {
    'lima_df': pd.read_csv(_args.input_file) if _args.input_file else train_dataset_df,
    'file_path': _args.output_file,
    'instruction': instruction_ko if _args.mode=='ko' else instruction_en,
    'input_col': _args.input_col,
    'output_col': _args.output_col,
    'resume': _args.resume,
    'first_sleep': _args.first_sleep,
    'second_sleep': _args.second_sleep,
    'verbose': _args.verbose,
  }
  
  failed = gemini_lima(**args)
  print(f"\n\nFailed: {', '.join([str(i) for i in failed])}")
from typing import Optional, Iterable, Literal
import pandas as pd
from os import path

class BiLIMA(object):
  def __init__(self, file_path:str, col_qa:str, col_a:Optional[str]=None, rev:bool=False):
    df = pd.read_csv(file_path)
    self._col_qa = self._to_list(df[col_qa])
    if col_a is None:
      self._col_a = None
      self._col_names = [col_qa]
    else:
      self._col_a = self._to_list(df[col_a])
      self._col_names = [col_qa, col_a]
    self._sources = df["source"].to_list()
    self.set_rev(rev)

  def set_rev(self, rev:Optional[bool]=None):
    if rev is None:
      self._rev = not self._rev
    else:
      assert isinstance(rev, bool)
      self._rev = rev
        
  def get_rev(self):
    return self._rev

  def __len__(self):
    return len(self._col_qa)

  def __getitem__(self, i):
    if self._col_a is None:
      data = self._col_qa[i]
    else:
      conv_len = len(self._col_qa[i])
      if self.get_rev():
        data = [self._col_qa[i][j] if j%2 else self._col_a[i][j] for j in range(conv_len)]
      else:
        data = [self._col_a[i][j] if j%2 else self._col_qa[i][j] for j in range(conv_len)]
    return {
      "conversation": data,
      "source": self._sources[i],
      "mode": '_'.join(self._col_names[::-1]) if self.get_rev() else '_'.join(self._col_names)
    }

  @staticmethod
  def _to_list(convs:Iterable):
    return [
      [
        s.strip()
        for s in conv.replace("[start conversation]", "").replace("[end conversation]", "").split("[sep]")
      ]
      for conv in convs
    ]

class KoBLIMA(BiLIMA):
  _file_path = path.join(path.dirname(path.abspath(__file__)), 'data/koblima.csv')
  koblima_info = {
    "en": {
      'file_path': _file_path,
      'col_qa': 'en_gemini',
    },
    "ko": {
      'file_path': _file_path,
      'col_qa': 'ko_gemini',
    },
    "en_ko": {
      'file_path': _file_path,
      'col_qa': 'en_gemini',
      'col_a': 'ko_gemini',
    },
    "ko_en": {
      'file_path': _file_path,
      'col_qa': 'en_gemini',
      'col_a': 'ko_gemini',
      'rev': True,
    },
  }

  def __init__(self, mode:Literal["en", "ko", "en_ko", "ko_en"]):
    self.mode = mode
    super().__init__(**self.koblima_info.get(mode))

  def __getitem__(self, i):
    item = super().__getitem__(i)
    item['mode'] = self.mode
    return item
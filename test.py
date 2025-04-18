# from datasets.text_data import TextData
# textData = TextData(batch_size=10,max_len=55)
# print(textData.char_to_idx)

import codecs

with codecs.open("your_file_fixed.txt", "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

print(text)  # Check if it prints correctly now

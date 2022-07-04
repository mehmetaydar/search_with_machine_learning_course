import sys
import csv
import fasttext

model = fasttext.load_model('/workspace/datasets/fasttext/title_model.bin')
top_words = '/workspace/datasets/fasttext/top_words.txt'

with open(top_words, 'r') as f:
    words = f.readlines()
    synonyms_list=[]
    for word in words:
        neighbors = model.get_nearest_neighbors(word.strip(), k=100)
        synonyms = [word.strip()]
        for n in neighbors:
            if n[0]>0.75:
                synonyms.append(n[1])
        if(len(synonyms)>1):
            synonyms_list.append(synonyms)

    with open("synonyms.csv", "w") as f:
        wr = csv.writer(f)
        wr.writerows(synonyms_list)
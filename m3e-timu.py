import time
import os
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model = SentenceTransformer('moka-ai/m3e-base', device=device)

if not os.path.exists("embedding"):
    os.mkdir("embedding")

def embedding_one(sentence):
    embeddings = model.encode([sentence], device=device)
    return embeddings[0].tolist()

def embedding_file(subject, grade, maxcnt=0):
    timu_file_name = f"timu/tm_{subject}_{grade}.tsv"
    if not os.path.exists(timu_file_name):
        return 0
    f_timu = open(timu_file_name, "r", encoding="utf-8")
    f_embedding = open(f"embedding/em_{subject}_{grade}.tsv", "w", encoding="utf-8")
    r_cnt = 0
    for line in tqdm(f_timu, mininterval=1):
        words = line.rstrip('\n').split('\t')
        if not words: 
            break
        s_id = words[0]
        sentence = words[-2]
        embedding = embedding_one(sentence)
        f_embedding.write(s_id + "\t")
        f_embedding.write("\t".join([str(e) for e in embedding]) + "\n")
        r_cnt += 1
        if maxcnt > 0 and r_cnt >= maxcnt:
            break

    f_timu.close()
    f_embedding.close()
    print("处理行数", r_cnt)
    return r_cnt


t0 = time.time()



'''
for subject in range(1, 9+1):
    for grade in range(7, 12+1):
        print(f"\n subject={subject} grade={grade}...")
        embedding_file(subject, grade)
'''

subject = 1
for grade in range(7, 12+1):
    print(f"\n subject={subject} grade={grade}...")
    embedding_file(subject, grade)

print(time.time()-t0)

import pandas as pd
import time
import re
import random
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import chromadb

subject = 1 #语文


idx_id, idx_type_code, idx_type_name, idx_difficult, idx_diffcult_code, idx_kpoint_id, idx_kpoint_name, \
    idx_article, idx_answer = range(9)
idx_embedding = 9

if not os.path.exists("test_embedding_result"):
    os.mkdir("test_embedding_result")

def read_timu(subject, grade, maxcnt=0):
    timu_file_name = f"timu/tm_{subject}_{grade}.tsv"
    embedding_file_name = f"embedding-colab/em_{subject}_{grade}.tsv"
    if not os.path.exists(timu_file_name):
        print("------ no timu file: ", timu_file_name)
        return None
    if not os.path.exists(embedding_file_name):
        print("------ no embedding file: ", embedding_file_name)
        return None

    
    list_timu = [] 
    f_timu = open(timu_file_name, "r", encoding="utf-8")
    r_cnt = 0
    err_cnt = 0
    for line in tqdm(f_timu, mininterval=1):
        words = line.rstrip('\n').split('\t')
        if not words: 
            break
        list_timu.append(words)
        
        r_cnt += 1
        if len(words) != 9:
            err_cnt += 1

        if maxcnt > 0 and r_cnt >= maxcnt:
            break
    f_timu.close()

    f_embedding = open(embedding_file_name, "r", encoding="utf-8")
    r_cnt = 0
    for line in tqdm(f_embedding, mininterval=1):
        words = line.rstrip('\n').split('\t')
        if not words: 
            break
        if words[0] != list_timu[r_cnt][0]:
            r_cnt += 1
            continue
        list_timu[r_cnt].append([float(n) for n in words[1:]])
        
        r_cnt += 1
        if maxcnt > 0 and r_cnt >= maxcnt:
            break

    f_embedding.close()
    print(f"{err_cnt=}")

    for i in range(len(list_timu)-1, -1, -1):
        timu = list_timu[i]
        if len(timu) != 10 or len(timu[-1]) != 768:
            del list_timu[i]

    print("读取行数", len(list_timu))
    return list_timu


t0 = time.time()

re_tag = re.compile(r'<.+?>')
re_space = re.compile(r'\s+')
def escape_chars(s):
    if not s:
        return ""
    s = s.replace('&nbsp;', ' ')
    s = s.replace('&ensp;', ' ')
    s = s.replace('&emsp;', ' ')
    s = re_tag.sub('', s)
    s = re_space.sub(' ', s)
    s = s.replace(',', '，')
    return s.strip()

def difficult_int(d):
    if not d:
        return 0
    return int(d*10 + 0.4)



def test_one_embedding(list_timu):
    sentences=[list_timu[0][idx_article]]
    embeddings = model.encode(sentences, device=device)

    for sentence, embedding in zip(sentences, embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding.shape, ":", embedding)  # embedding.shape # (768,)
        print("")
    print(list_timu[0][idx_embedding])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model = SentenceTransformer('moka-ai/m3e-base')
model = model.to(device)

chroma_client = chromadb.PersistentClient(path=f"./presist_timu_{subject}")

def embedding_to_db(list_timu, subject, grade, remain_cnt=0):
    '''存入向量数据库'''
    collection = chroma_client.get_or_create_collection(name=f"timu_{subject}_{grade}")

    ids = []
    embeddings = []
    metadatas = []

    r_cnt = 0
    for timu in tqdm(list_timu, mininterval=1):
        if len(timu) != 10:
            continue
        embedding = timu[-1]
        if len(embedding) != 768:
            continue
        
        ids.append(timu[0])
        embeddings.append(embedding)
        metadatas.append({
                "typecode": int(timu[idx_type_code]), 
                "type": timu[idx_type_name], 
                "difficult": float(timu[idx_difficult]), 
                "difficultcode": int(timu[idx_diffcult_code]), 
                "kpointid": int(timu[idx_kpoint_id]), 
                "kpoint": timu[idx_kpoint_name], 
                #"article": timu[idx_article], 
                #"answer": timu[idx_answer],
                })

        r_cnt += 1
        if r_cnt + remain_cnt >= len(list_timu):
            break
        
    # 批量添加，25万条记录用时几秒钟； 而如果逐条添加，用时35分钟
    time0 = time.time()
    collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)

    print("加入记录：", r_cnt, "，用时", time.time()-time0)


def embedding_query_test(list_timu, subject, grade, remain_cnt=1000):
    '''测试向量数据库'''
    time0 = time.time()
    collection = chroma_client.get_or_create_collection(name=f"timu_{subject}_{grade}")

    result_file_name = f"test_embedding_result/result_{subject}_{grade}.tsv"
    f1 = open(result_file_name, "w", encoding="utf-8")

    test_ids = []
    similar_ids = []
    distances = []
    diff_diffcults = []
    cnt_not_found = 0
    cnt_kpoint_not_match = 0

    r_cnt = len(list_timu) - remain_cnt
    for line in tqdm(range(remain_cnt), mininterval=1):
        timu = list_timu[r_cnt]
        embedding = timu[-1]
        article = list_timu[idx_article]
        difficult = float(timu[idx_difficult])

        results = collection.query(
            query_embeddings=[embedding],
            n_results=1,
        )
        if not results['ids'][0]:
            cnt_not_found += 1
            r_cnt += 1
            continue
        
        s_id = results['ids'][0][0]
        distance = results['distances'][0][0]
        s_difficult = results['metadatas'][0][0]['difficult']
        s_kpoint = results['metadatas'][0][0]['kpoint']
        if not s_difficult: s_difficult = 0

        test_ids.append(timu[0])
        similar_ids.append(s_id)
        distances.append(distance)
        diff_diffcults.append(abs(s_difficult - difficult))

        compare_result = [timu[0], s_id, distance, timu[idx_kpoint_name], s_kpoint,
            difficult, s_difficult, abs(s_difficult - difficult)]
        f1.write("\t".join([str(v) for v in compare_result]) + "\n")
      
        if timu[idx_kpoint_name] != s_kpoint:
            cnt_kpoint_not_match += 1
        r_cnt += 1

    print("cnt:", len(distances))
    print("distance:", sum(distances)/len(distances))
    print("diff:", sum(diff_diffcults)/len(diff_diffcults))
    print("不同知识点的有：", cnt_kpoint_not_match)
    print("没查找到的有：", cnt_not_found)
    f1.close()
    print("1000条查询用时：", time.time() - time0)


def embedding_text_test(list_timu, subject, grade, remain_cnt=1000):
    collection = chroma_client.get_or_create_collection(name=f"timu_{subject}_{grade}")
    cnt_not_match = 0
    time0 = time.time()
    for i in tqdm(range(100)):
        r_cnt = random.randrange(0, len(list_timu) - remain_cnt)
        timu = list_timu[r_cnt]

        article = escape_chars(timu[idx_article])
        embeddings = model.encode([article], device=device)
        embedding = embeddings[0].tolist()

        results = collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include = ["metadatas", "documents", "distances", "embeddings"],
        )
        s_id = results['ids'][0][0]
        if timu[0] != s_id: 
            cnt_not_match += 1
            del results["embeddings"]
    print("共有多少是不同的：", cnt_not_match)
    print("100条embedding并查询用时：", time.time() - time0)


time0 = time.time()
remain_cnt = 1000

for grade in range(7, 8):
    print(f"\n{subject=}, {grade=}...")

    list_timu = read_timu(subject, grade)
    if not list_timu or len(list_timu) < remain_cnt * 10:
        print(f"\n---- skip {subject=}, {grade=} -----")
        continue

    embedding_to_db(list_timu, subject, grade, remain_cnt=remain_cnt)
    embedding_query_test(list_timu, subject, grade, remain_cnt=remain_cnt)
    embedding_text_test(list_timu, subject, grade, remain_cnt=remain_cnt)

print("总用时:", time.time()-time0)

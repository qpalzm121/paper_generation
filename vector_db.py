import chromadb
import numpy as np
import random
from sentence_transformers import SentenceTransformer
import re
import torch

class VectorDB:
    def __init__(self, subject, grade, db_directory="./presist_timu_1", paper_path="./shijuan/shijuan_1.txt"):
        self.client = chromadb.PersistentClient(path=db_directory)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer('moka-ai/m3e-base', device=self.device)
        self.paper_path = paper_path
        self.subject = subject
        self.grade = grade

    def extract_id(s):
        match = re.search(r'timu_id: (\d+)', s)
        if match:
            return match.group(1)
        else:
            return None
    def query_the_same(self, embedding):
        collection_name = f"timu_{self.subject}_{self.grade}"
        collection = self.client.get_or_create_collection(collection_name)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=1,
        )
        return results['ids'][0][0]
    
    def query_similar(self, embedding, top_k=10):
        collection_name = f"timu_{self.subject}_{self.grade}"
        collection = self.client.get_or_create_collection(collection_name)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k+1,
        )
        #去掉最相似的，因为可能是同一个题目，然后从剩下的里面随机选一个
        results['ids'] = results['ids'][0][1:]
        random.shuffle(results['ids'])
        return results['ids'][0]

    def split_paper(self):
        with open(self.paper_path, 'r', encoding='utf-8') as f:
            content = f.read()
        questions = re.split(r'\d+\.', content)
        questions = [question.strip() for question in questions if question.strip() != '']
        return questions
    
    def encode_question(self, question):
        embedding = self.model.encode([question])
        return embedding[0].tolist()
    
    def query_and_replace(self, question_list):
        new_question_list = []
        for question in question_list:
            embedding = self.encode_question(question)
            similar_question_id = self.query_similar(embedding)
            new_question_list.append(similar_question_id)
        return new_question_list
    def query_thesame(self, question_list):
        new_question_list = []
        for question in question_list:
            embedding = self.encode_question(question)
            similar_question_id = self.query_the_same(embedding)
            new_question_list.append(similar_question_id)
        return new_question_list

    def vector_to_text(self, vector_ids):
        timu_file_name = f"timu/tm_{self.subject}_{self.grade}.tsv"
        with open(timu_file_name, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        question_dict = {}
        for line in lines:
            parts = line.split('\t')
            question_id = parts[0]
            question_text = parts[-2]
            question_dict[question_id] = question_text

        questions = []
        for vector_id in vector_ids:
            if vector_id in question_dict:
                questions.append((vector_id, question_dict[vector_id]))

        return questions
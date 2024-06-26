import gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import chromadb
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True).to(device)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

subject = 7
grade = 10

def vector_to_text(vector_ids):
    timu_file_name = f"timu/tm_{subject}_{grade}.tsv"
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
            questions.append(question_dict[vector_id])

    return questions

def vector_to_dk(subject, grade, vector_ids):
    timu_file_name = f"timu/tm_{subject}_{grade}.tsv"
    with open(timu_file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    question_dict = {}
    difficulty_dict = {}
    kpoint_dict = {}
    for line in lines:
        parts = line.split('\t')
        question_id = parts[0]
        question_text = parts[1]
        difficulty = parts[3]
        kpoint = parts[5]
        question_dict[question_id] = question_text
        difficulty_dict[question_id] = difficulty
        kpoint_dict[question_id] = kpoint

    questions = []
    for vector_id in vector_ids:
        if vector_id in question_dict:
            questions.append((vector_id, question_dict[vector_id], difficulty_dict[vector_id], kpoint_dict[vector_id]))

    return questions



def extract_features(vector_ids):
    promt = '帮我再生成一道根据下文延伸的题目'
    question_texts = vector_to_text(vector_ids)
    combined_text = promt + ' '.join([f'{i+1}. {text}' for i, text in enumerate(question_texts)])
    inputs = gpt2_tokenizer([combined_text], return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    outputs = gpt2_model(**inputs)
    return outputs.hidden_states[-1][0,0,:].detach().cpu().numpy().tolist()

def calculate_structure_similarity(vector_list, new_vector):
    total_similarity = sum(dot(vector, new_vector)/(norm(vector)*norm(new_vector)) for vector in vector_list)
    return total_similarity-1

def calculate_question_similarity(question1, question2):
    cos_sim = dot(question1, question2)/(norm(question1)*norm(question2))
    return cos_sim


class CustomNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int):
        super(CustomNetwork, self).__init__(observation_space, features_dim)
        self.network = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

class PaperGenerationEnv(gym.Env):

    def __init__(self, collection):
        super(PaperGenerationEnv, self).__init__()
        self.collection = collection
        self.state = None
        self.states = []
        self.metadata = []
        self.metadata_list = []
        self.ids = None
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(768,))
        self.action_space = gym.spaces.Box(low=-2, high=2, shape=(768,))
        self.reset()


    def step(self, action):
        predicted_feature = extract_features(self.ids)
        action = action.tolist()
        result = self.collection.query(query_embeddings=[action], n_results=1)
        self.ids.append(result['ids'][0][0])
        self.state = predicted_feature
        self.states.append(predicted_feature)
        self.actions.append(action)
        question = result['metadatas'][0][0]
        #self.metadata.append(question)
        difficult = float(question['difficult'])*2
        kpointid = float(question['kpointid'])/10000
        typecode = float(question['typecode'])/5

        self.metadata_list.append([difficult, kpointid, typecode])
        reward = self.calculate_reward(action)
        return self.state, reward, False, {}

    def reset(self):
        random_embedding = np.random.rand(768).tolist()
        initial_result = self.collection.query(query_embeddings=[random_embedding], n_results=1)
        self.state = random_embedding
        self.states.append(self.state)
        #self.metadata = [initial_result['metadatas']]
        question = initial_result['metadatas'][0][0]
        difficult = question['difficult']
        kpointid = question['kpointid']
        typecode = question['typecode']
        self.metadata_list.append([difficult, kpointid, typecode])
        self.ids = [initial_result['ids'][0][0]]
        self.actions.append(self.state)
        return self.state

    def get_vector_from_id(self, id):
        em_file = f'./embedding/em_{subject}_{grade}.tsv'
        with open(em_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.split('\t')
            if parts[0] == id:
                vector = [float(part) for part in parts[1:]]
                return vector
        #如果有意外返回全是1的768维度向量避免出现问题，我的测试中没有出现意外。
        return [1.0]*768

    def calculate_reward(self, action):
        result = self.collection.query(query_embeddings=[action], n_results=1)
        result_id = result['ids'][0][0]
        result_vector = self.get_vector_from_id(result_id)
        similarity = calculate_question_similarity(self.state, result_vector)
        penalty = calculate_structure_similarity(self.metadata_list, self.metadata_list[-1])
        total_reward = similarity - penalty
        return total_reward

class PaperTrainer:
    def __init__(self, collection, features_dim=768):
        self.collection = collection
        self.net = None
        self.env = DummyVecEnv([lambda: PaperGenerationEnv(self.collection)])
        self.model = PPO("MlpPolicy", self.env, policy_kwargs={"features_extractor_class": CustomNetwork, "features_extractor_kwargs": {"features_dim": features_dim}},batch_size=4)

    def train(self, total_timesteps=1000):
        print('start training')
        cnt = 0
        for i in tqdm(range(total_timesteps // 4)):
            self.model.learn(total_timesteps=4)
            if i % 4 == 0:
                if cnt<=10:
                    cnt+=1
                    self.model.save(f"model_{cnt}")
            if i%41 == 0:
                cnt+=1
                self.model.save(f"model_{cnt}")
        self.model.save("final_model")

class PaperGenerator:
    def __init__(self, collection, model_path='./model3.zip'):
        self.collection = collection
        self.env = DummyVecEnv([lambda: PaperGenerationEnv(self.collection)])
        self.model = PPO.load(model_path, env=self.env)

    def generate(self, num_questions=10):
        obs = self.env.reset()
        generated_paper = []
        for i in range(num_questions):
            action, _states = self.model.predict(obs)
            obs, rewards, dones, info = self.env.step(action)
            result = self.collection.query(query_embeddings=[np.array(action).flatten().tolist()], n_results=1)
            generated_paper.append(result['ids'][0][0])
        return generated_paper


def plot_distribution(questions, prefix="RL",model_num = 1):
    types = [int(q[1]) for q in questions]
    difficulties = [float(q[2]) for q in questions]
    kpoints = [int(q[3]) for q in questions]

    type_counter = Counter(types)

    plt.figure(figsize=(10, 6))
    plt.bar(type_counter.keys(), type_counter.values())
    plt.title('Type Distribution')
    plt.xlabel('Type')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.savefig(f'similar_images/{prefix}_{model_num}_type_distribution.png')

    plt.figure(figsize=(10, 6))
    plt.hist(difficulties, bins=10, edgecolor='black')
    plt.title('Difficulty Distribution')
    plt.xlabel('Difficulty')
    plt.ylabel('Count')
    plt.savefig(f'similar_images/{prefix}_{model_num}_difficulty_distribution.png')

    plt.figure(figsize=(10, 6))
    plt.hist(kpoints, bins=10, edgecolor='black')
    plt.title('Knowledge Point Distribution')
    plt.xlabel('Knowledge Point')
    plt.ylabel('Count')
    plt.savefig(f'similar_images/{prefix}_{model_num}_kpoint_distribution.png')

chroma_client = chromadb.PersistentClient(path=f"./presist_timu_{subject}")

collection = chroma_client.get_or_create_collection(name=f"timu_{subject}_{grade}")

generator = PaperGenerator(collection, model_path='./ckpt/model_6')
generated_paper = generator.generate()
generated_dk = vector_to_dk(subject, grade, generated_paper)
plot_distribution(generated_paper, prefix="RL",model_num = 6)

generated_questions = vector_to_text(generated_paper)

with open('./shijuan/generated_shijuan_6.txt', 'w', encoding='utf-8') as f:
    for question in generated_questions:
        f.write(question + '\n')
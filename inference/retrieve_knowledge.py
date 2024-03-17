import pandas as pd
import os

from transformers import AutoTokenizer
from transformers import AutoModel

import torch
from tqdm import tqdm
import faiss

# 加载知识库
# 定义函数来读取txt文件并分割文本
def read_and_split_txt(file_path, chunk_size=50):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# 指定文件夹路径
folder_path = r'/media/a6000/D/Workspace/ZX-GPT/ZX-GPT/rag'

# 读取文件夹下所有txt文件
file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]

# 读取并分割所有文件的内容
all_chunks = []
for file_path in file_paths:
    chunks = read_and_split_txt(file_path)
    all_chunks.extend(chunks)

# 将分割后的文本段落保存到DataFrame中
datasets = pd.DataFrame({'text': all_chunks})

# 加载分词模型
tokenizer = AutoTokenizer.from_pretrained(r'/media/a6000/D/Workspace/ZX-GPT/ZX-GPT/privateGPT/models/embedding')

# 加载模型
model = AutoModel.from_pretrained(r'/media/a6000/D/Workspace/ZX-GPT/ZX-GPT/privateGPT/models/embedding')
model = model.cuda()
model.eval()

# 将知识库中的句子转换为向量
knowledge = datasets['text'].tolist()

knowledge_vectors = []
with torch.inference_mode():
    for i in tqdm(range(0, len(knowledge), 256)):
        batch_sens = knowledge[i:i + 256]
        inputs = tokenizer(batch_sens, return_tensors='pt', padding=True, max_length=128, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        model_output = model(**inputs)
        vector = model_output[0][:, 0]
        vector = torch.nn.functional.normalize(vector, p=2, dim=1)
        knowledge_vectors.append(vector.cpu())
knowledge_vectors = torch.concat(knowledge_vectors, dim=0).numpy()

# 输入问题并进行检索
question = input("请输入问题：")

index = faiss.IndexFlatIP(384)  # 使用模型输出维度作为索引维度
faiss.normalize_L2(knowledge_vectors)
index.add(knowledge_vectors)

with torch.inference_mode():
    inputs = tokenizer(question, return_tensors="pt", padding=True, max_length=128, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    model_output = model(**inputs)
    q_vector = model_output[0][:, 0]
    q_vector = torch.nn.functional.normalize(q_vector, p=2, dim=1)
q_vector = q_vector.cpu().numpy()

scores, indexes = index.search(q_vector, 5)
topk_result = [datasets.loc[i, 'text'] for i in indexes[0].tolist()]
print(topk_result)

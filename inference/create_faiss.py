import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import faiss

def load_model_and_tokenizer(model_path, tokenizer_path):
    """ 加载模型和tokenizer """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer

def create_faiss_index(model, tokenizer, document_path, vector_size=384):
    """ 创建FAISS索引 """
    source_documents = read_documents(document_path)
    vectors = encode_documents(model, tokenizer, source_documents)
    index = faiss.IndexFlatIP(vector_size)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index, source_documents

def read_documents(document_path):
    """ 读取文档并拆分为句子 """
    source_documents = []
    with open(document_path, 'r') as f:
        for line in f:
            source_documents.extend(line.strip().split('. '))  # 根据需要调整分割逻辑
    return source_documents

def encode_documents(model, tokenizer, documents, batch_size=256, max_length=128):
    """ 将文档编码为向量 """
    vectors = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(documents), batch_size)):
            batch_sens = documents[i:i + batch_size]
            inputs = tokenizer(batch_sens, return_tensors='pt', padding=True, max_length=max_length, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            output = model(**inputs)
            vector = output[0][:, 0]
            vector = torch.nn.functional.normalize(vector, p=2, dim=1)
            vectors.append(vector)
    return torch.concat(vectors, dim=0).cpu().numpy()

def find_documents(index, source_documents, model, tokenizer, query, top_k=5):
    """ 在索引中查找文档 """
    q_vector = encode_documents(model, tokenizer, [query])
    scores, indexes = index.search(q_vector, top_k)
    # print(indexes)
    return [source_documents[i] for i in indexes[0].tolist()]

# # 示例使用
# embedding_model_path = r'/media/a6000/D/Workspace/ZX-GPT/ZX-GPT/privateGPT/models/embedding'
# embedding_tokenizer_path = r'/media/a6000/D/Workspace/ZX-GPT/ZX-GPT/privateGPT/models/embedding'
# document_path = r'/media/a6000/D/Workspace/ZX-GPT/ZX-GPT/Chinese-LLaMA-Alpaca-2-1/datasets/data/1、《中华人民共和国特种设备安全法》.txt'

# embedding_model, embedding_tokenizer = load_model_and_tokenizer(embedding_model_path, embedding_tokenizer_path)
# index, source_documents = create_faiss_index(embedding_model, embedding_tokenizer, document_path)

# # 查询
# query = "喜欢打篮球的男生喜欢什么样的女生"
# topk_results = find_documents(index, source_documents, embedding_model, embedding_tokenizer, query)
# for doc in topk_results:
#     print(doc)

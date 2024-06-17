import torch
from transformers import CodeLlamaTokenizer#AutoTokenizer

model_path = "/home/nich/CodeInjection/model_files/CodeLlama-7b-Instruct/consolidated.00.pth"
model_weights = torch.load(model_path)

tokenizer = CodeLlamaTokenizer.from_pretrained("~/CodeInjection/model_files/CodeLlama-7b-Instruct", 
                                          repo_type="local")


from transformers import AutoConfig, AutoModel, RobertaTokenizer

CTP10 = [
    'implementation',
    'dp',
    'math',
    'greedy',
    'data structures',
    'brute force',
    'geometry',
    'constructive algorithms',
    'dfs and similar',
    'strings'
]

model_config = AutoConfig.from_pretrained("google/bigbird-roberta-base", max_position_embeddings=1024)

config = {
    'seed' : 42,
    'tags' : CTP10,
    'batchSize' : 4,
    'lr' : 5e-6,
    'trainMaxLength' : 1024,
    'testMaxLength' : 1024,
    'numEpochs' : 200,
    'model' : AutoModel.from_config(model_config),
    'tokenizer' : RobertaTokenizer.from_pretrained('roberta-base'),
    'gradient_accumulation_steps' : 4,
    'max_grad_norm' : 1.0,
    'task' : 'tag',
    'lambda' : 10,
    'save' : True,
}
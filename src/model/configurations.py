from transformers import AutoConfig, AutoModel, RobertaTokenizer

tags = [
    'implementation',
    'dp',
    'math',
    'greedy',
    'datastructures',
    'bruteforce',
    'geometry',
    'constructivealgorithms',
    'dfsandsimilar',
    'strings'
]

model_config = AutoConfig.from_pretrained("google/bigbird-roberta-base", max_position_embeddings=1024)

config = {
    'seed' : 42,
    'tags' : tags,
    'batchSize' : 4,
    'lr' : 5e-6,
    'trainMaxLength' : 1024,
    'validMaxLength' : 1024,
    'numEpochs' : 100,
    'model' : AutoModel.from_config(model_config),
    'tokenizer' : RobertaTokenizer.from_pretrained('roberta-base'),
    'gradient_accumulation_steps' : 4,
    'max_grad_norm' : 1.0,
    'lambda' : 10,
    'save' : True,
}
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import torch.nn.functional as F

cos_sim = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# This model supports two prompts: "s2p_query" and "s2s_query" for sentence-to-passage and sentence-to-sentence tasks, respectively.
# They are defined in `config_sentence_transformers.json`
query_prompt_name = "s2p_query"
queries = [
    "query1"
]

model_dir = 'NovaSearch/stella_en_400M_v5'

model = SentenceTransformer(
    model_dir, 
    trust_remote_code=True, 
    config_kwargs={'use_memory_efficient_attention': False, 'unpad_inputs': False}, # Cannot use flash attention on CPU
    tokenizer_kwargs={
        'padding':'longest', 
        'truncation':True, 
        'return_tensors':'np', 
        'model_max_length': 25000 # assume that 25k is the maximum ever
    }, 
)

def tokenize_text(text:str):
    # Given a text, tokenizes it and returns the tokenizer outputs along with the span annotations

    tokenizer = model.tokenizer
    tokens = tokenizer(text, return_tensors='pt', return_offsets_mapping=True, padding=True, return_token_type_ids=True, truncation=False)
    
    # Note that BERT tokenizers put [CLS] as the first token and [SEP] as the last token
    input_ids = tokens["input_ids"][0]
    offset_mapping = tokens['offset_mapping'][0]
    attention_mask = tokens['attention_mask'][0]
    token_type_ids = tokens['token_type_ids'][0]

    return {
        'input_ids': input_ids, 
        'attention_mask': attention_mask, 
        'token_type_ids': token_type_ids, 
        'offset_mapping': offset_mapping
    }

def get_chunked_text(text:str, tokens:dict, chunk_size:int, overlap:int):
    # Refer to: https://arxiv.org/pdf/2409.04701 
    # Some notations used are as per the paper. Refer to page 5 of the paper

    m = tokens['input_ids'].shape[0]

    chunk_size = chunk_size-2

    tokenizer = model.tokenizer
        
    chunks = []
    # We start at i = 1 because first token is [CLS]. We don't want 
    # that- as we'll be tokenizing again (redundant, I know. But works for the moment )
    i = 1 
    while i < m:
        chunk_start, chunk_end = i, min(i+chunk_size, m)

        # Offset maps: [start idx of token in text, end idx of token in text]
        # Get the start and end indices of the text of the chunk from offset mapping
        text_start = tokens['offset_mapping'][chunk_start][0]

        # Edge Case: When you reach the end of the text, you want to ignore [SEP] at end
        if chunk_end == m:
            chunk_end -= 1
            
        # need "chunk_end - 1", because we're indexing exactly into the list.
        text_end =  tokens['offset_mapping'][chunk_end - 1][1] 

        chunk_text = text[text_start:text_end]

        # Tokenize the chunk_text again because we want [CLS] and [SEP] tokens
        chunk_tokens = tokenizer(chunk_text, return_tensors='pt', 
                                 return_offsets_mapping=True, 
                                 padding=True, return_token_type_ids=True, 
                                 truncation=False)
        
        chunk_object = {
            'chunk_text': chunk_text, 
            'chunk_tokens': chunk_tokens
        }

        chunks.append(chunk_object)
        i += chunk_size - overlap
    
    return chunks

def late_chunking(document:str, chunk_tokens:dict, offset_mapping, optimal_chunk_size:int=512):
    # Inputs: Take a long text (e.g.8192 tokens) 
    # Output: A tensor of size [8192//optimal_chunk_size, embedding_dimension]

    # make sure that the context window of the large model is the divisible by the optimal chunk length that we want
    print(f"Num tokens: ", chunk_tokens['input_ids'][0].shape[0])
    print("-"*80)

    # Stella Architecture is:
        # 1. Transformers
        # 2. Pooling
        # 3. Feedforward

    # When implementing late chunking, we need to change the 2. Pooling module. The rest should be same.

    # 1.
    # Get the output of the first module i.e. transformer module. 
    with torch.no_grad():
        outputs = model._first_module().auto_model(**chunk_tokens)
        raw_embeddings = outputs.last_hidden_state

        raw_embeddings = raw_embeddings[0] # get rid of the batch dim

        print(f"Raw Embeddings: ", raw_embeddings.shape) # [1, 8192, 1024]
        print("-"*80)

        print("Raw Embeddings and Chunk Tokens Shapes: ")
        print(raw_embeddings.shape[0], chunk_tokens['input_ids'][0].shape[0])

        print("-"*80)

        # 2. Pooling via Late Chunking
        outputs = [ ]

        num_tokens = chunk_tokens['input_ids'][0].shape[0]

        for start in range(0, num_tokens, optimal_chunk_size):
            end = min(start + optimal_chunk_size, num_tokens)
            chunk_token_embeddings = raw_embeddings[start:end]
            
            print("Chunk Token Embeddings: ", chunk_token_embeddings.shape)
            print("-"*80)

            # Get the corresponding attention mask, unsqueeze it to shape [512, 1], and convert to float
            mask = chunk_tokens["attention_mask"][0][start:end].unsqueeze(-1).float()  # Shape: [512, 1]
            weighted_embeddings = chunk_token_embeddings * mask  # Shape: [512, 1024]
            sum_embeddings = torch.sum(weighted_embeddings, dim=0, keepdim=True)  # Shape: [1, 1024]
            sum_mask = torch.sum(mask, dim=0, keepdim=True)  # Shape: [1, 1]
            late_chunked_embeddings = sum_embeddings / sum_mask  # Shape: [1, 1024]

            # Edge Case: If you're at the end, then you want to skip last char becoz it's [SEP]
            if end == num_tokens:
                end -= 1
            chunk_text = document[offset_mapping[0][start][0]:offset_mapping[0][end-1][1]]

            print("Late chunk shape: ", late_chunked_embeddings.shape)

            # 3. Feedforward
            # SentenceTransformer expects the pooled embeddings to be in a certain format as follow
            features = {"sentence_embedding": late_chunked_embeddings}
            
            dense_layer = model[2]
            features = dense_layer(features) 
            dense_outputs = features["sentence_embedding"]  
            
            # Do L2 normalization
            normalized_chunk_embeddings = F.normalize(dense_outputs, p=2, dim=1)

            output = normalized_chunk_embeddings[0].detach().numpy()
            outputs.append({
                'chunk_embedding': output, 
                'chunk_text': chunk_text
            })
    return outputs
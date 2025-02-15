
# Implementation of Late Chunking

In this project, I implemented the research paper: Late Chunking:Contextual Chunk Embeddings Using Long-Context Embedding Models.<sup>[1](#1)</sup>

As my understanding of the paper goes, the idea of it is as follows:

When storing embeddings of chunks of textual data in a vector database (for use cases like RAG applications), it is often more efficient and useful to store smaller chunks. While they are efficient, the information in those chunks is totally isolated from other chunks. As a result, the information *across* chunks is lost due to this process. 

On the other hand, we have long context embedding models which can take large textual documents and embed them as one chunk. As the entire document is being treated as a single chunk, there is not the issue of the information being lost across chunks. However, these long context embeddings have their own issues of inefficiency and loss of information, due to phenomena like lost in the middle.

So, late chunking offers the best of both worlds: embed using long context model, but chunk as if they are being embedded using small context models. It achieves this using the following steps:

1. Embed the entire long document using long context model.
2. Define a size that you want your chunk to have. 
3. Pool only this subset of embedding vectors to get a representation of a much smaller chunk than what was initially embedded.

Using this, we hope to maintain the entire information across chunks as well, while keeping a manageable chunk size.

## Tech Stack

1. PyTorch
2. HuggingFace: I have used *NovaSearch/stella_en_400M_v5* model as a long context model.

---

## References

<a id="1" href="https://arxiv.org/pdf/2409.04701" ><strong>1. </strong></a> Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models

2. Image Reference: [Late Chunking Article From JinaAI](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)

---

# pico-llm
A Custom Decoder-Only Transformer &amp; LLM Pre-training, trained on datasets like TinyStories and TinyShakespeare. Architecture is based on Meta's Llama-3 with minor changes on where RMS Norm is added.

### Following is the architecture of the Transformer model in this project:

1. The first layer, is the input which is converted to Embeddings (torch.nn.Embeddings) , both Positional and Token Embeddings are added
2. The next layer, multiple self-attention heads, which make the Q-K-V Matrix (with a pre-norm RMS Layer)
3. The next layer, we have a Residual connection, post which,a Feed Forward Swish layer.
4. One more residual connection, and finally add the Post Norm (RMS) and output the logits linearly.
5. The Softmax finally gives out the probabilities for the next token prediction

### Meta Llama-2 Model Architecture for reference:

<img src="/images/meta-llama-3.png" alt="Meta Llama-3 Model Architecture" width="500">

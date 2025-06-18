# Bayesian Hypervector Rule Embeddings

Bayesian Hypervector Rule Embeddings (BHRE) map discrete tokens and rules to continuous vectors using a six dimensional separable sinc kernel with compact Fourier support. A dual-channel Adaptive Discrepancy Filter (ADF) memory stores positive and negative examples to produce similarity scores when querying. This approach was demonstrated on a chess proof-of-concept but is intended for any online, explainable embedding that does not rely on backpropagation.

# ResNet50 Image Retrieval Benchmarking

This project benchmarks image retrieval using a fine-tuned ResNet50 model on a custom dataset of 451,958 images across 100 classes. The pipeline covers stratified data splitting (70/15/15), transfer learning by unfreezing layer4 with dropout regularisation, embedding extraction from the penultimate 2048-dimensional layer, and retrieval evaluation using FAISS-accelerated cosine similarity.

## Results
| Metric | Score |
|--------|-------|
| Precision@1 | 91.20% |
| Precision@5 | 90.89% |
| kNN Accuracy @21 | 91.75% |

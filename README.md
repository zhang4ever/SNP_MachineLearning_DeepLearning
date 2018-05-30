# SNP_MachineLearning_DeepLearning

## 2018-5-30 DeepBind: deep convolutional

Alipanahi B, Delong A, Weirauch M T, et al. [Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning](https://www.nature.com/articles/nbt.3300 "Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning")[J]. Nature Biotechnology, 2015, 33(8):831.

### Introduction

**DeepBind**, is based on deep convolutional neural networks and can discover new patterns even when the locations of patterns within sequences are unknown

**challenging aspects** of modern high-throughput technologies:

- Data come in qualitatively different forms
- The quantity of data is large(10,000 and 100,000 sequences)
- Each data acquisition technology has its own artifacts, biases and limitations
#### Model Layers

1. **Input:**The sequence specificities of DNA- and RNA-binding proteins

2. **1 Convolution Layer** 
  Purpose: to scan sequences for motif detectors;

3. **1 Rectification Layer:** to isolate positions with a good pattern match by shifting the response of detector

4. **1 Pooling Layer:** computes the maximum and average of each motif detector’s rectified response across the sequence

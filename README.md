# Texture Segmentation with Clustering

This project implements an **unsupervised image segmentation pipeline** for texture analysis.  
The main goal was to explore **classical computer vision** approaches without relying on large deep learning models.  
Instead, the focus is on **feature extraction**, **clustering**, and **morphological post-processing**.

## Methodology

1. **Preprocessing**  
   - Images follow the format required by the [Prague Texture Segmentation Datagenerator and Benchmark](http://mosaic.utia.cas.cz/).

2. **Feature Extraction**  
   - **Gabor filters** are applied with multiple orientations and frequencies to capture texture patterns.  
   - **Local entropy maps** are computed to describe texture complexity at a local scale.

3. **Clustering**  
   - Extracted features are reshaped into vectors.  
   - **K-Means clustering** is applied to group pixels into texture-based regions.  
   - The number of clusters was set to `6` in this experiment.

4. **Post-processing**  
   - Morphological operations (dilation + erosion) are applied to smooth the segmented regions and remove noise.

5. **Evaluation**  
   - Segmentation results were compared against ground-truth masks provided in the benchmark.  

## Benchmarking

The pipeline was tested on the **Prague Texture Segmentation Benchmark**:

- Dataset: **Grayscale [normal]**  
- Evaluation: Mean MS score reported by the benchmark server  
- **Result: 49.24**

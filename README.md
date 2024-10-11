# 🌌AstrNNPapers
🪐 Here is a collection of astronomical papers involving neural network - Version 1.0.

- The entire dataset can be viewed [here](./dataset/AstrNNPapers_V1.md).
- Download the CSV format [here](./dataset/AstrNNPapers_V1.csv).

- Table-of-Contents
    * [Data Collection](#data-collection)
    * [Categorization of Machine Learning Tasks](#categorization-of-machine-learning-tasks-mission)
    * [Categorization of Neural Network](#categorization-of-neural-network-methods)

## Data Colletion
The V1 version of the paper collection was compiled by the end of 2023. The collection process was done via [NASA ADS](https://ui.adsabs.harvard.edu/), using an abstract search with the keyword "neural network," with the years set from 01/2014 to 01/2024. The collection was filtered for astronomy and limited to refereed publications.

**Search Query:**
```bash
abs:"neural network"
year:2014-2024
collection:astronomy
property:refereed
```
The query consists of 1,240 papers sourced from various journals. The pie chart below illustrates the distribution of these papers across the different journals.

![Distribution of Papers by Journal](./figs/distribution_jornals.png)

The journals included are:

- **MNRAS**: 477 papers
- **APJ**: 276 papers
- **A&A**: 203 papers
- **PhRvD**: 106 papers
- **AJ**: 71 papers
- **APJS**: 71 papers
- **RAA**: 36 papers

🌟🌟🌟 The editor (Xingzhong Fan) reviewed each paper and selected 1,092 articles that applied neural network, categorizing each one based on '**Mission**', '**Methods**', and '**Object**'. 


## Categorization of Machine Learning Tasks (Mission)
![Categorization of Machine Learning Tasks](./figs/distribution_mission.svg)
The table below categorizes the 1,092 papers based on their machine learning tasks:
|           Task Category                                | Number of Papers |                  Explanation                        |
|:------------------------------------------------------:|:----------------:|:--------------------------------------------------:|
|           Object Detection                             |        273       | Identifying objects within multimodal data         |
|         Prediction-Regression                          |        257       | Predicting continuous values                       |
|        Object Classification                           |        184       | Classifying objects into predefined categories     |
|              Prediction                                |        82        | Predicting future outcomes/discrete values          |
|              Simulation                                |        61        | Generating synthetic data or models of reality     |
|        Statistical Inference                           |        54        | Drawing conclusions from data using statistical methods |
|        Data Reconstruction                             |        53        | Rebuilding or estimating missing data              |
|              Generation                                |        38        | Creating new data based on learned models          |
|           Data Processing                              |        25        | Preparing and transforming raw data for analysis   |
|           Image Segmentation                           |        17        | Dividing an image into meaningful regions          |
|  Feature Extraction/Dim Reduction/Clustering           |        15        | Reducing data dimensions or grouping similar data  |
|           Anomaly Detection                            |        13        | Identifying unusual patterns or outliers           |
|           Image Refinement                             |        11        | Enhancing the quality of images                    |
|           Image Restoration                            |        8         | Recovering a clean image from noisy or corrupted versions |
|              Denoising                                 |        8         | Removing noise from data or images                 |
|           Explainable AI                               |        6         | Providing insights into how and why AI models make decisions       |
|              Optimization                              |        2         | Improving efficiency or performance in tasks       |
|           Recommender System                           |        1         | Suggesting items         |
|           Model Compression                            |        2         | Reducing the size of AI models while retaining accuracy |
|           Domain Adaptation                            |        1         | Adapting models to perform well across different domains |

Some papers involve multiple deep learning tasks and are therefore counted under different categories.

## Categorization of Neural Network (Methods)
![Categorization of Methods](./figs/distribution_net.svg)
The table below categorizes the 1,092 papers based on the methods used:

|           Method                                      | Number of Papers |                  Explanation                        |
|:------------------------------------------------------:|:----------------:|:--------------------------------------------------:|
|           CNN                                         |        540       | Convolutional Neural Networks, widely used for image data  |
|           NN                                          |        339       | Neural Networks, general-purpose machine learning models |
|           Unet                                        |        59        | U-Net, primarily used for image segmentation tasks  |
|           LSTM                                        |        36        | Long Short-Term Memory, used for sequential data processing |
|           VAE                                         |        23        | Variational Autoencoders, used for generative models |
|           AE                                          |        24        | Autoencoders, used for unsupervised learning  |
|           BNN                                         |        16        | Bayesian Neural Networks, incorporating uncertainty into predictions |
|           GAN                                         |        21        | Generative Adversarial Networks, used for generative tasks |
|           PINNs                                       |        10        | Physics-Informed Neural Networks, incorporating physical laws into learning |
|           GNN                                         |        10        | Graph Neural Networks, used for graph-structured data |
|           Flow                                        |        11        | Normalizing Flows, used for generative modeling |
|           Transformer/Attention                       |        9         | Attention-based models, including Transformers |
|           RNN                                         |        10        | Recurrent Neural Networks, used for sequence modeling |
|           RL                                          |        2         | Reinforcement Learning, used for decision-making tasks in environments |
|           Diffusion                                   |        2         | Diffusion models, used for generative tasks |

Some papers involve multiple Neural Network methods and are therefore counted under different categories.

⭐⭐⭐ **For the detailed internal categorization of CNN and NN, see [here](./stats/network_details.md).**

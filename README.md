# MISR

We propose a multiplex interaction-oriented service recommendation approach (referred to as MISR) to address the cold-start problem of developing new mashups. An interaction in MISR represents an underlying relationship between a mashup and a service. The objective of MISR is to take advantage of the dominant representation learning ability of deep learning to learn hidden structures from various interactions between services and mashups. In the proposed approach, three types of interactions between services (or APIs) and mashups, including content interaction, implicit neighbor interaction, and explicit neighbor interaction, are identified and incorporated into a deep neural network (DNN), which can predict ratings of candidate services on a new mashup.

This work was supported by the National Key Research and Development Program of China under Grant No. 2017YFB1400602 and the National Science Foundation of China under Grant Nos. 61972292. For researchers who are interested in our recommendation algorithm, you can feel free to download and use it as a baseline algorithm. Also, if you think that the algorithm is useful for your work, please help cite the following paper.

Yutao Ma, Xiao Geng, and Jian Wang. A Deep Neural Network with Multiplex Interactions for Cold-Start Service Recommendation. _IEEE Transactions on Engineering Management_, DOI: 10.1109/TEM.2019.2961376, 2020.

## Requirements

1. Python version >= 3.6.5
2. Tensorflow >= 1.9.0
3. Keras >= 2.2.0
4. gensim 
5. nltk.data, nltk.corpus, and nltk.tokenize

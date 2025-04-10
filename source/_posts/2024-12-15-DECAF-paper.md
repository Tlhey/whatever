---
title: 2024-12-15-DECAF-paper
date: 2024-12-15 19:49:52
tags:
---
Thesis:
GAD inherently have imbalance data problem, normal representation learning have problem of (?), leads to bias towards minority groups, arousing fairness concern. Counterfactual fairness method helps improve.


审稿人：
1. Motivation: improve motivation
2. Synthetic dataset: consider multi-valued or continuous sensitive attributes, need multi-valued or continuous sensitive attributes
3. Evaluation: more dataset and method
4. Ablation study of loss. 
5. Proof of SCM and formula.

可以吧reddit twitter的dataset加上去，在很多GAD方法上的benchmark



Intro:
Graph data is pervasive across numerous real-world applications, including social networks, financial systems, and healthcare. Detecting anomalous patterns within these graphs, known as Graph Anomaly Detection (GAD), has become a critical task due to its implications in areas such as fraud detection, network security, and medical diagnosis. Recent advancements in GAD models, particularly those applying Graph Neural Networks (GNNs), have demonstrated impressive performance in identifying anomalies by capturing complex structural and attribute-based patterns.

However, GAD inherently faces the challenge of class imbalance, as anomalous instances are typically scarce compared to normal ones. This imbalance can hinder the model's ability to accurately detect rare anomalies, leading to high false-negative rates where actual anomalies go unnoticed. In homogeneous node classification problems, popular methods to address class imbalance include oversampling the minority class, undersampling the majority class, and employing cost-sensitive learning strategies. Similarly, traditional GAD methods tackle data imbalance through techniques such as graph embedding, Variational Autoencoders (VAEs), and Generative Adversarial Networks (GANs), which aim to learn robust representations that can effectively distinguish between normal and anomalous nodes.

Despite their effectiveness, representation learning approaches often introduce algorithmic biases against certain subpopulations within the graph. These biases can stem from the underlying data distribution, where specific groups may be underrepresented or possess distinct structural characteristics that the model inadvertently leverages to make biased predictions. Such biases raise significant fairness concerns, especially in sensitive domains like finance and healthcare. For instance, biased GAD models might disproportionately flag transactions from particular ethnic or socioeconomic groups as fraudulent in financial fraud detection systems. Similarly, in healthcare, biased anomaly detection could lead to the misidentification of treatment patterns for underrepresented populations, potentially resulting in delayed or denied care. Therefore, it is imperative to develop fair GAD methods that not only excel in anomaly detection performance but also mitigate biases to ensure equitable outcomes across all subpopulations.






My question:
1. 怎么理解class imbalance: bias (容易对少数的class产生bias？) vs. 数据中含有sensitive feature的bias
imbalance:

sens:
    (1) biases induced by one’s neighboring nodes and （NIFTY： https://proceedings.mlr.press/v161/agarwal21b）
    (2) biases induced by the causal relations from the sensitive attributes to other features as
well as the graph structure. （https://proceedings.mlr.press/v151/agarwal22b.html）




Related Works:


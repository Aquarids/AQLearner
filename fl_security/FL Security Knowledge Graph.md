# Federated Learning Security

## 1. Federated Learning Techniques

### iid

#### Gradient-Based Implementation

- **FedAvg**
  - **summary**: FedAvg is a commonly used federated learning algorithm that updates the global model by averaging gradients from clients.
  - **paper link**: [FedAvg Paper](https://arxiv.org/abs/1602.05629)
- **Adaptive Federated Optimization (FedOpt)**
  - **summary**: FedOpt is an adaptive federated optimization algorithm that combines various optimization techniques.
  - **paper link**: [FedOpt Paper](https://arxiv.org/abs/2003.00295)

#### Weight-Based Implementation

- **FedProx**
  - **summary**: FedProx is an extension of FedAvg that introduces a regularization term to handle non-i.i.d. data.
  - **paper link**: [FedProx Paper](https://arxiv.org/abs/1812.06127)
- **SCAFFOLD**
  - **summary**: SCAFFOLD reduces bias in non-i.i.d. data using control variates techniques.
  - **paper link**: [SCAFFOLD Paper](https://arxiv.org/abs/1910.06378)
- **FedNova**
  - **summary**: FedNova solves the problem of non-i.i.d. data by normalizing local updates.
  - **paper link**: [FedNova Paper](https://arxiv.org/abs/2007.07481)

#### Voting-Based Implementation

- **Federated Averaging with Voting**
  - **summary**: Combines the voting mechanism with the federated averaging algorithm to determine model updates each round.
  - **paper link**: [FedAvg with Voting Paper](https://arxiv.org/abs/2103.00102)
- **Democratic Federated Learning**
  - **summary**: Democratic federated learning achieves model updates through a democratic voting mechanism.
  - **paper link**: [Democratic FL Paper](https://arxiv.org/abs/1907.06629)

### non-iid

- **FedAvg+**
  - **summary**: FedAvg+ is an enhanced version of FedAvg that addresses challenges in non-i.i.d. data and improves communication efficiency.
  - **paper link**: [FedAvg+ Paper](https://arxiv.org/abs/2003.00295)

- **Clustered Federated Learning**
  - **summary**: Clustered FL improves federated learning by clustering clients based on data similarity and training separate models for each cluster.
  - **paper link**: [Clustered FL Paper](https://arxiv.org/abs/2006.04088)

- **Per-FedAvg**
  - **summary**: Per-FedAvg improves performance on non-i.i.d. data by personalizing models based on the global model.
  - **paper link**: [Per-FedAvg Paper](https://arxiv.org/abs/2003.13461)
- **Ditto**
  - **summary**: Ditto is a personalized federated learning algorithm that uses a personalized regularization term to adapt to different client data distributions.
  - **paper link**: [Ditto Paper](https://arxiv.org/abs/2012.04221)
- **Personalized Federated Learning with Clustered Models**
  - **summary**: This method achieves personalized federated learning by clustering clients.
  - **paper link**: [Clustered Models Paper](https://arxiv.org/abs/2105.07079)
- **pFedMe**
  - **summary**: pFedMe combines federated learning with meta-learning to provide personalized models for each client.
  - **paper link**: [pFedMe Paper](https://arxiv.org/abs/2006.08848)

- **Federated Ensemble Learning (FedEM)**
  - **summary**: FedEM improves federated learning performance by integrating multiple models through ensemble methods.
  - **paper link**: [FedEM Paper](https://arxiv.org/abs/2102.04802)

## 2. Federated Learning Security

### Cryptographic Techniques

#### MPC

- **Homomorphic Encryption (HE)**
  - **summary**: Homomorphic encryption allows computations to be performed directly on encrypted data, ensuring data privacy.
  - **paper link**: [HE Paper](https://link.springer.com/article/10.1007/s00145-015-9205-2)
- **Zero-Knowledge Proofs (ZKP)**
  - **summary**: Zero-knowledge proofs allow one party to prove possession of certain information without revealing the information itself.
  - **paper link**: [ZKP Paper](https://ieeexplore.ieee.org/document/841861)
- **Secret Sharing**
  - **summary**: Secret sharing protects privacy by dividing secret data into multiple parts.
  - **paper link**: [Secret Sharing Paper](https://dl.acm.org/doi/10.1145/359340.359342)

#### Secure Multi-Party Computation (SMC)

- **Yao's Garbled Circuits**
  - **summary**: Yao's garbled circuits are a classic method for implementing secure multi-party computation.
  - **paper link**: [Garbled Circuits Paper](https://dl.acm.org/doi/10.1145/28395.28420)
- **Oblivious Transfer**
  - **summary**: Oblivious transfer is a technique that ensures secure data transmission.
  - **paper link**: [Oblivious Transfer Paper](https://link.springer.com/article/10.1007/BF00196742)
- **Secure Function Evaluation**
  - **summary**: Secure function evaluation is a method for securely computing functions among multiple parties.
  - **paper link**: [SFE Paper](https://link.springer.com/article/10.1007/s00145-002-0206-4)

### Federated Learning Attacks

#### Data Poisoning

- **Training Stage Poisoning**
  - **Label Flip**
    - **summary**: Label flip attacks manipulate the labels of training data to degrade model performance.
    - **paper link**: [Label Flip Paper](https://dl.acm.org/doi/10.1145/3431920)
  - **Sample Injection**
    - **summary**: Sample injection attacks inject malicious samples into the training data to affect model training.
    - **paper link**: [Sample Injection Paper](https://dl.acm.org/doi/10.1145/3274694.3274740)
  - **Backdoor Attack**
    - **summary**: Backdoor attacks plant triggers in the training data to implant backdoors in the model.
    - **paper link**: [Backdoor Attack Paper](https://arxiv.org/abs/1708.06733)

#### Model Poisoning

- **summary**: Model poisoning attacks compromise the global model by uploading malicious model updates.
- **paper link**: [Model Poisoning Paper](https://dl.acm.org/doi/10.1145/3383313.3383315)
- **summary**: Model poisoning can also be achieved by introducing malicious weights into the model.
- **paper link**: [Model Poisoning Weights Paper](https://dl.acm.org/doi/10.1145/3319535.3363221)

#### Inference Attacks

- **Training Stage Attacks**
  - **Gradient-Based**
    - **Sample Reconstruction**
      - **summary**: Sample reconstruction attacks reconstruct training data by analyzing gradient information.
      - **paper link**: [Gradient-Based Reconstruction Paper](https://arxiv.org/abs/1909.13829)
    - **Model Inversion**
      - **summary**: Model inversion attacks infer the characteristics of training data by analyzing gradient information.
      - **paper link**: [Model Inversion Paper](https://dl.acm.org/doi/10.1145/2810103.2813677)
    - **Gradient Leakage**
      - **summary**: Gradient leakage attacks infer training data by intercepting and analyzing gradient data.
      - **paper link**: [Gradient Leakage Paper](https://arxiv.org/abs/2006.07198)
  - **Weight-Based**
    - **Weight Inference**
      - **summary**: Weight inference attacks infer the distribution of training data by analyzing model weights.
      - **paper link**: [Weight Inference Paper](https://arxiv.org/abs/2009.06097)
    - **Membership Inference**
      - **summary**: Membership inference attacks determine if specific data was part of the training set by analyzing model weights.
      - **paper link**: [Membership Inference Paper](https://dl.acm.org/doi/10.1145/3319535.3363214)
    - **Model Memorization**
      - **summary**: Model memorization attacks identify patterns in training data by analyzing model weights.
      - **paper link**: [Model Memorization Paper](https://dl.acm.org/doi/10.1145/3274694.3274725)
- **Inference Stage Attacks**
  - **Attribute Inference**
    - **Attribute Inference**
      - **summary**: Attribute inference attacks infer characteristics of training data by analyzing model outputs.
      - **paper link**: [Attribute Inference Paper](https://arxiv.org/abs/1802.08232)
    - **Model Extraction**
      - **summary**: Model extraction attacks reconstruct the model by querying the model and analyzing responses.
      - **paper link**: [Model Extraction Paper](https://dl.acm.org/doi/10.1145/2810103.2813657)
    - **Black-Box Inference**
      - **summary**: Black-box inference attacks infer characteristics of training data by analyzing only the model outputs.
      - **paper link**: [Black-Box Inference Paper](https://arxiv.org/abs/1905.05113)

### Federated Learning Defenses

#### Based on Secure Multi-Party Computation

- **Homomorphic Encryption**
  - **summary**: Methods based on homomorphic encryption protect privacy by allowing computations on encrypted data.
  - **paper link**: [HE Defense Paper](https://link.springer.com/article/10.1007/s00145-015-9205-2)
  - **Homomorphic Encryption for Federated Learning**
    - **summary**: Using homomorphic encryption to protect data privacy in federated learning.
    - **paper link**: [HE for FL Paper](https://ieeexplore.ieee.org/document/8844632)
  - **Secure Aggregation with Homomorphic Encryption**
    - **summary**: Using homomorphic encryption for secure aggregation in federated learning.
    - **paper link**: [Secure Aggregation Paper](https://dl.acm.org/doi/10.1145/3321705.3329845)

- **Differential Privacy**
  - **summary**: Methods based on differential privacy protect data privacy by adding noise.
  - **paper link**: [DP Defense Paper](https://dl.acm.org/doi/10.1145/3319535.3363222)
  - **Differential Privacy for Federated Learning**
    - **summary**: Application of differential privacy techniques in federated learning.
    - **paper link**: [DP for FL Paper](https://arxiv.org/abs/1712.07557)
  - **Improved Privacy with Differential Privacy**
    - **summary**: Improved differential privacy techniques enhance privacy protection in federated learning.
    - **paper link**: [Improved DP Paper](https://dl.acm.org/doi/10.1145/3411504.3421221)

- **Secret Sharing**
  - **summary**: Methods based on secret sharing protect privacy by distributing data among multiple parties.
  - **paper link**: [Secret Sharing Defense Paper](https://dl.acm.org/doi/10.1145/359340.359342)
  - **Secure Federated Learning using Secret Sharing**
    - **summary**: Using secret sharing to protect data privacy in federated learning.
    - **paper link**: [Secret Sharing for FL Paper](https://ieeexplore.ieee.org/document/8713993)
  - **Efficient Secret Sharing for Federated Learning**
    - **summary**: Enhancing efficiency of secret sharing in federated learning.
    - **paper link**: [Efficient Secret Sharing Paper](https://arxiv.org/abs/2008.08489)

#### Based on Robust Aggregation

- **summary**: Methods based on robust aggregation use robust aggregation algorithms to resist malicious data interference.
- **paper link**: [Robust Aggregation Defense Paper](https://dl.acm.org/doi/10.1145/3394486.3403360)
- **Robust Aggregation for Federated Learning**
  - **summary**: Using robust aggregation algorithms to enhance security in federated learning.
  - **paper link**: [Robust Aggregation for FL Paper](https://arxiv.org/abs/1909.06320)
- **Krum: Robust Aggregation Technique**
  - **summary**: The Krum algorithm enhances robustness of aggregation by selecting trusted updates.
  - **paper link**: [Krum Paper](https://dl.acm.org/doi/10.1145/3052973.3053006)

#### Based on Adversarial Training

- **summary**: Methods based on adversarial training enhance model robustness by generating adversarial examples to defend against adversarial attacks.
- **paper link**: [Adversarial Training Defense Paper](https://arxiv.org/abs/1705.07204)
- **Adversarial Training for Federated Learning**
  - **summary**: Using adversarial training to enhance model robustness in federated learning.
  - **paper link**: [Adversarial Training for FL Paper](https://arxiv.org/abs/2008.07920)
- **Improving Robustness with Adversarial Training**
  - **summary**: Improving model robustness in federated learning through adversarial training.
  - **paper link**: [Improving Robustness Paper](https://dl.acm.org/doi/10.1145/3351099.3351101)

#### Based on Defense Mechanism Enhancement

- **summary**: Methods based on defense mechanism enhancement protect federated learning security by combining multiple defense techniques.
- **paper link**: [Defense Mechanism Enhancement Paper](https://arxiv.org/abs/2103.09328)
- **Hybrid Defense Mechanism for Federated Learning**
  - **summary**: Combining multiple defense techniques to enhance security in federated learning.
  - **paper link**: [Hybrid Defense Mechanism Paper](https://arxiv.org/abs/2009.07840)
- **Comprehensive Defense Strategy for Federated Learning**
  - **summary**: Application of comprehensive defense strategies in federated learning.
  - **paper link**: [Comprehensive Defense Strategy Paper](https://dl.acm.org/doi/10.1145/3383313.3383325)
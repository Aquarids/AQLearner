# AQLearner

## Project Introduction

This project is a learning-use demo aimed at beginners. It provides basic practical code for machine learning, deep learning, cryptography, federated learning, and blockchain. The code in this project has only been tested on a Mac (Intel) environment and may have compatibility and other issues. This project is not suitable for actual commercial projects. If you encounter any issues or have suggestions for improvement during use, you are welcome to submit an issue or pull request.

## Project Contents

Here is a detailed list of the contents of this project, including the corresponding directories and test files:

| Category      | Content                          | Description                                 |
|---------------|----------------------------------|---------------------------------------------|
| Machine Learning | Linear Regression                | Implementation of linear regression using the normal equation |
| Machine Learning | Logistic Regression              | Implementation of logistic regression using gradient descent |
| Machine Learning | KNN                              | K-nearest neighbors classification algorithm |
| Machine Learning | DBSCAN                           | Density-based clustering algorithm          |
| Machine Learning | K-means                          | K-means clustering algorithm                |
| Machine Learning | Naive Bayes Classifier           | Implementation of a naive Bayes classifier  |
| Machine Learning | Gaussian Naive Bayes Classifier  | Naive Bayes classifier assuming Gaussian-distributed features |
| Machine Learning | AdaBoost                         | AdaBoost algorithm based on decision trees  |
| Machine Learning | Random Forest                    | Random forest algorithm based on decision trees |
| Machine Learning | PCA                              | Principal Component Analysis using EVD and SVD |
| Machine Learning | Decision Trees                   | Implementation of ID3, C45, and CART decision tree algorithms |
| Machine Learning | SVM                              | Simple linear kernel SVM implementation     |
| Deep Learning | Linear Regression                 | Simple linear regression using a single-layer network |
| Deep Learning | Logistic Regression               | Simple logistic regression using a single-layer network |
| Deep Learning | CNN Classifier                    | Simple CNN classifier using a single-layer convolution |
| Deep Learning | CNN Regression                    | Simple CNN regression using a single-layer convolution |
| Deep Learning | ResNet                            | Simplified version of a residual network    |
| Deep Learning | RNN                               | Basic implementation of a recurrent neural network |
| Deep Learning | LSTM                              | Long Short-Term Memory network implementation |
| Deep Learning | GRU                               | Gated Recurrent Unit network implementation |
| Deep Learning | Seq2Seq                           | Sequence to sequence model implementation   |
| Deep Learning | Transformer                       | Implementation of a Transformer model based on attention mechanisms |
| Deep Learning | Model Distillation                | Simple demonstration of model distillation techniques |
| Deep Learning | Diffusion Model                   | Simple demonstration of a diffusion model using Gaussian noise |
| Cryptography | DH Shared Encryption               | Implementation of Diffie-Hellman key exchange |
| Cryptography | Garbled Circuit                    | Simple implementation of an encrypted logic AND gate circuit |
| Cryptography | Additive Homomorphic Encryption    | Simple noise-based additive homomorphic encryption implementation |
| Cryptography | ElGamal Algorithm                  | Multiplicative homomorphic version of ElGamal encryption |
| Cryptography | Oblivious Transfer                 | Simple 1 out of 2 oblivious transfer based on the DH algorithm |
| Cryptography | XOR Secret Sharing                 | XOR-based n out of n secret sharing implementation |
| Cryptography | Shamir's Secret Sharing            | Implementation of Shamir's 2 out of n secret sharing scheme |
| Cryptography | Zero-Knowledge Proofs              | Simple demonstration of zero-knowledge proofs for root_x and root_xv_inv challenges |
| Cryptography | Secure Multi-Party Computation     | Demonstration of secure multi-party computation using ElGamal encryption |
| Cryptography | Differential Privacy               | Demonstration of differential privacy using Laplacian noise |
| Federated Learning | Privacy Intersection          | Simple demonstration of privacy intersection using hashing |
| Federated Learning | Configuration Mapping          | Implementation of JSON configuration to PyTorch model mapping |
| Federated Learning | Federated Learning             | Simple federated learning demonstration using gradient averaging, supports multi-party secure computation based on simple homomorphic encryption |


## Usage

- Clone the project：

```sh
git clone <repository-url>
```

- Install the requirements

```sh
pip install -r requirements.txt
```

- Run a specific test

```sh
python -m unittest <test_file>.<test_class>
```

--------

## 项目介绍

本项目是一个学习用demo项目，仅面向初学者，提供机器学习、深度学习、密码学、联邦学习及区块链的基础实践示意代码。本项目提供的代码仅在Mac（Intel）环境下进行过测试，可能存在环境兼容性和其他问题。本项目不适用于实际商业项目，如果您在使用过程中遇到任何问题或有改进建议，欢迎提交issue或pull request。

## 项目内容

以下是本项目包含的各部分内容及其对应目录和测试文件的详细列表：

| 分类         | 内容                           | 简单描述                                     |
|------------|-------------------------------|--------------------------------------------|
| 机器学习      | 线性回归                       | 实现基于正规方程的线性回归                   |
| 机器学习      | 逻辑回归                       | 实现基于梯度下降的逻辑回归                   |
| 机器学习      | K最近邻算法 (KNN)              | 实现K最近邻分类算法                         |
| 机器学习      | 密度聚类算法 (DBSCAN)          | 实现密度聚类算法                            |
| 机器学习      | K均值聚类算法 (K-means)        | 实现K均值聚类算法                           |
| 机器学习      | 朴素贝叶斯分类器               | 实现朴素贝叶斯分类器                        |
| 机器学习      | 高斯朴素贝叶斯分类器           | 特征假设符合高斯分布的朴素贝叶斯分类器        |
| 机器学习      | AdaBoost算法                  | 实现基于决策树的AdaBoost提升算法             |
| 机器学习      | 随机森林算法                   | 实现基于决策树的随机森林算法                 |
| 机器学习      | 主成分分析 (PCA)               | 主成分分析，EVD和SVD实现                     |
| 机器学习      | 决策树算法                     | 实现id3、c45、cart三种决策树算法             |
| 机器学习      | 支持向量机 (SVM)               | 实现简单线性核SVM                           |
| 深度学习      | 线性回归                       | 使用单层网络进行简单线性回归示意              |
| 深度学习      | 逻辑回归                       | 使用单层网络进行简单逻辑回归示意              |
| 深度学习      | 单层卷积CNN分类器              | 使用单层卷积的CNN分类示意                    |
| 深度学习      | 单层卷积CNN回归                | 使用单层卷积的CNN回归示意                    |
| 深度学习      | 简化版残差网络 (ResNet)        | 简化版的残差网络示意                         |
| 深度学习      | 循环神经网络 (RNN)             | 实现基本的循环神经网络示意                   |
| 深度学习      | 长短期记忆网络 (LSTM)          | 实现长短期记忆网络                           |
| 深度学习      | 门控循环单元网络 (GRU)         | 实现门控循环单元网络                         |
| 深度学习      | 序列到序列模型 (Seq2Seq)       | 实现序列到序列模型                           |
| 深度学习      | Transformer                  | 实现基于注意力机制的Transformer模型           |
| 深度学习      | 蒸馏模型                      | 实现模型蒸馏技术的简单示意                     |
| 深度学习      | 扩散模型                      | 使用高斯噪声生成扩散模型的简单示意               |
| 深度学习      | Metric工具类                   | 包含性能评估指标的工具类                      |
| 密码学        | DH共享加密                     | 实现Diffie-Hellman密钥交换                 |
| 密码学        | 混淆电路                      | 实现简单的逻辑与门电路加密                     |
| 密码学        | 加法同态加密                  | 实现基于噪音的简单加法同态加密                  |
| 密码学        | ElGamal算法                 | 实现ElGamal加密的乘法同态版本                  |
| 密码学        | 不经意传输                   | 实现基于DH算法的1 out of 2简单不经意传输        |
| 密码学        | XOR秘密共享                 | 实现基于XOR的n out of n秘密共享                |
| 密码学        | Shamir秘密共享               | 实现Shamir的2 out of n秘密共享方案            |
| 密码学        | 零知识证明                   | 实现简单root_x和root_xv_inv挑战的零知识证明示意  |
| 密码学        | 安全多方计算                 | 实现基于ElGamal加密的安全多方计算示意            |
| 密码学        | 差分隐私                      | 实现拉普拉斯噪声的差分隐私示意                 |
| 联邦学习      | 隐私求交                       | 基于哈希的简单隐私求交示意                    |
| 联邦学习      | 配置映射                       | 实现JSON配置到PyTorch模型的映射               |
| 联邦学习      | 联邦学习                       | 实现基于梯度平均的简单联邦学习示意，支持基于简单同态的多方安全计算      |
| 联邦学习      | 联邦学习Metric工具类            | 实现每轮计算后的性能评估指标的工具类              |
| 区块链        | 待实现                         | 待开发                                       |

## 使用方法

- 克隆项目到本地：

```sh
git clone <repository-url>
```

- 安全依赖

```sh
pip install -r requirements.txt
```

- 运行某个测试

```sh
python -m unittest <test_file>.<test_class>
```

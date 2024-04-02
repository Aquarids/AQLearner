import unittest
import numpy as np
import torch
import torch.utils.data
import dl.metrics as Metrics
import sklearn.datasets
import sklearn.model_selection
import torchvision.transforms
import matplotlib.pyplot as plt

from dl.simple_linear_regression import SimpleLinearRegression
from dl.simple_logistic_regression import SimpleLogisticRegression
from dl.simple_cnn_classifier import SimpleCNNClassifier
from dl.simple_cnn_regression import SimpleCNNRegression
from dl.res_net import ResNet
from dl.rnn import rnn
from dl.gru import GRU
from dl.lstm import LSTM
from dl.seq2seq import Seq2Seq

from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestNN(unittest.TestCase):
    def test_simple_linear_regression(self):
        X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=42)

        num_features = X_train.shape[1]
        model = SimpleLinearRegression(num_features, 1)
        model.fit(X_train, y_train)
        model.summary()
        y_pred = model.predict(X_test)

        y_test_np = y_test.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()

        print('SimpleLinearRegression MSE:', Metrics.mse(y_test_np, y_pred_np))

    def test_simple_logistic_regression(self):
        X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=42)

        num_features = X_train.shape[1]
        model = SimpleLogisticRegression(num_features, 1)
        model.fit(X_train, y_train)
        model.summary()
        y_pred = model.predict(X_test)

        y_test_np = y_test.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()

        print('SimpleLogisticRegression Accuracy:', Metrics.accuracy(y_test_np, y_pred_np))

class TestCNN(unittest.TestCase):
    def test_simple_cnn_classifier(self):
        num_classes = 10
        num_channels = 1

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),  # 确保图像大小为28x28
            torchvision.transforms.ToTensor(),  # 将图像转换为Tensor
            torchvision.transforms.Normalize((0.5,), (0.5,))  # 标准化
        ])

        # 下载/加载MNIST数据集
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                                download=True, transform=transform)

        # 创建DataLoader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = SimpleCNNClassifier().to(device)
        model.fit(train_loader)
        model.summary()
        y_pred, y_possibility = model.predict(test_loader)

        y_test = []
        for _, y in test_loader:
            y_test += y.tolist()
        
        print('SimpleCNNClassifier Accuracy:', Metrics.accuracy(y_test, y_pred))
        print('SimpleCNNClassifier Precision, Recall, F1:', Metrics.precision_recall_f1(y_test, y_pred))
        Metrics.plot_roc_curve(y_test, y_possibility)

    def test_simple_cnn_regression(self):
        num_samples = 1000

        X = torch.rand((num_samples, 1, 28, 28))
        y_regression = torch.randn((num_samples, 1))
        dataset_regression = torch.utils.data.TensorDataset(X, y_regression)

        train_size_regression = int(0.8 * num_samples)
        test_size_regression = num_samples - train_size_regression
        train_dataset_regression, test_dataset_regression = torch.utils.data.random_split(dataset_regression, [train_size_regression, test_size_regression])

        train_loader_regression = torch.utils.data.DataLoader(train_dataset_regression, batch_size=64, shuffle=True)
        test_loader_regression = torch.utils.data.DataLoader(test_dataset_regression, batch_size=64, shuffle=False)

        model = SimpleCNNRegression()
        model.fit(train_loader_regression)
        model.summary()
        y_pred = model.predict(test_loader_regression)

        y_test = []
        for _, y in test_loader_regression:
            y_test += y.tolist()

        print('SimpleCNNRegression MSE:', Metrics.mse(np.array(y_test), np.array(y_pred)))

class TestResNet(unittest.TestCase):
    def test_res_net(self):
        num_classes = 10
        num_channels = 1

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),  # 确保图像大小为28x28
            torchvision.transforms.ToTensor(),  # 将图像转换为Tensor
            torchvision.transforms.Normalize((0.5,), (0.5,))  # 标准化
        ])

        # 下载/加载MNIST数据集
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                                download=True, transform=transform)

        # 创建DataLoader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = ResNet(num_channels, num_classes).to(device)
        model.fit(train_loader)
        model.summary()

        y_pred, y_possibility = model.predict(test_loader)

        y_test = []
        for _, y in test_loader:
            y_test += y.tolist()

        print('ResNet Accuracy:', Metrics.accuracy(y_test, y_pred))
        print('ResNet Precision, Recall, F1:', Metrics.precision_recall_f1(y_test, y_pred))
        Metrics.plot_roc_curve(y_test, y_possibility)

class TestRNN(unittest.TestCase):
    def test_rnn(self):
        t = np.linspace(0, 100, 1000)
        sin_wave = np.sin(t)
        lock_back = 5

        X, y = [], []
        for i in range(len(sin_wave) - lock_back):
            seq_in = sin_wave[i:i + lock_back]
            seq_out = sin_wave[i + lock_back]
            X.append(seq_in)
            y.append(seq_out)

        X, y = torch.FloatTensor(np.array(X)).unsqueeze(-1), torch.FloatTensor(np.array(y)).unsqueeze(-1)
        dataset = torch.utils.data.TensorDataset(X, y)

        train_size = int(0.8 * len(X))
        test_size = len(X) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = rnn(1, 10, 1)
        model.fit(train_loader)
        model.summary()

        y_pred = model.predict(test_loader)

        y_test = []
        for _, y in test_loader:
            y_test += y.tolist()
        
        print('SimpleRNN MSE:', Metrics.mse(np.array(y_test), np.array(y_pred)))
        y_pred_np = np.concatenate(y_pred, axis=0).flatten()
        y_test_np = np.concatenate(y_test, axis=0).flatten()
        plt.plot(t[train_size + lock_back:], y_test_np, label='Test')
        plt.plot(t[train_size + lock_back:], y_pred_np, label='Predict')
        plt.legend()
        plt.show()

class TestGRU(unittest.TestCase):
    def test_gru(self):
        t = np.linspace(0, 100, 1000)
        sin_wave = np.sin(t)
        lock_back = 5

        X, y = [], []
        for i in range(len(sin_wave) - lock_back):
            seq_in = sin_wave[i:i + lock_back]
            seq_out = sin_wave[i + lock_back]
            X.append(seq_in)
            y.append(seq_out)

        X, y = torch.FloatTensor(np.array(X)).unsqueeze(-1), torch.FloatTensor(np.array(y)).unsqueeze(-1)
        dataset = torch.utils.data.TensorDataset(X, y)   

        train_size = int(0.8 * len(X))
        test_size = len(X) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)  

        model = GRU(1, 16, 1)
        model.fit(train_loader)
        model.summary()

        y_pred = model.predict(test_loader)

        y_test = []
        for _, y in test_loader:
            y_test += y.tolist()

        print('GRU MSE:', Metrics.mse(np.array(y_test), np.array(y_pred)))
        y_pred_np = np.array(y_pred)
        y_test_np = np.array(y_test)

        plt.plot(t[train_size + lock_back:], y_test_np, label='Test')
        plt.plot(t[train_size + lock_back:], y_pred_np, label='Predict')
        plt.legend()
        plt.show() 

class TestLSTM(unittest.TestCase):
    def test_lstm(self):
        t = np.linspace(0, 100, 1000)
        sin_wave = np.sin(t)
        lock_back = 5

        X, y = [], []
        for i in range(len(sin_wave) - lock_back):
            seq_in = sin_wave[i:i + lock_back]
            seq_out = sin_wave[i + lock_back]
            X.append(seq_in)
            y.append(seq_out)

        X, y = torch.FloatTensor(np.array(X)).unsqueeze(-1), torch.FloatTensor(np.array(y)).unsqueeze(-1)
        dataset = torch.utils.data.TensorDataset(X, y)

        train_size = int(0.8 * len(X))
        test_size = len(X) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = LSTM(1, 16, 1)
        model.fit(train_loader)
        model.summary()

        y_pred = model.predict(test_loader)

        y_test = []
        for _, y in test_loader:
            y_test += y.tolist()
        
        print('LSTM MSE:', Metrics.mse(np.array(y_test), np.array(y_pred)))
        y_pred_np = np.array(y_pred)
        y_test_np = np.array(y_test)

        plt.plot(t[train_size + lock_back:], y_test_np, label='Test')
        plt.plot(t[train_size + lock_back:], y_pred_np, label='Predict')
        plt.legend()
        plt.show()

class TestSeq2Seq(unittest.TestCase):
    def tokenize(self, data):
        return data.lower().split(' ')
    
    def build_vocab(self, data):
        vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        index = 4

        for sentence in data:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = index
                    index += 1
        return vocab
    
    def decode_sentence(self, indices, index_to_word, special_tokens={'<sos>', '<eos>', '<pad>', '<unk>'}):
        words = [index_to_word.get(idx, '<unk>') for idx in indices]
        filtered_words = [word for word in words if word not in special_tokens]
        
        # 如果过滤后的句子主要是<unk>，返回"unknown"
        if len(filtered_words) == 0 or filtered_words.count('<unk>') / len(filtered_words) > 0.5:
            return "unknown"
        else:
            return ' '.join(filtered_words)

    def numericalize(self, sentence, vocab):
        return [vocab.get(word.lower(), vocab["<unk>"]) for word in sentence.split()]
    
    def pad_sequences(self, sequences, pad_token=0):
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = [seq + [pad_token] * (max_len - len(seq)) for seq in sequences]
        return padded_sequences

    def test_seq2seq(self):
        data = [
            ("I am a student", "我 是 一 个 学生"),
            ("He is a teacher", "他 是 一 名 教师"),
            ("I love you", "我 爱 你"),
            ("I miss you", "我 想 你"),
            ("I am a programmer", "我 是 一 名 程序员"),
            ("I am a data scientist", "我 是 一名 数据 科学家"),
            ("I am a machine learning engineer", "我 是 一 名 机器 学习 工程师"),
            ("I am a deep learning engineer", "我 是 一 名 深度 学习 工程师"),
            ("I am a software engineer", "我 是 一 名 软件 工程师"),
            ("What is your name", "你 叫 什么 名字")
        ]

        src_data, tgt_data = [self.tokenize(d[0]) for d in data], [self.tokenize(d[1]) for d in data]
        src_vocab, tgt_vocab = self.build_vocab(src_data), self.build_vocab(tgt_data)

        index_to_src_word = {index: word for word, index in src_vocab.items()}
        index_to_tgt_word = {index: word for word, index in tgt_vocab.items()}

        print("Source Vocabulary:", src_vocab)
        print("Target Vocabulary:", tgt_vocab)

        src_num = self.pad_sequences([self.numericalize(d[0], src_vocab) for d in data], src_vocab["<pad>"])
        tgt_num = self.pad_sequences([self.numericalize(d[1], tgt_vocab) for d in data], tgt_vocab["<pad>"])

        dataset = torch.utils.data.TensorDataset(torch.tensor(src_num), torch.tensor(tgt_num))

        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = Seq2Seq(len(src_vocab), len(tgt_vocab), 16, 16)
        model.fit(train_loader)
        model.summary()

        predictions = model.predict(test_loader)
        print("Predictions:" ,predictions)
        
        for i in range(len(predictions)):
            src_sentence = test_loader.dataset[i][0].tolist()
            tgt_sentence = test_loader.dataset[i][1].tolist()
            print("Source:", self.decode_sentence(src_sentence, index_to_src_word))
            print("Target:", self.decode_sentence(tgt_sentence, index_to_tgt_word))

            pred_indices = predictions[i].squeeze().tolist()  # 使用squeeze()移除大小为1的维度，然后转换为列表
            # 如果pred_indices是嵌套列表（即模型返回的是批量预测），只取第一个
            if isinstance(pred_indices[0], list):
                pred_indices = pred_indices[0]
            pred_sentence = self.decode_sentence(pred_indices, index_to_tgt_word)
            print("Predicted:", pred_sentence)

if __name__ == '__main__':
    unittest.main()
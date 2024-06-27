import unittest
import numpy as np
import random
import torch
import torch.utils.data.dataloader
import torchtext
import torch.utils.data
import dl.metrics as Metrics
import sklearn.datasets
import sklearn.model_selection
import torchvision.transforms
import matplotlib.pyplot as plt
import dl.nlp.preprocess as Preprocess
import pandas as pd

from dl.simple_linear_regression import SimpleLinearRegression
from dl.simple_logistic_regression import SimpleLogisticRegression
from dl.simple_cnn_classifier import SimpleCNNClassifier
from dl.simple_cnn_regression import SimpleCNNRegression
from dl.res_net import ResNet
from dl.gan import GAN
from dl.rnn import RNN
from dl.gru import GRU
from dl.lstm import LSTM
from dl.nlp.seq2seq import Seq2Seq
from dl.nlp.transformer import Transformer
from dl.nlp.gpt import GPT
from dl.nlp.wiki_text2 import WikiText2
from dl.distillation import TeacherModel, StudentModel
from dl.diffusion import DiffusionModel

from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestNN(unittest.TestCase):

    def test_simple_linear_regression(self):
        X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32).view(-1, 1)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.1, random_state=42)

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
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32).view(-1, 1)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.1, random_state=42)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=32,
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_test, y_test),
            batch_size=32,
            shuffle=False)

        num_features = X_train.shape[1]
        model = SimpleLogisticRegression(num_features, 1)
        model.fit(train_loader)
        model.summary()
        y_pred, _ = model.predict(test_loader)

        y_test = []
        for _, y in test_loader:
            y_test += y.tolist()

        print('SimpleLogisticRegression Accuracy:',
              Metrics.accuracy(np.array(y_test), np.array(y_pred)))

class TestCNN(unittest.TestCase):

    def test_simple_cnn_classifier(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
        ])

        train_dataset = torchvision.datasets.MNIST(root='./data',
                                                   train=True,
                                                   download=True,
                                                   transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data',
                                                  train=False,
                                                  download=True,
                                                  transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=32,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=32,
                                                  shuffle=False)
        model = SimpleCNNClassifier().to(device)
        model.fit(train_loader, n_iters=1)

        # model.summary()
        y_pred, y_prob = model.predict(test_loader)

        y_test = []
        for _, y in test_loader:
            y_test += y.tolist()

        print('SimpleCNNClassifier Accuracy:',
              Metrics.accuracy(y_test, y_pred))
        # print('SimpleCNNClassifier Precision, Recall, F1:',
        #       Metrics.precision_recall_f1(y_test, y_pred))

    def test_simple_cnn_regression(self):
        num_samples = 1000

        X = torch.rand((num_samples, 1, 28, 28))
        y_regression = torch.randn((num_samples, 1))
        dataset_regression = torch.utils.data.TensorDataset(X, y_regression)

        train_size_regression = int(0.8 * num_samples)
        test_size_regression = num_samples - train_size_regression
        train_dataset_regression, test_dataset_regression = torch.utils.data.random_split(
            dataset_regression, [train_size_regression, test_size_regression])

        train_loader_regression = torch.utils.data.DataLoader(
            train_dataset_regression, batch_size=64, shuffle=True)
        test_loader_regression = torch.utils.data.DataLoader(
            test_dataset_regression, batch_size=64, shuffle=False)

        model = SimpleCNNRegression()
        model.fit(train_loader_regression)
        model.summary()
        y_pred = model.predict(test_loader_regression)

        y_test = []
        for _, y in test_loader_regression:
            y_test += y.tolist()

        print('SimpleCNNRegression MSE:',
              Metrics.mse(np.array(y_test), np.array(y_pred)))


class TestResNet(unittest.TestCase):

    def test_res_net(self):
        num_classes = 10
        num_channels = 1

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),  # 确保图像大小为28x28
            torchvision.transforms.ToTensor(),  # 将图像转换为Tensor
            torchvision.transforms.Normalize((0.5, ), (0.5, ))  # 标准化
        ])

        # 下载/加载MNIST数据集
        train_dataset = torchvision.datasets.MNIST(root='./data',
                                                   train=True,
                                                   download=True,
                                                   transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data',
                                                  train=False,
                                                  download=True,
                                                  transform=transform)

        # 创建DataLoader
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=64,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=64,
                                                  shuffle=False)

        model = ResNet(num_channels, num_classes).to(device)
        model.fit(train_loader)
        model.summary()

        y_pred, y_prob = model.predict(test_loader)

        y_test = []
        for _, y in test_loader:
            y_test += y.tolist()

        print('ResNet Accuracy:', Metrics.accuracy(y_test, y_pred))
        print('ResNet Precision, Recall, F1:',
              Metrics.precision_recall_f1(y_test, y_pred))


class TestGAN(unittest.TestCase):

    def test_gan(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, ), (0.5, ))
        ])
        dataset = torchvision.datasets.MNIST(root='./data',
                                             train=True,
                                             download=True,
                                             transform=transform)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=64,
                                             shuffle=True)

        input_dim = 100
        output_dim = 28 * 28

        gan = GAN(input_dim, output_dim)
        gan.train(loader, lr=0.0002, n_iter=10)

        noise = torch.randn(1, input_dim)
        generated_image = gan.generator(noise).view(28,
                                                    28).detach().cpu().numpy()
        plt.imshow(generated_image, cmap='gray')
        plt.show()


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

        X, y = torch.FloatTensor(np.array(X)).unsqueeze(-1), torch.FloatTensor(
            np.array(y)).unsqueeze(-1)
        dataset = torch.utils.data.TensorDataset(X, y)

        train_size = int(0.8 * len(X))
        test_size = len(X) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size])
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=64,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=64,
                                                  shuffle=False)

        model = RNN(1, 10, 1)
        model.fit(train_loader)
        model.summary()

        y_pred = model.predict(test_loader)

        y_test = []
        for _, y in test_loader:
            y_test += y.tolist()

        print('SimpleRNN MSE:', Metrics.mse(np.array(y_test),
                                            np.array(y_pred)))
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

        X, y = torch.FloatTensor(np.array(X)).unsqueeze(-1), torch.FloatTensor(
            np.array(y)).unsqueeze(-1)
        dataset = torch.utils.data.TensorDataset(X, y)

        train_size = int(0.8 * len(X))
        test_size = len(X) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size])
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=64,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=64,
                                                  shuffle=False)

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

        X, y = torch.FloatTensor(np.array(X)).unsqueeze(-1), torch.FloatTensor(
            np.array(y)).unsqueeze(-1)
        dataset = torch.utils.data.TensorDataset(X, y)

        train_size = int(0.8 * len(X))
        test_size = len(X) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size])
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=64,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=64,
                                                  shuffle=False)

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

    def test_seq2seq(self):
        data = [("I am a student", "我 是 一 个 学生"),
                ("He is a teacher", "他 是 一 名 教师"), ("I love you", "我 爱 你"),
                ("I miss you", "我 想 你"), ("I am a programmer", "我 是 一 名 程序员"),
                ("I am a data scientist", "我 是 一名 数据 科学家"),
                ("I am a machine learning engineer", "我 是 一 名 机器 学习 工程师"),
                ("I am a deep learning engineer", "我 是 一 名 深度 学习 工程师"),
                ("I am a software engineer", "我 是 一 名 软件 工程师"),
                ("What is your name", "你 叫 什么 名字")]

        max_length = 10
        src_numericalized, tgt_numericalized, src_vocab, tgt_vocab, index_to_src_word, index_to_tgt_word = Preprocess.preprocess_texts(
            data, max_length)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(src_numericalized), torch.tensor(tgt_numericalized))

        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size])

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=1,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False)

        model = Seq2Seq(len(src_vocab), len(tgt_vocab), 16, 16)
        model.fit(train_loader)
        model.summary()

        predictions = model.predict(test_loader, max_length)

        for i in range(len(predictions)):
            src_sentence = test_loader.dataset[i][0].tolist()
            tgt_sentence = test_loader.dataset[i][1].tolist()
            print("Source:",
                  Preprocess.decode_sentence(src_sentence, index_to_src_word))
            print("Target:",
                  Preprocess.decode_sentence(tgt_sentence, index_to_tgt_word))

            prediction = predictions[i][0].tolist()
            pred_sentence = Preprocess.decode_sentence(prediction,
                                                       index_to_tgt_word)
            print("Predicted:", pred_sentence)


class TestTransformer(unittest.TestCase):

    def test_transformer(self):
        data = [("I am a student", "我 是 一 个 学生"),
                ("He is a teacher", "他 是 一 名 教师"), ("I love you", "我 爱 你"),
                ("I miss you", "我 想 你"), ("I am a programmer", "我 是 一 名 程序员"),
                ("I am a data scientist", "我 是 一名 数据 科学家"),
                ("I am a machine learning engineer", "我 是 一 名 机器 学习 工程师"),
                ("I am a deep learning engineer", "我 是 一 名 深度 学习 工程师"),
                ("I am a software engineer", "我 是 一 名 软件 工程师"),
                ("What is your name", "你 叫 什么 名字")]

        max_length = 10
        src_numericalized, tgt_numericalized, src_vocab, tgt_vocab, index_to_src_word, index_to_tgt_word = Preprocess.preprocess_texts(
            data, max_length)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(src_numericalized), torch.tensor(tgt_numericalized))

        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size])

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=1,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False)

        model = Transformer(len(src_vocab),
                            len(tgt_vocab),
                            d_model=512,
                            num_heads=8,
                            num_layers=6,
                            ff_hidden_dim=2048,
                            dropout=0.1)
        model.fit(train_loader)
        model.summary()

        predictions = model.predict(test_loader, max_length)

        for i in range(len(predictions)):
            src_sentence = test_loader.dataset[i][0].tolist()
            tgt_sentence = test_loader.dataset[i][1].tolist()
            print("Source:",
                  Preprocess.decode_sentence(src_sentence, index_to_src_word))
            print("Target:",
                  Preprocess.decode_sentence(tgt_sentence, index_to_tgt_word))

            prediction = predictions[i][0].tolist()
            pred_sentence = Preprocess.decode_sentence(prediction,
                                                       index_to_tgt_word)
            print("Predicted:", pred_sentence)


class TestGPT(unittest.TestCase):

    def yield_tokens(self, tokenizer, data_iter):
        for text in data_iter:
            yield tokenizer(text.lower())

    def data_process(self, tokenizer, vocab, raw_text_iter):
        data = []
        for line in raw_text_iter:
            tokens = tokenizer(line)
            indices = [vocab[token] for token in tokens]
            data.extend(indices)
        return torch.tensor(data, dtype=torch.long)

    def batchify(self, data, bsz):
        seq_len = data.size(0) // bsz
        data = data[:seq_len * bsz]
        data = data.view(bsz, seq_len).t().contiguous()
        return data.to(device)

    def load_local_wiki_text2(self):
        test = pd.read_parquet(
            './data/WikiText2/test.parquet')['text'].tolist()
        train = pd.read_parquet(
            './data/WikiText2/train.parquet')['text'].tolist()
        valid = pd.read_parquet(
            './data/WikiText2/validation.parquet')['text'].tolist()

        return train, valid, test

    def custom_collate(self, batch):
        input_seqs, target_seqs = zip(*batch)
        input_seqs = torch.stack(input_seqs, dim=0)
        target_seqs = torch.stack(target_seqs, dim=0)
        return input_seqs, target_seqs

    def test_gpt(self):
        tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

        # train_iter, val_iter, test_iter = torchtext.datasets.WikiText2()
        train_iter, val_iter, test_iter = self.load_local_wiki_text2()
        vocab = torchtext.vocab.build_vocab_from_iterator(
            self.yield_tokens(tokenizer, train_iter),
            specials=["<unk>", "<pad>", "<eos>"])
        vocab.set_default_index(vocab["<unk>"])

        batch_size = 10
        seq_length = 20
        train_data = self.data_process(tokenizer, vocab, train_iter)[:10000]
        val_data = self.data_process(tokenizer, vocab, val_iter)[:1000]
        test_data = self.data_process(tokenizer, vocab, test_iter)[:1000]

        train_dataset = WikiText2(train_data, seq_length)
        val_dataset = WikiText2(val_data, seq_length)
        test_dataset = WikiText2(test_data, seq_length)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.custom_collate)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.custom_collate)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.custom_collate)

        model = GPT(vocab_size=len(vocab),
                    embed_size=256,
                    num_layers=6,
                    heads=8,
                    forward_expansion=4,
                    dropout=0.1,
                    max_length=200)
        model.pre_train(train_loader, n_epochs=1)

        # it should use specific datasets, but here we use test_loader for simplicity
        model.fine_tune(test_loader, val_loader, n_epochs=1)

        index_to_token = {
            index: token
            for token, index in vocab.get_stoi().items()
        }
        ans = model.predict("Hellow ", seq_length, vocab, index_to_token)
        print(ans)


class TestDistillation(unittest.TestCase):

    def test_distillation(self):
        train_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor())
        test_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor())

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=64,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=64,
                                                  shuffle=False)

        teacher = TeacherModel().to(device)
        teacher.fit(train_loader)

        student = StudentModel(teacher).to(device)
        student.distill(train_loader)

        y_pred_teacher, y_prob_teacher = teacher.predict(test_loader)
        y_pred_student, y_prob_student = student.predict(test_loader)

        accuracy = Metrics.accuracy(np.array(y_pred_teacher),
                                    np.array(y_pred_student))
        print('Distillation Accuracy:', accuracy)


class TestDiffusion(unittest.TestCase):

    def imshow(self, img, title):
        npimg = img.numpy()
        npimg = np.squeeze(npimg, axis=0)
        plt.imshow(npimg, cmap='gray')
        plt.title(title)
        plt.show()

    def test_diffusion(self):
        train_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor())
        test_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor())

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=1,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False)

        model = DiffusionModel().to(device)
        model.fit(train_loader)

        input = test_loader.dataset[0][0]
        noise_level = 0.3
        corrupted_data = model.corrupt(input, noise_level, mean=0, std=0.5)

        reconstructed_data = model.reverse_diffusion(corrupted_data,
                                                     steps=50,
                                                     std=0.5)

        self.imshow(input, 'Original Image')
        self.imshow(corrupted_data.view(1, 28, 28).detach(), 'Noisy Image')
        self.imshow(
            reconstructed_data.view(1, 28, 28).detach(), 'Reconstructed Image')


import os
from PIL import Image
from tqdm import tqdm

class DownloadImage(unittest.TestCase):


    def downloadMNIST(self, save_path, save_rules, train=True):
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])

        # Download the MNIST dataset
        if train:
            mnist_data = torchvision.datasets.MNIST(root='./data',
                                                    train=True,
                                                    download=True,
                                                    transform=transform)
        else:
            mnist_data = torchvision.datasets.MNIST(root='./data',
                                                    train=False,
                                                    download=True,
                                                    transform=transform)

        # Specify the directory to save the images
        save_dir = save_path
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        toPil = torchvision.transforms.ToPILImage()
    
        # Save images
        for label in save_rules:
            folder_name = os.path.join(save_dir, save_rules[label])
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

        progress_bar = tqdm(total=len(mnist_data))
        for idx, (image, label) in enumerate(mnist_data):
            folder_name = save_rules[label]
            img = toPil(image)
            img.save(os.path.join(save_dir, folder_name, f"{idx}.png"))
            progress_bar.update(1)
        progress_bar.close()


    def test_download_mnist(self):
        path = "./data/mnist_images"
        label_rules = {
            0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
            5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'
        }
        self.downloadMNIST(path, label_rules, True)
        print("Images have been saved to", path)


if __name__ == '__main__':
    unittest.main()

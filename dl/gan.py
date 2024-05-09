import torch
from tqdm import tqdm


class Generator(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(input_dim, 128),
                                       torch.nn.LeakyReLU(0.2),
                                       torch.nn.Linear(128, 256),
                                       torch.nn.LeakyReLU(0.2),
                                       torch.nn.Linear(256, 512),
                                       torch.nn.LeakyReLU(0.2),
                                       torch.nn.Linear(512, output_dim),
                                       torch.nn.Tanh())

    def forward(self, x):
        return self.net(x)


class Discriminator(torch.nn.Module):

    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(input_dim, 512),
                                       torch.nn.LeakyReLU(0.2),
                                       torch.nn.Linear(512, 256),
                                       torch.nn.LeakyReLU(0.2),
                                       torch.nn.Linear(256, 1),
                                       torch.nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


class GAN:

    def __init__(self, input_dim, output_dim):
        self.generator = Generator(input_dim, output_dim)
        self.discriminator = Discriminator(output_dim)

    def train(self, loader, lr, n_iter):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        criterion = torch.nn.BCELoss()

        progress_bar = tqdm(range(n_iter * len(loader)))

        for _ in range(n_iter):
            for real_data, _ in loader:
                real_data = real_data.view(real_data.size(0), -1)
                batch_size = real_data.size(0)

                # Real data label is 1, fake data label is 0
                real_label = torch.ones(batch_size, 1)
                fake_label = torch.zeros(batch_size, 1)

                # Train Discriminator
                self.discriminator.zero_grad()
                output_real = self.discriminator(real_data)
                loss_real = criterion(output_real, real_label)

                # Generate fake data
                noise = torch.randn(batch_size, 100)
                fake_data = self.generator(noise)
                output_fake = self.discriminator(fake_data.detach())
                loss_fake = criterion(output_fake, fake_label)

                # Discriminator loss
                loss_d = (loss_real + loss_fake) / 2
                loss_d.backward()
                optimizer_D.step()

                # Train Generator
                self.generator.zero_grad()
                output_fake = self.discriminator(fake_data)
                loss_g = criterion(output_fake, real_label)
                loss_g.backward()
                optimizer_G.step()

                progress_bar.update(1)
                progress_bar.set_description(
                    f'Loss D: {loss_d.item()}, Loss G: {loss_g.item()}')
        progress_bar.close()

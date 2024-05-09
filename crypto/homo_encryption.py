import random
import Crypto.Util.number


class AddHomomorphicEncryption:

    def __init__(self, noise_range=100):
        self.noise = random.randint(-noise_range, noise_range)

    def encrypt(self, value):
        return value + self.noise

    def decrypt(self, value):
        return value - self.noise

    def add(self, encrypted_values):
        if len(encrypted_values) < 3:
            raise ValueError(
                "Require at least 3 members to join the addition operation")
        return sum(encrypted_values) - self.noise * (len(encrypted_values) - 1)


class ElGamalHE:

    def __init__(self, bits=512):
        self.bits = bits
        self.p = Crypto.Util.number.getPrime(bits)
        self.g = random.randint(2, self.p - 1)
        self.x = random.randint(2, self.p - 1)  # private key
        self.y = pow(self.g, self.x, self.p)  # public key

    def __init__(self, elgamal_public_key, elgamal_private_key=None):
        self.g, self.p, self.y = elgamal_public_key
        self.x = elgamal_private_key
        return

    def encrypt(self, value):
        k = random.randint(2, self.p - 1)
        while Crypto.Util.number.GCD(k, self.p - 1) != 1:
            k = random.randint(2, self.p - 1)
        c1 = pow(self.g, k, self.p)
        c2 = (value * pow(self.y, k, self.p)) % self.p
        return c1, c2

    def decrypt(self, encrypted_value):
        c1_1, c1_2 = encrypted_value
        value = (c1_2 * Crypto.Util.number.inverse(pow(c1_1, self.x, self.p),
                                                   self.p)) % self.p
        return value

    def multiply(self, encrypted_value1, encrypted_value2):
        c1_1, c1_2 = encrypted_value1
        c2_1, c2_2 = encrypted_value2
        return (c1_1 * c2_1) % self.p, (c1_2 * c2_2) % self.p

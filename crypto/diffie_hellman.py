import Crypto.Util.number
import hashlib

class DiffieHellman:
    def __init__(self, p, g):
        self.p = p
        self.g = g
        self.private_key = Crypto.Util.number.getRandomNBitInteger(256)
        self.public_key = pow(self.g, self.private_key, self.p)

    def get_public_key(self):
        return self.public_key

    def get_shared_secret(self, other_public_key):
        return hashlib.sha256(str(pow(other_public_key, self.private_key, self.p)).encode('utf-8')).digest()

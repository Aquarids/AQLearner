from crypto.homo_encryption import ElGamalHE

import Crypto.Util.number
import random

# a simple smpc with elgamal with central server
class SMPCParty:
    def __init__(self, value, elgamal_public_key):
        self.value = value
        self.elgamal_public_key = elgamal_public_key
        self.encrypted_input_value = None
        self.elgamal_he = ElGamalHE(elgamal_public_key)
        
    def encrypt_input(self):
        g, p, y = self.elgamal_public_key
        self.encrypted_input_value = self.elgamal_he.encrypt(self.value)
        return self.encrypted_input_value
    
class SMPCCenter:
    def __init__(self, bits=256):
        self.bits = bits
        self.p = Crypto.Util.number.getPrime(bits)
        self.g = random.randint(2, self.p - 1)
        self.x = random.randint(2, self.p - 1) # private key
        self.y = pow(self.g, self.x, self.p) # public key
        self.elgamal_public_key = (self.g, self.p, self.y)
        self.elgamal_he = ElGamalHE(self.elgamal_public_key, self.x)

    def get_public_key(self):
        return self.elgamal_public_key

    def aggregate_encrypted_inputs(self, encrypted_inputs):
        agg_c1, agg_c2 = 1, 1
        for c1, c2 in encrypted_inputs:
            agg_c1 = (agg_c1 * c1) % self.p
            agg_c2 = (agg_c2 * c2) % self.p
        return agg_c1, agg_c2

    def decrypt(self, encrypted_value):
        return self.elgamal_he.decrypt(encrypted_value)
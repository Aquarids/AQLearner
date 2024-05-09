import Crypto.Cipher.AES
import Crypto.Random


# a simple AND garbled circuit implementation
class GarbledCircuit:

    def __init__(self):
        self.keys = self.generate_keys()

    def generate_keys(self):
        return {
            'A0': Crypto.Random.get_random_bytes(16),
            'A1': Crypto.Random.get_random_bytes(16),
            'B0': Crypto.Random.get_random_bytes(16),
            'B1': Crypto.Random.get_random_bytes(16),
        }

    def encrypt(self, key, message):
        cipher = Crypto.Cipher.AES.new(key, Crypto.Cipher.AES.MODE_ECB)
        return cipher.encrypt(message.ljust(16, b'\0'))

    def garble_circuit(self):
        table = {}
        for a in ['A0', 'A1']:
            for b in ['B0', 'B1']:
                if a == 'A1' and b == 'B1':
                    table[(a, b)] = self.encrypt(self.keys[a], b'1')
                else:
                    table[(a, b)] = self.encrypt(self.keys[a], b'0')
        return table

    def evaluate(self, table, a, b):
        for (a_input, b_input), encrypted_output in table.items():
            try:
                if (a == self.keys[a_input] and b == self.keys[b_input]):
                    output = Crypto.Cipher.AES.new(
                        a, Crypto.Cipher.AES.MODE_ECB).decrypt(
                            encrypted_output).rstrip(b'\0')
                    return int(output.decode())
            except ValueError:
                continue
        return -1

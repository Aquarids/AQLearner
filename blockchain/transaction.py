from ecdsa import SigningKey, SECP256k1, VerifyingKey
import hashlib
import time

class Transaction:
    def __init__(self, sender, recipient, amount, timestamp=None, signature=None, is_coinbase=False):
        self.sender = sender
        self.recipient = recipient
        self.amount = amount
        self.is_coinbase = is_coinbase
        self.timestamp = timestamp if timestamp else time.time()
        self.signature = signature

        self.inputs = []
        self.outputs = [(recipient, amount)]
        self.txid = self.calculate_txid()

    def calculate_txid(self):
        tx = f"{self.sender}{self.recipient}{self.amount}{self.timestamp}{self.is_coinbase}"
        return hashlib.sha256(tx.encode()).hexdigest()
    
    def sign(self, private_key):
        # Simplified signature, for real-world applications, need generate a real system signature
        if private_key == "system":
            self.signature = "system_signature"
        else:
            sk = SigningKey.from_string(bytes.fromhex(private_key), curve=SECP256k1)
            self.signature = sk.sign(self.txid.encode()).hex()

    def verify(self):
        if not self.signature:
            return False
        if self.signature == "system_signature":
            return True
        vk = VerifyingKey.from_string(bytes.fromhex(self.sender), curve=SECP256k1)
        return vk.verify(bytes.fromhex(self.signature), self.txid.encode())
    
    def add_input(self, txid, output_index):
        self.inputs.append((txid, output_index))
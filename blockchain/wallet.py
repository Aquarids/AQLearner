from ecdsa import SigningKey, SECP256k1
from blockchain.transaction import Transaction

class Wallet:
    def __init__(self):
        self.private_key = SigningKey.generate(curve=SECP256k1)
        self.public_key = self.private_key.verifying_key
        self.balance = 0

    def get_address(self):
        return self.public_key.to_string().hex()
    
    def sign_transaction(self, transaction: Transaction):
        transaction.sign(self.private_key.to_string().hex())

    def add_balance(self, amount):
        self.balance += amount

    def subtract_balance(self, amount):
        self.balance -= amount

    def get_balance(self):
        return self.balance
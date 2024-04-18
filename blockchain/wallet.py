from ecdsa import SigningKey, SECP256k1
from blockchain.transaction import Transaction
from blockchain.utxo import UTXOSet

class Wallet:
    def __init__(self):
        self.private_key = SigningKey.generate(curve=SECP256k1)
        self.public_key = self.private_key.verifying_key

    def get_address(self):
        return self.public_key.to_string().hex()
    
    def sign_transaction(self, transaction: Transaction):
        transaction.sign(self.private_key.to_string().hex())

    def get_balance(self, utxo_set: UTXOSet):
        return utxo_set.get_balance(self.get_address())
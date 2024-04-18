from ecdsa import SigningKey, SECP256k1, VerifyingKey
from blockchain.utxo import UTXOSet
import hashlib
import time

class Transaction:
    def __init__(self, sender, inputs, outputs, timestamp=None, signature=None, is_coinbase=False, contract=None):
        self.sender = sender
        self.is_coinbase = is_coinbase
        self.timestamp = timestamp if timestamp else time.time()
        self.signature = signature

        self.inputs = inputs # List of tuples (txid, output_index)
        self.outputs = outputs # List of tuples (recipient, amount)
        self.contract = contract
        self.txid = self.calculate_txid()

    def calculate_txid(self):
        tx = f"{self.sender}{str(self.inputs)}{str(self.outputs)}{self.timestamp}{self.is_coinbase}"
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
    
    def check_blance(self, utxo_set: UTXOSet):
        total_input_value = 0
        for txid, idx in self.inputs:
            total_input_value += utxo_set.get_utxo(txid, idx)[0]
        total_output_value = sum(amount for _, amount in self.outputs)
        return total_input_value >= total_output_value

    def is_valid(self, utxo_set: UTXOSet):
        if not self.verify():
            return False
        if self.sender != "system":
            return self.check_blance(utxo_set)
        return True

    def serialize(self):
        return {
            "sender": self.sender,
            "inputs": [{"txid": txid, "index": index} for txid, index in self.inputs],
            "outputs": [{"amount": amt, "recipient": rec} for amt, rec in self.outputs],
            "timestamp": self.timestamp,
            "signature": self.signature,
            "is_coinbase": self.is_coinbase,
            "contract": self.contract,
            "txid": self.txid,
        }
    
    def deserialize(data):
        return Transaction(
            sender=data["sender"],
            inputs=[(input_data["txid"], input_data["index"]) for input_data in data["inputs"]],
            outputs=[(output["amount"], output["recipient"]) for output in data["outputs"]],
            timestamp=data["timestamp"],
            signature=data["signature"],
            is_coinbase=data["is_coinbase"],
            contract=data["contract"],
        )

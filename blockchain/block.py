import hashlib
import time

from blockchain.transaction import Transaction

class Block:
    def __init__(self, index, transactions, previous_hash, difficulty, nonce=0, timestamp=None):
        self.index = index
        self.timestamp = timestamp if timestamp else time.time()
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.difficulty = difficulty
        self.nonce = nonce

        self.merkle_root = self.create_merkle_root(transactions)
        self.hash = self.calculate_hash()

    def create_merkle_root(self, transactions):
        if not transactions:
            return None
        
        tx_ids = [tx.txid for tx in transactions]
        if len(tx_ids) % 2 != 0:
            tx_ids.append(tx_ids[-1]) # if odd, duplicate the last element
        
        while len(tx_ids) > 1:
            new_level = []
            for i in range(0, len(tx_ids), 2):
                pair = tx_ids[i] + (tx_ids[i + 1] if i + 1 < len(tx_ids) else tx_ids[i])
                new_hash = hashlib.sha256(pair.encode()).hexdigest()
                new_level.append(new_hash)
            tx_ids = new_level

        return tx_ids[0]

    def calculate_hash(self):
        block_header = f"{self.index}{self.timestamp}{self.merkle_root}{self.previous_hash}{self.nonce}{self.difficulty}"
        return hashlib.sha256(block_header.encode()).hexdigest()
    
    def mine_block(self):
        target = "0" * self.difficulty
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.calculate_hash()
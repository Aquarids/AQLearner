from blockchain.blockchain import Blockchain
from blockchain.block import Block

class FullNode:
    def __init__(self, blockchain: Blockchain):
        self.blockchain = blockchain
    
    def _add_block(self, block: Block):
        if self.blockchain.validate_new_block(block):
            self.blockchain.add_block(block)
            return True
        return False
    
    def verify_chain(self):
        if self.blockchain.is_chain_valid():
            return True
        return False
    
    def receive_block(self, new_block: Block):
        if self._add_block(new_block):
            return True
        return False
    
    def receive_transaction(self, new_transaction):
        if self.blockchain.add_transaction(new_transaction):
            return True
        return False
    
class LightNode:
    def __init__(self):
        self.headers = [] # light node only stores block headers
    
    def receive_block(self, new_block: Block):
        self.headers.append(new_block.hash)
        print(f"Block {new_block.index} added to the light node headers")

    def get_headers(self):
        return self.headers
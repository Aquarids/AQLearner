from blockchain.block import Block
from blockchain.transaction import Transaction 
from blockchain.node import FullNode, LightNode

import json

# just for assuming the blockchain network
class Net:
    def __init__(self):
        self.full_nodes: list[FullNode] = []
        self.light_nodes: list[LightNode] = []

    def add_full_node(self, node: FullNode):
        self.full_nodes.append(node)

    def add_light_node(self, node: LightNode):
        self.light_nodes.append(node)

    def broadcast_block(self, block: Block, sender: FullNode):
        # simulate the json network
        block_data = self._serialize_block(block)

        for id, node in enumerate(self.full_nodes):
            if node != sender:
                new_block = self._deserialize_block(block_data)
                print(f"Node {id} Receiving block {new_block.index}")
                result = node.receive_block(new_block)
                if (result):
                    print(f"Block {new_block.index} added to the blockchain by node {id}")
                else:
                    print(f"Block {new_block.index} rejected by node {id}")

        for node in self.light_nodes:
            new_block = self._deserialize_block(block_data)
            node.receive_block(new_block)

    def broadcast_transaction(self, transaction, sender: FullNode):
        transaction_data = self._serialize_transaction(transaction)

        for id, node in enumerate(self.full_nodes):
            if node != sender:
                print(f"Node {id} Receiving transaction {transaction_data['txid']}")
                new_transaction = self._deserialize_transaction(transaction_data)
                result = node.receive_transaction(new_transaction)
                if (result):
                    print(f"Transaction {new_transaction.txid} added to the blockchain by node {id}")
                else:
                    print(f"Transaction {new_transaction.txid} rejected by node {id}")

    # should not broadcast wallet, just for simplfy the test
    def broadcast_wallet(self, wallet):
        for node in self.full_nodes:
            node.blockchain.add_wallet(wallet)

    def _serialize_block(self, block: Block):
        return {
            "index": block.index,
            "transactions": block.transactions,
            "previous_hash": block.previous_hash,
            "difficulty": block.difficulty,
            "nonce": block.nonce,
            "timestamp": block.timestamp,
            "hash": block.hash,
        }
    
    def _deserialize_block(self, block_data):
        return Block(
            index=block_data["index"],
            transactions=block_data["transactions"],
            previous_hash=block_data["previous_hash"],
            difficulty=block_data["difficulty"],
            nonce=block_data["nonce"],
            timestamp=block_data["timestamp"],
        )
    
    def _serialize_transaction(self, transaction: Transaction):
        return {
            "txid": transaction.txid,
            "sender": transaction.sender,
            "recipient": transaction.recipient,
            "amount": transaction.amount,
            "signature": transaction.signature,
            "timestamp": transaction.timestamp,
            "is_coinbase": transaction.is_coinbase,
        }
    
    def _deserialize_transaction(self, transaction_data):
        return Transaction(
            sender=transaction_data["sender"],
            recipient=transaction_data["recipient"],
            amount=transaction_data["amount"],
            timestamp=transaction_data["timestamp"],
            signature=transaction_data["signature"],
            is_coinbase=transaction_data["is_coinbase"],
        )

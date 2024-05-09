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
                    print(
                        f"Block {new_block.index} added to the blockchain by node {id}"
                    )
                else:
                    print(f"Block {new_block.index} rejected by node {id}")

        for node in self.light_nodes:
            new_block = self._deserialize_block(block_data)
            node.receive_block(new_block)

    def broadcast_transaction(self, transaction, sender: FullNode):
        transaction_data = self._serialize_transaction(transaction)

        for id, node in enumerate(self.full_nodes):
            if node != sender:
                print(
                    f"Node {id} Receiving transaction {transaction_data['txid']}"
                )
                new_transaction = self._deserialize_transaction(
                    transaction_data)
                result = node.receive_transaction(new_transaction)
                if (result):
                    print(
                        f"Transaction {new_transaction.txid} added to the blockchain by node {id}"
                    )
                else:
                    print(
                        f"Transaction {new_transaction.txid} rejected by node {id}"
                    )

    def _serialize_block(self, block: Block):
        return block.serialize()

    def _deserialize_block(self, block_data):
        return Block.deserialize(block_data)

    def _serialize_transaction(self, transaction: Transaction):
        return transaction.serialize()

    def _deserialize_transaction(self, transaction_data):
        return Transaction.deserialize(transaction_data)

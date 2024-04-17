from blockchain.blockchain import Blockchain
from blockchain.block import Block
from blockchain.node import FullNode, LightNode
from blockchain.transaction import Transaction
from blockchain.wallet import Wallet
from blockchain.net import Net

import unittest

class TestBlockchain(unittest.TestCase):
    def _transaction(self, sender: Wallet, recipient:Wallet, amount, sender_node):
        tx = Transaction(sender.get_address(), recipient.get_address(), amount)
        sender.sign_transaction(tx)

        self.full_node_1.blockchain.add_transaction(tx)
        self.net.broadcast_transaction(tx, sender_node)

        block = self.full_node_1.blockchain.mine_transactions(sender.get_address())
        if block:
            print(f"Block {block.index} mined")
            self.net.broadcast_block(block, self.full_node_1)
        else:
            print("No block mined because no enough transactions")

    def test_bloch_chain(self):
        # assuming the blockchain network with 2 full nodes and 2 light nodes
        self.full_node_1 = FullNode(Blockchain(4))
        # genesis block is fixed here, thus no difference
        self.full_node_2 = FullNode(Blockchain(4))
        assert self.full_node_1.blockchain.chain[0].hash == self.full_node_2.blockchain.chain[0].hash

        self.light_node_1 = LightNode()
        self.light_node_2 = LightNode()

        self.net = Net()
        self.net.add_full_node(self.full_node_1)
        self.net.add_full_node(self.full_node_2)
        self.net.add_light_node(self.light_node_1)
        self.net.add_light_node(self.light_node_2)

        # assuming create two wallets at node 1
        wallet_a = self.full_node_1.blockchain.create_wallet()
        wallet_b = self.full_node_1.blockchain.create_wallet()
        print(f"Wallet A address: {wallet_a.get_address()}")
        print(f"Wallet B address: {wallet_b.get_address()}")

        # should not happen in real blockchain network
        self.net.broadcast_wallet(wallet_a)
        self.net.broadcast_wallet(wallet_b)
        
        self.full_node_1.blockchain.mine_coinbase(wallet_a.get_address())
        print(f"Wallet A balance: {self.full_node_1.blockchain.get_wallet(wallet_a.get_address()).get_balance()}")
        print(f"Wallet B balance: {self.full_node_1.blockchain.get_wallet(wallet_b.get_address()).get_balance()}")
        self.net.broadcast_block(self.full_node_1.blockchain.get_latest_block(), self.full_node_1)

        # assuming transactions between wallet A and wallet B
        self._transaction(wallet_a, wallet_b, 7, self.full_node_1)
        self._transaction(wallet_a, wallet_b, 2, self.full_node_1)
        self._transaction(wallet_a, wallet_b, 1, self.full_node_1)
        self._transaction(wallet_b, wallet_a, 5, self.full_node_1)
        self._transaction(wallet_a, wallet_b, 3, self.full_node_1)
        self._transaction(wallet_a, wallet_b, 3, self.full_node_1)
        print(f"Wallet A balance: {wallet_a.get_balance()}")
        print(f"Wallet B balance: {wallet_b.get_balance()}")
        print(f"Node 1 Block count: {len(self.full_node_1.blockchain.chain)}")
        print(f"Node 2 Block count: {len(self.full_node_2.blockchain.chain)}")

        
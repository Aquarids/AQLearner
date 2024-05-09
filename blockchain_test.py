from blockchain.blockchain import Blockchain
from blockchain.block import Block
from blockchain.node import FullNode, LightNode
from blockchain.transaction import Transaction
from blockchain.wallet import Wallet
from blockchain.utxo import UTXOSet
from blockchain.net import Net

import unittest


class TestBlockchain(unittest.TestCase):

    def _transaction(self, sender: Wallet, recipient: Wallet, amount,
                     utxo: UTXOSet, sender_node):
        inputs = []
        outputs = [(recipient.get_address(), amount)]
        utxos = utxo.find_utxos(sender.get_address())

        cur_amt = 0
        for txid, idx, amt in utxos:
            if cur_amt >= amount:
                break
            inputs.append((txid, idx))
            cur_amt += amt
        if cur_amt < amount:
            raise Exception("Not enough funds")

        change = cur_amt - amount
        if change > 0:
            outputs.append((sender.get_address(), change))

        tx = Transaction(sender.get_address(), inputs, outputs)
        sender.sign_transaction(tx)

        self.full_node_1.blockchain.add_transaction(tx)
        self.net.broadcast_transaction(tx, sender_node)

        block = self.full_node_1.blockchain.mine_transactions(
            sender.get_address())
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
        assert self.full_node_1.blockchain.chain[
            0].hash == self.full_node_2.blockchain.chain[0].hash

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

        self.full_node_1.blockchain.mine_coinbase(wallet_a.get_address())
        print(
            f"Wallet A balance: {wallet_a.get_balance(self.full_node_1.blockchain.utxo_set)}"
        )
        print(
            f"Wallet B balance: {wallet_b.get_balance(self.full_node_1.blockchain.utxo_set)}"
        )

        self.net.broadcast_block(
            self.full_node_1.blockchain.get_latest_block(), self.full_node_1)

        # assuming transactions between wallet A and wallet B
        utxo = self.full_node_1.blockchain.utxo_set
        self._transaction(wallet_a, wallet_b, 7, utxo, self.full_node_1)
        print(f"Round 1 Wallet A balance: {wallet_a.get_balance(utxo)}")
        print(f"Round 1 Wallet B balance: {wallet_b.get_balance(utxo)}")
        self._transaction(wallet_a, wallet_b, 2, utxo, self.full_node_1)
        print(f"Round 2 Wallet A balance: {wallet_a.get_balance(utxo)}")
        print(f"Round 2 Wallet B balance: {wallet_b.get_balance(utxo)}")
        self._transaction(wallet_a, wallet_b, 1, utxo, self.full_node_1)
        print(f"Round 3 Wallet A balance: {wallet_a.get_balance(utxo)}")
        print(f"Round 3 Wallet B balance: {wallet_b.get_balance(utxo)}")
        self._transaction(wallet_b, wallet_a, 5, utxo, self.full_node_1)
        print(f"Round 4 Wallet A balance: {wallet_a.get_balance(utxo)}")
        print(f"Round 4 Wallet B balance: {wallet_b.get_balance(utxo)}")
        self._transaction(wallet_a, wallet_b, 3, utxo, self.full_node_1)
        print(f"Round 5 Wallet A balance: {wallet_a.get_balance(utxo)}")
        print(f"Round 5 Wallet B balance: {wallet_b.get_balance(utxo)}")
        self._transaction(wallet_a, wallet_b, 3, utxo, self.full_node_1)
        print(f"Round 6 Wallet A balance: {wallet_a.get_balance(utxo)}")
        print(f"Round 6 Wallet B balance: {wallet_b.get_balance(utxo)}")
        print(f"Node 1 Block count: {len(self.full_node_1.blockchain.chain)}")
        print(f"Node 2 Block count: {len(self.full_node_2.blockchain.chain)}")

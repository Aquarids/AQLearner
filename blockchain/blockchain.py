from blockchain.block import Block
from blockchain.transaction import Transaction
from blockchain.wallet import Wallet
from blockchain.utxo import UTXOSet

const_coinbase_reward = 20
const_mining_reward = 10
const_block_size = 4


class Blockchain:

    def __init__(self, difficulty):
        self.difficulty = difficulty
        self.pending_transactions = []
        self.utxo_set = UTXOSet()

        self.chain = [self.create_genesis_block()]

    def create_wallet(self):
        wallet = Wallet()
        return wallet

    def create_genesis_block(self):
        genesis_transaction = Transaction(sender="system",
                                          inputs=[],
                                          outputs=[("system", 50)],
                                          timestamp=1710000000)
        block = Block(
            index=0,
            transactions=[genesis_transaction],
            previous_hash="0",
            difficulty=self.difficulty,
            nonce=0,
            timestamp=1710000000,
        )
        self.utxo_set.add_utxo(genesis_transaction.txid, 0, 0, "system")
        return block

    def get_latest_block(self):
        return self.chain[-1]

    def _clear_pending_transactions(self):
        self.pending_transactions = []

    def mine_coinbase(self, miner_address):
        inputs = []
        outputs = [(miner_address, const_coinbase_reward)]
        coinbase_tx = Transaction("system", inputs, outputs, is_coinbase=True)
        self.utxo_set.add_transaction(coinbase_tx)

        transactions = [coinbase_tx] + self.pending_transactions
        new_block = Block(len(self.chain), transactions,
                          self.get_latest_block().hash, self.difficulty)
        new_block.mine_block()
        self.chain.append(new_block)

        self._clear_pending_transactions()

    def mine_transactions(self, miner_address):
        inputs = []
        outputs = [(miner_address, const_mining_reward)]
        reward_tx = Transaction("system", inputs, outputs)
        reward_tx.sign('system')  # System signature (simplified)

        valid_transactions = [
            tx for tx in self.pending_transactions if tx.verify()
        ]
        valid_transactions.append(reward_tx)

        # Assuming enough transactions are collected
        if len(valid_transactions) >= const_block_size:
            self.utxo_set.add_transaction(reward_tx)

            new_block = Block(len(self.chain), valid_transactions,
                              self.get_latest_block().hash, self.difficulty)
            new_block.mine_block()
            self.chain.append(new_block)

            self._clear_pending_transactions()
            return new_block
        else:
            return None

    def add_transaction(self, tx: Transaction):
        if not tx.inputs or not tx.outputs:
            print("Transaction must include from and to addresses")
            return False
        if not tx.is_valid(self.utxo_set):
            print("Transaction is invalid")
            return False

        self.utxo_set.add_transaction(tx)
        self.pending_transactions.append(tx)
        return True

    def add_block(self, block: Block):
        block.previous_hash = self.get_latest_block().hash
        block.mine_block()
        if self.validate_new_block(block):
            for tx in block.transactions:
                self.utxo_set.add_transaction(tx)
                if tx in self.pending_transactions:
                    self.pending_transactions.remove(tx)
            self.chain.append(block)

    def validate_new_block(self, block: Block):
        last_block = self.get_latest_block()
        if block.previous_hash != last_block.hash:
            return False
        if not block.hash.startswith("0" * self.difficulty):
            return False
        return True

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            if current_block.previous_hash != previous_block.hash:
                return False
            if not current_block.hash.startswith("0" * self.difficulty):
                return False
        return True

from blockchain.block import Block
from blockchain.transaction import Transaction
from blockchain.wallet import Wallet

const_coinbase_reward = 20
const_mining_reward = 10
const_block_size = 4

class Blockchain:
    def __init__(self, difficulty):
        self.difficulty = difficulty
        self.pending_transactions = []
        self.wallets = {} # wallet should not be stored in blockchain, just for simplify the test

        self.chain = [self.create_genesis_block()]

    def create_wallet(self):
        wallet = Wallet()
        self.wallets[wallet.get_address()] = wallet
        return wallet
    
    def add_wallet(self, wallet: Wallet):
        if wallet.get_address() not in self.wallets:
            self.wallets[wallet.get_address()] = wallet
    
    def get_wallet(self, address) -> Wallet:
        return self.wallets.get(address)
    
    def get_all_wallets(self):
        return self.wallets

    def create_genesis_block(self):
        return Block(
            index=0,
            transactions=[
                Transaction(
                    sender="system",
                    recipient="system",
                    amount=0,
                    timestamp=1710000000,
                )
            ],
            previous_hash="0",
            difficulty=self.difficulty,
            nonce=0,
            timestamp=1710000000,
        )
    
    def get_latest_block(self):
        return self.chain[-1]
    
    def _clear_pending_transactions(self):
        self.pending_transactions = []

    def _modify_wallet_balance(self, transactions=[]):
        for transaction in transactions:
            if transaction.sender == "system":
                continue
            sender_wallet = self.get_wallet(transaction.sender)
            recipient_wallet = self.get_wallet(transaction.recipient)
            sender_wallet.subtract_balance(transaction.amount)
            recipient_wallet.add_balance(transaction.amount)
    
    def mine_coinbase(self, miner_address):
        coinbase_tx = Transaction("system", miner_address, const_coinbase_reward, is_coinbase=True)
        transactions = [coinbase_tx] + self.pending_transactions
        new_block = Block(len(self.chain), transactions, self.get_latest_block().hash, self.difficulty)
        new_block.mine_block()
        self.chain.append(new_block)
        self.get_wallet(miner_address).add_balance(const_coinbase_reward)
        self._clear_pending_transactions()
        
    def mine_transactions(self, miner_address):
        reward_tx = Transaction("system", miner_address, const_mining_reward)
        reward_tx.sign('system')  # System signature (simplified)

        valid_transactions = [tx for tx in self.pending_transactions if tx.verify()]
        valid_transactions.append(reward_tx)
        
        # Assuming enough transactions are collected
        if len(valid_transactions) >= const_block_size:
            new_block = Block(len(self.chain), valid_transactions, self.get_latest_block().hash, self.difficulty)
            new_block.mine_block()
            self.chain.append(new_block)

            self.get_wallet(miner_address).add_balance(const_mining_reward)
            self._modify_wallet_balance(valid_transactions)
            self._clear_pending_transactions()
            return new_block
        else:
            return None
    
    def add_transaction(self, transaction: Transaction):
        if not transaction.sender or not transaction.recipient:
            print("Transaction must include from and to addresses")
            return False
        if not transaction.verify():
            print("Transaction signature is invalid")
            return False
        sender_balance = self.get_wallet(transaction.sender).get_balance()
        if sender_balance < transaction.amount:
            print("Not enough balance")
            return False

        self.pending_transactions.append(transaction)
        return True
    
    def add_block(self, block: Block):
        block.previous_hash = self.get_latest_block().hash
        block.mine_block()
        if self.validate_new_block(block):
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
        
    

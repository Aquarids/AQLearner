from blockchain.transaction import Transaction

class SmartContract:
    def __init__(self, code):
        self.code = compile(code, '<string>', 'exec')

    def execute(self, context):
        exec(self.code, context)

class ContractTransaction(Transaction):
    def __init__(self, sender, inputs, outputs, timestamp=None, signature=None, is_coinbase=False, contract=None):
        super().__init__(sender, inputs, outputs, timestamp, signature, is_coinbase)
        self.contract = SmartContract(contract)

    def apply(self, blockchain):
        context = {'blockchain': blockchain, 'transaction': self}
        self.contract.execute(context)
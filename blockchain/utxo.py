class UTXOSet:

    def __init__(self):
        self.utxos = {}

    def add_transaction(self, tx):
        for txid, index in tx.inputs:
            self.remove_utxo(txid, index)
        for i, (address, amount) in enumerate(tx.outputs):
            self.add_utxo(tx.txid, i, amount, address)

    def withdraw_transaction(self, tx):
        for txid, index in tx.inputs:
            self.add_utxo(txid, index, tx.outputs[index].amount,
                          tx.outputs[index].address)
        for i, (address, amount) in enumerate(tx.outputs):
            self.remove_utxo(tx.txid, i)

    def add_utxo(self, txid, output_index, amount, address):
        self.utxos[(txid, output_index)] = (amount, address)

    def remove_utxo(self, txid, index):
        if (txid, index) in self.utxos:
            del self.utxos[(txid, index)]

    def contains(self, txid, index):
        return (txid, index) in self.utxos

    def get_utxo(self, txid, index):
        return self.utxos.get((txid, index), None)

    def find_utxos(self, address):
        return [(txid, idx, amt)
                for (txid, idx), (amt, addr) in self.utxos.items()
                if addr == address]

    def get_balance(self, address):
        return sum(amt for (txid, idx, amt) in self.find_utxos(address))

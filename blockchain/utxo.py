class UTXOSet:
    def __init__(self):
        self.utxos = {}

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
        return [(txid, idx, amt) for (txid, idx), (amt, addr) in self.utxos.items() if addr == address]
    
    def get_balance(self, address):
        return sum(amt for (txid, idx, amt) in self.find_utxos(address))


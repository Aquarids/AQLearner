class SimplePSI:
    def __init__(self):
        self.dic = []
        pass

    def build_dic(self, data):
        for key in data:
            if key not in self.dic:
                self.dic.append(key)

    def psi(self, hashed_data):
        common_keys = []
        for key_hashed in hashed_data:
            for key in self.dic:
                if key_hashed == hash(key):
                   common_keys.append(key)
        return common_keys

import os
import random

# a simple XOR secret share
class SecretShare:
    def __init__(self, num_shares):
        self.num_shares = num_shares
        self.secret_type = None
        if num_shares < 2:
            raise ValueError("Require at least 2 shares")
        
    def _convert_secret(self, secret):
        if isinstance(secret, str):
            secret = secret.encode('utf-8')
            self.secret_type = "str"
        elif isinstance(secret, int):
            num_bytes = (secret.bit_length() + 7) // 8
            secret = secret.to_bytes(num_bytes, 'big')
            self.secret_type = "int"
        else:
            raise ValueError("Only support string or integer secret")
        return secret
    
    def _undo_convert_secret(self, secret):
        if self.secret_type == "str":
            return secret.decode('utf-8')
        elif self.secret_type == "int":
            return int.from_bytes(secret, 'big')

    def generate_shares(self, secret):
        secret = self._convert_secret(secret)

        shares = [os.urandom(len(secret)) for _ in range(self.num_shares - 1)]

        last = secret
        for share in shares:
            last = bytes([a ^ b for a, b in zip(last, share)])
        shares.append(last)

        return shares
    
    def recover_secret(self, shares):
        if len(shares) < self.num_shares:
            raise ValueError("Not enough shares to recover secret")
        
        secret = bytes([0] * len(shares[0]))
        for share in shares:
            secret = bytes([a ^ b for a, b in zip(secret, share)])

        return self._undo_convert_secret(secret)
        
class ShamirSecretShare(SecretShare):
    def __init__(self, num_shares, prime):
        self.num_shares = num_shares
        self.secret_type = None
        self.prime = prime
        if num_shares < 3:
            raise ValueError("Require at least 3 shares for 2 out of N secret sharing")
    
    def _convert_secret(self, secret):
        if isinstance(secret, str):
            secret = int.from_bytes(secret.encode('utf-8'), 'big')
            self.secret_type = "str"
        elif isinstance(secret, int):
            self.secret_type = "int"
        else:
            raise ValueError("Only support string or integer secret")
        return secret
    
    def _undo_convert_secret(self, secret):
        if self.secret_type == "str":
            return secret.to_bytes((secret.bit_length() + 7) // 8, 'big').decode('utf-8')
        elif self.secret_type == "int":
            return secret

    def generate_shares(self, secret):
        secret = self._convert_secret(secret)

        a1 = random.randint(1, self.prime-1)  # Random coefficient for x
        shares = [(i, (secret + a1 * i) % self.prime) for i in range(1, self.num_shares + 1)]
        return shares
    
    def recover_secret(self, shares):
        if len(shares) < 2:
            raise ValueError("Not enough shares to recover secret")
        
        x1, y1 = shares[0]
        x2, y2 = shares[1]
        secret = (y1 * (x2 % self.prime) - y2 * (x1 % self.prime)) * pow(x2 - x1, -1, self.prime) % self.prime
    
        return self._undo_convert_secret(secret)
        
        
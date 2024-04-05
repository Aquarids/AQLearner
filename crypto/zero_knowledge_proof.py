import random
import math

# a simple zero knowledge proof
class ZeroKnowledgeProof:
    def __init__(self, secret, r):
        self.secret = secret
        self.r = r
        self.v = secret ** 2 # public value v = x^2
        self.x = self.r ** 2 # commitment value x = r^2

        print("Secret:", self.secret)
        print("R:", self.r)
        print("V:", self.v)
        print("Commitment:", self.x)

    def get_commitment(self):
        return self.x
    
    def challenge(self):
        challenge_type = random.choice(['root_x', 'root_xv_inv'])
        return challenge_type
    
    def response(self, challenge_type=None):
        response = 0
        if challenge_type == 'root_x':
            response = self.r # return the square root of x
        elif challenge_type == 'root_xv_inv':
            response = self.r * pow(self.secret, -1) # return the square root of x * v^-1
        else:
            raise ValueError("Invalid challenge type")
        print("Cauculate for response", challenge_type, response)
        return response
        
    def verify(self, challenge_type, response):
        calc_value = 0
        verified_value = -1
        if challenge_type == 'root_x':
            calc_value = response ** 2
            verified_value = self.x
        elif challenge_type == 'root_xv_inv':
            v_inv = pow(self.v, -1)
            xv_inv = self.x * v_inv
            calc_value = response ** 2
            verified_value = xv_inv
        else:
            raise ValueError("Invalid challenge type")
        print("Verify for response", challenge_type, calc_value, verified_value)
        return calc_value == verified_value

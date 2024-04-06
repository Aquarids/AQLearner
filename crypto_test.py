import unittest
import Crypto.Util.number
import random

from crypto.zero_knowledge_proof import ZeroKnowledgeProof
from crypto.oblivious_transfer import OTSender, OTReceiver
from crypto.garbled_circuit import GarbledCircuit
from crypto.secret_share import SecretShare
from crypto.secret_share import ShamirSecretShare
from crypto.homo_encryption import AddHomomorphicEncryption
from crypto.homo_encryption import ElGamalHE

class TestZeroKnowledgeProof(unittest.TestCase):
    def test_zkp(self):
        secret, r = random.randint(1, 100), random.randint(1, 100)
        zkp = ZeroKnowledgeProof(secret, r)

        challenge_type = zkp.challenge()
        response = zkp.response(challenge_type)
        self.assertTrue(zkp.verify(challenge_type, response))

class TestObliviousTransfer(unittest.TestCase):
    def test_ot(self):
        p, g = Crypto.Util.number.getPrime(512), 2
        messages = ["Hello", "World"]

        sender = OTSender(p, g, messages)
        choice = random.choice([0, 1]) # 0 or 1

        receiver = OTReceiver(p, g, choice)

        sender_publickeys = sender.get_public_keys()
        receiver_publickey = receiver.get_public_key()
        print("Sender Public Keys:", sender_publickeys)
        print("Receiver Public Key:", receiver_publickey)

        encryped_messages = sender.encrypt(receiver_publickey)
        print("Encrypted Messages:", encryped_messages)

        # here should ensure receiver can only get the message he wants and sender do not know which message receiver get
        sender_publickey = sender_publickeys[choice] 
        decrypted_message, cannot_decrypt_message = receiver.decrypt(encryped_messages, sender_publickey)
        print("Decrypted Message:", decrypted_message)
        print("Cannot Decrypt Message:", cannot_decrypt_message)
        self.assertEqual(messages[choice], decrypted_message)

class TestGarbledCircuit(unittest.TestCase):
    def test_garbled_circuit(self):
        gc = GarbledCircuit()
        table = gc.garble_circuit()
        print("Garbled Table:", table)

        a, b = 'A1', 'B1'
        encrypted_output = table[(a, b)]
        print("Encrypted Output:", encrypted_output)
        output = gc.evaluate(table, gc.keys[a], gc.keys[b])
        print("Evaluate Output:", output)
        if a == 'A1' and b == 'B1':
            self.assertEqual(output, 1)
        else:
            self.assertEqual(output, 0)

    def test_gc_with_ob(self):
        p, g = Crypto.Util.number.getPrime(512), Crypto.Util.number.getPrime(512)
        alice_inputs = ['A0', 'A1']
        bob_inputs = ['B0', 'B1']
        alice_choice = random.choice([0, 1])
        bob_choice = random.choice([0, 1])

        gc = GarbledCircuit()
        table = gc.garble_circuit()

        # asume alice is the sender and bob is the receiver
        alice_messages = [gc.keys[alice_inputs[alice_choice]].hex(), gc.keys[alice_inputs[1 - alice_choice]].hex()]
        alice_ot = OTSender(p, g, alice_messages)
        bob_ot = OTReceiver(p, g, bob_choice)

        bob_publickey = bob_ot.get_public_key()
        alice_encrypted_messages = alice_ot.encrypt(bob_publickey)

        bob_decrypted_message, _ = bob_ot.decrypt(alice_encrypted_messages, alice_ot.get_public_keys()[bob_choice])
        bob_key = bytes.fromhex(bob_decrypted_message)
        result = gc.evaluate(table, bob_key, gc.keys[bob_inputs[bob_choice]])

        if alice_choice == 1 and bob_choice == 1:
            self.assertEqual(result, 1)
        else:
            self.assertEqual(result, 0)

class TestSecretShare(unittest.TestCase):
    def test_secret_share(self):
        secret_share = SecretShare(num_shares=5)
        secret = random.randint(1, 100)
        print("Original Secret:", secret)

        shares = secret_share.generate_shares(secret)
        print("Shares:", shares)

        recovered_secret = secret_share.recover_secret(shares)
        print("Recovered Secret:", recovered_secret)

        self.assertEqual(secret, recovered_secret)
    
    def test_two_out_of_n_secret_share(self):
        prime = Crypto.Util.number.getPrime(512)
        secret_share = ShamirSecretShare(num_shares=5, prime=prime)
        secret = "This is a secret message."
        print("Original Secret:", secret)

        shares = secret_share.generate_shares(secret)
        print("Shares:", shares)

        recovered_shares = random.sample(shares, 2)
        recovered_secret = secret_share.recover_secret(recovered_shares)
        print("Recovered Secret:", recovered_secret)

class TestHomomorphicEncryption(unittest.TestCase):
    def test_homomorphic_encryption(self):

        he = AddHomomorphicEncryption()

        a, b, c = random.randint(1, 100), random.randint(1, 100), random.randint(1, 100)
        encrypted_a, encrypted_b, encrypted_c = he.encrypt(a), he.encrypt(b), he.encrypt(c)
        print("Encrypted A:", encrypted_a)
        print("Encrypted B:", encrypted_b)
        print("Encrypted C:", encrypted_c)

        encrypted_sum = he.add([encrypted_a, encrypted_b, encrypted_c])
        print("Encrypted Sum:", encrypted_sum)
        decrypted_sum = he.decrypt(encrypted_sum)
        print("Decrypted Sum:", decrypted_sum)
        self.assertEqual(a + b + c, decrypted_sum)

    def test_elgamal_homomorphic_encryption(self):
        he = ElGamalHE(bits=512)

        a, b = random.randint(1, 100), random.randint(1, 100)
        encrypted_a, encrypted_b = he.encrypt(a), he.encrypt(b)
        print("Encrypted A:", encrypted_a)
        print("Encrypted B:", encrypted_b)

        encrypted_product = he.multiply(encrypted_a, encrypted_b)
        print("Encrypted Product:", encrypted_product)
        decrypted_product = he.decrypt(encrypted_product)
        print("Decrypted Product:", decrypted_product)
        self.assertEqual(a * b, decrypted_product)


if __name__ == '__main__':
    unittest.main()
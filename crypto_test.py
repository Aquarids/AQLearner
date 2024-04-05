import unittest
import Crypto.Util.number
import random

from crypto.zero_knowledge_proof import ZeroKnowledgeProof
from crypto.oblivious_transfer import OTSender, OTReceiver
from crypto.garbled_circuit import GarbledCircuit


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

if __name__ == '__main__':
    unittest.main()
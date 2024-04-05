import unittest
import Crypto.Util.number
import random

from crypto.zero_knowledge_proof import ZeroKnowledgeProof
from crypto.oblivious_transfer import OTSender, OTReceiver
from crypto.zero_knowledge_proof import ZeroKnowledgeProof

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
        choice = 1 # 0 or 1

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

if __name__ == '__main__':
    unittest.main()
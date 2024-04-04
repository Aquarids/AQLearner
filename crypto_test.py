import unittest
import Crypto.Util.number

from crypto.ot.oblivious_transfer import OTSender, OTReceiver

class TestObliviousTransfer(unittest.TestCase):
    def test_ot(self):
        p, g = Crypto.Util.number.getPrime(512), 2
        messages = ["Hello", "World"]

        sender = OTSender(p, g, messages)
        choices = 1 # 0 or 1
        receiver = OTReceiver(p, g, choices)

        sender_publickeys = sender.get_public_keys()
        receiver_publickey = receiver.get_public_key()
        print("Sender Public Keys:", sender_publickeys)
        print("Receiver Public Key:", receiver_publickey)

        encryped_messages = sender.encrypt(receiver_publickey)
        print("Encrypted Messages:", encryped_messages)

        decrypted_message, cannot_decrypt_message = receiver.decrypt(encryped_messages, sender_publickeys)
        print("Decrypted Message:", decrypted_message)
        print("Cannot Decrypt Message:", cannot_decrypt_message)

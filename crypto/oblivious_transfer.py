from crypto.diffie_hellman import DiffieHellman
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

import os


class OTSender:

    def __init__(self, p, g, messages):
        if len(messages) != 2:
            raise ValueError("Only support 1 out of 2 OT")
        self.messages = messages

        self.dh1 = DiffieHellman(p, g)
        self.dh2 = DiffieHellman(p, g)

    def encrypt(self, receiver_public_key):
        secret1 = self.dh1.get_shared_secret(receiver_public_key)
        secret2 = self.dh2.get_shared_secret(receiver_public_key)

        encrypted_messages = []
        for secret, message in zip([secret1, secret2], self.messages):
            cipher = AES.new(secret, AES.MODE_CBC, os.urandom(16))
            encrypted_messages.append(
                (cipher.encrypt(pad(message.encode('utf-8'),
                                    AES.block_size)), cipher.iv))
        return encrypted_messages

    def get_public_keys(self):
        return self.dh1.get_public_key(), self.dh2.get_public_key()


class OTReceiver:

    def __init__(self, p, g, choice):
        self.choice = choice
        self.dh = DiffieHellman(p, g)

    def get_public_key(self):
        return self.dh.get_public_key()

    def decrypt(self, encrypted_messages, sender_public_key):
        if len(encrypted_messages) != 2:
            raise ValueError("Only support 1 out of 2 OT")

        shared_secret = self.dh.get_shared_secret(sender_public_key)
        cipher = AES.new(shared_secret, AES.MODE_CBC,
                         encrypted_messages[self.choice][1])
        decrypted_message = unpad(
            cipher.decrypt(encrypted_messages[self.choice][0]),
            AES.block_size).decode('utf-8')
        try:
            cannot_decrypt_message = unpad(
                cipher.decrypt(encrypted_messages[1 - self.choice][0]),
                AES.block_size).decode('utf-8')
        except ValueError:
            cannot_decrypt_message = "Cannot decrypt because Padding Error"
        return decrypted_message, cannot_decrypt_message

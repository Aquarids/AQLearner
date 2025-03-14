import unittest

from llm.llm_model import LLMModel
import os


class TestLLM(unittest.TestCase):

    def test_deepseek_7b(self):
        project_root = os.path.dirname(__file__)
        dir = os.path.join(project_root, "models")

        model = LLMModel("deepseek-7b", "deepseek-ai/deepseek-llm-7b-chat", dir)
        model.init()

        template = """<|begin▁of▁sentence|>User: {instruction}<|end▁of▁sentence|>\nAssistant:"""
        instruction = "请问你是谁？"

        response = model._talk(instruction, template)
        print(response)

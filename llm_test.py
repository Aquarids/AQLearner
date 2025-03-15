import unittest

from llm.llm_model import LLMModel
import os


class TestLLM(unittest.TestCase):

    def test_deepseek_7b(self):
        project_root = os.path.dirname(__file__)
        dir = os.path.join(project_root, "models")

        model = LLMModel("deepseek-ai/deepseek-llm-7b-chat", dir)
        model.init()

        instruction = "请问你是谁？"

        response = model.talk(instruction)
        print(response)

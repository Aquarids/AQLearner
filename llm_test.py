import unittest

from llm.llm_model import LLMModel
import os


class TestLLM(unittest.TestCase):

    def test_deepseek_7b(self):
        project_root = os.path.dirname(__file__)
        dir = os.path.join(project_root, "models")

        model = LLMModel("deepseek-ai/deepseek-llm-7b-chat", dir, use_cot=True, use_retrieval=False)
        model.init()

        instruction = "鸡兔同笼：一个笼子里关着若干只鸡和兔，头数共有35个，脚数共有94只，问笼中鸡和兔各有多少只？"

        response = model.talk(instruction)
        print(response)

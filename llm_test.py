import unittest

from llm.llm_model import LLMModel
import os
import gc
import torch


class TestLLM(unittest.TestCase):

    def test_deepseek_7b(self):
        project_root = os.path.dirname(__file__)
        dir = os.path.join(project_root, "models")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = LLMModel("deepseek-ai/deepseek-llm-7b-chat", dir, use_cot=True, use_retrieval=False, use_history=False)
        model.init()

        instruction = "What is the capital of France?"

        response = model.talk(instruction)
        print(response)

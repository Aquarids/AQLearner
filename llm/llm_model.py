from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_core.prompts import PromptTemplate

import os
import torch

class LLMModel:

    def __init__(self, model_name, model_id, dir):
        self.model_name = model_name
        self.model_id = model_id
        self.save_dir = dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.tokenizer = None

        self.llm = None

    def init(self):
        model, tokenizer = self._load_model(self.model_id)

        text_generation_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=128
        )
        self.llm = HuggingFacePipeline(pipeline=text_generation_pipe)

    def _load_model(self, model_id):
        model_path = self._model_path()
        if os.path.exists(model_path):
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            model, tokenizer = self._download_model(model_id, model_path)

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer
    
    def _quant_config(self):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

    def _download_model(self, model_id, model_path):
        model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", low_cpu_mem_usage=True, quantization_config=self._quant_config()
            )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        return model, tokenizer

    def _model_path(self):
        return f"{self.save_dir}/{self.model_name}"
    
    def _talk(self, instruction, template=None):
        if template is None:
            template = """基于以下指令生成文本：
                指令：{instruction}
                输出："""
        prompt = PromptTemplate(template=template, input_variables=["instruction"])
        response = prompt | self.llm
        result = response.invoke({"instruction": instruction})
        return result

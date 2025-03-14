from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

import os


class LLMModel:

    def __init__(self, model_name, model_id, dir):
        self.model_name = model_name
        self.model_id = model_id
        self.save_dir = dir

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
        self.llm = HuggingFacePipeline(text_generation_pipe)

    def _load_model(self, model_id):
        model_path = self._model_path()
        if os.path.exists(model_path):
            model = AutoModelForCausalLM.from_pretrained(
                model_path, load_in_4bit=True, device_map="auto", low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            model, tokenizer = self._download_model(model_id, model_path)

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    def _download_model(self, model_id, model_path):
        model = AutoModelForCausalLM.from_pretrained(
                model_id, load_in_4bit=True, device_map="auto", low_cpu_mem_usage=True
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
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(instruction=instruction)

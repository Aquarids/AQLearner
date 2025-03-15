from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_core.prompts import PromptTemplate
from .template_manager import TemplateManager
from langchain.memory import ConversationBufferWindowMemory

import os
import torch

class LLMModel:

    def __init__(self, model_id, dir):
        self.model_id = model_id
        self.save_dir = dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.template_manager = TemplateManager()
        self.template_config = None
        self.model_name = None

        self.model = None
        self.tokenizer = None
        self.llm = None
        self.template = None
        self.memory = None
        self.chain = None

    def init(self):
        model, tokenizer = self._load_model(self.model_id)

        text_generation_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=128,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        llm = HuggingFacePipeline(pipeline=text_generation_pipe)
        prompt = PromptTemplate.from_template(
            self.template_config.template,
        )
        memory = self._memory()
        
        self.llm = llm
        self.template = prompt
        self.memory = memory

        self.chain = (
            {"instruction": lambda x: x["instruction"], "history": memory.load_memory_variables}
            | prompt
            | llm
        )

    def _load_model(self, model_id):
        model_path = self._model_path()
        need_save = False
        if os.path.exists(model_path):
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            model, tokenizer = self._download_model(model_id, model_path)
            need_save = True

        self.template_config = self.template_manager.get_template_config(self.model_id, self.tokenizer)
        self.model_name = self.template_config.name

        tokenizer.bos_token = self.template_config.bos_token
        tokenizer.eos_token = self.template_config.eos_token
        tokenizer.pad_token = self.template_config.pad_token

        if need_save:
            self._save_model(model, tokenizer, model_path)

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
        return model, tokenizer
    
    def _save_model(self, model, tokenizer, model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

    def _model_path(self):
        return f"{self.save_dir}/{self.model_name}"
    
    def _memory(self):
        memory = ConversationBufferWindowMemory(
            k=10,
            memory_key="history",
            input_key="instruction"
        )
        return memory
    
    def talk(self, instruction):
        inputs = {"instruction": instruction}
        response = self.chain.invoke(inputs)
        self.memory.save_context(inputs, {"output": response})
        return response

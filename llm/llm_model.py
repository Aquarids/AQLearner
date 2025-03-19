from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables import RunnableParallel, RunnableLambda

from .template_manager import TemplateManager
from .rag import RAG

import os
import torch
from halo import Halo

class LLMModel:

    def __init__(self, model_id, dir, use_history=False, use_retrieval=False, use_cot=False):
        self.model_id = model_id
        self.save_dir = dir
        self.cache_dir = os.path.join(self.save_dir, "cache")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_history = use_history
        self.use_retrieval = use_retrieval
        self.use_cot = use_cot

        self.template_manager = TemplateManager()
        self.model_name = None

        self.model = None
        self.tokenizer = None
        self.llm = None
        self.template = None
        self.rag = None
        self.memory = None
        self.chain = None

    def init(self):
        if self.use_retrieval:
            self.rag = self._init_rag()

        if self.use_history:
            self.memory = self._init_memory()

        model, tokenizer = self._load_model(self.model_id)

        text_generation_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            truncation=True,
            return_full_text=False,
            temperature=0.9,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
        )
        llm = HuggingFacePipeline(pipeline=text_generation_pipe)

        template = self.template_manager.get_template(self.model_id, history=self.use_history, context=self.use_retrieval, cot=self.use_cot)
        print(template)
        prompt = PromptTemplate.from_template(template=template)

        self.llm = llm
        self.template = prompt

        input = self._build_dynamic_input(self.use_retrieval, self.use_history, self.use_cot)

        self.chain = (
            input
            | prompt
            | llm
        )

    def _init_memory(self):
        memory = ConversationBufferWindowMemory(
            k=1,
            memory_key="history",
            input_key="instruction"
        )
        return memory
    
    def _init_rag(self):
        rag = RAG(os.path.join(self.save_dir, "rag_cache"))
        return rag

    def _build_dynamic_input(
        self, 
        include_context: bool, 
        include_history: bool, 
        include_cot_reasoning: bool
    ) -> RunnableParallel:
        base_fields = {
            "instruction": lambda x: x['instruction']
        }

        conditional_fields = [
            ("context", include_context, 
            lambda x: self.rag.retrieve_fact(x['instruction'])),
            ("history", include_history,
            lambda x: self.memory.load_memory_variables(x)['history']),
            ("cot_reasoning", include_cot_reasoning,
            lambda x: self._generate_cot_reasoning(x['instruction']))
        ]

        dynamic_fields = {
            name: RunnableLambda(func) 
            for name, flag, func in conditional_fields 
            if flag
        }

        return RunnableParallel(**base_fields, **dynamic_fields)

    def _load_model(self, model_id):
        template_config = self.template_manager.get_template_config(self.model_id)
        self.model_name = template_config.name
        model_path = self._model_path()
        need_save = False

        if os.path.exists(model_path):
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", low_cpu_mem_usage=True, use_cache=True, cache_dir=self.cache_dir, local_files_only=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, cache_dir=self.cache_dir)
        else:
            model, tokenizer = self._download_model(model_id)
            need_save = True

        special_tokens = {}
        if template_config.bos_token is not None:
            special_tokens["bos_token"] = template_config.bos_token
        if template_config.eos_token is not None:
            special_tokens["eos_token"] = template_config.eos_token
        if template_config.pad_token is not None:
            special_tokens["pad_token"] = template_config.pad_token
        tokenizer.add_special_tokens(special_tokens)

        if need_save:
            self._save_model(model, tokenizer, model_path)

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer
    
    def _quant_config(self):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    
    def _generate_cot_reasoning(self, input):
        return f"""
                Step 1: analyze the instruction\n
                Step 2: contextual relative domain knowledge\n
                Step 3: reasoning\n
                Step 4: answer\n
                """

    def _download_model(self, model_id):
        model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", low_cpu_mem_usage=True, quantization_config=self._quant_config(), use_cache=True, cache_dir=self.cache_dir,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False, cache_dir=self.cache_dir)
        return model, tokenizer
    
    def _save_model(self, model, tokenizer, model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

    def _model_path(self):
        return f"{self.save_dir}/{self.model_name}"
    
    def talk(self, instruction):
        inputs = {"instruction": instruction}
        spinner = Halo(text='Processing...', spinner='dots')
        spinner.start()
        with torch.inference_mode():
            response = self.chain.invoke(inputs)
        spinner.succeed("Done!")
        if self.use_history:
            self.memory.save_context(inputs, {"history": response})
        return response

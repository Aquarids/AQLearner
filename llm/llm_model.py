from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables import RunnableParallel, RunnableLambda

from .template_manager import TemplateManager
from .rag.wiki_rag import WikiRAG
from .rag.contriever_rag import ContrieverRAG

import os
import torch
from halo import Halo

class LLMModel:

    def __init__(self, model_id, dir, use_history=False, use_retrieval=False, use_cot=False):
        self.model_id = model_id
        self.save_dir = dir
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

    def init(self, rag=None):
        if self.use_retrieval:
            self.rag = rag

        if self.use_history:
            self.memory = self._init_memory()

        model, tokenizer = self._load_model(self.model_id)

        text_generation_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=4096,
            truncation=True,
            return_full_text=False,
            temperature=0.9,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
        )
        llm = HuggingFacePipeline(pipeline=text_generation_pipe)

        template = self.template_manager.get_template(self.model_id, history=self.use_history, context=self.use_retrieval, cot=self.use_cot)
        prompt = self._prompt_template(template)

        self.llm = llm
        self.template = prompt

        input = self._build_dynamic_input(self.use_retrieval, self.use_history, self.use_cot)
        debug_input = RunnableLambda(lambda x: print("DEBUG INPUT:", x) or x)

        self.chain = (
            input
            | prompt
            | debug_input
            | llm
        )

    def _prompt_template(self, template):
        return PromptTemplate.from_template(template=template)


    def _init_memory(self):
        memory = ConversationBufferWindowMemory(
            k=1,
            memory_key="history",
            input_key="instruction"
        )
        return memory
    
    def _init_rag(self):
        # rag = WikiRAG(os.path.join(self.save_dir, "wiki_rag_cache"))
        rag = ContrieverRAG(os.path.join(self.save_dir, "contriever_rag_cache"))
        rag.load_index()
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
                model_path, device_map="auto", low_cpu_mem_usage=True, use_cache=True, local_files_only=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto", trust_remote_code=True, use_fast=False)
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
                model_id, device_map="auto", low_cpu_mem_usage=True, quantization_config=self._quant_config(), use_cache=True
            )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
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

    def tokenize(self, text, padding=True, truncation=True):
        return self.tokenizer(text, padding=padding, truncation=truncation, return_tensors="pt").to(self.device)

    def generate(self, input_ids, max_length=50):
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def forward(self, inputs_ids, labels_ids):
        with torch.no_grad():
            outputs = self.model(inputs_ids, labels=labels_ids)
        return outputs
    
    def compute_gradients(self, inputs_ids, labels_ids):
        with torch.enable_grad():
            outputs = self.model(inputs_ids, labels=labels_ids)
            loss = outputs.loss
            loss.backward()
            grad = self.model.get_input_embeddings().weight.grad
        return grad
    
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def get_input_embeddings_weight(self):
        return self.model.get_input_embeddings().weight.detach()
    
    def decode(self, input_ids):
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)
    
    def batch_decode(self, input_ids):
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    def get_vocab(self):
        return self.tokenizer.get_vocab()
    
    def embedding(self, input_ids):
        with torch.no_grad():
            outputs = self.model.get_input_embeddings()(input_ids)
        return outputs
    

from dataclasses import dataclass
from typing import Dict, Any
from transformers import AutoTokenizer

MODEL_ID_DEEPSEEK_7B = "deepseek-ai/deepseek-llm-7b-chat"
MODEL_ID_LLAMA_2 = "meta-llama/Llama-2"
MODEL_ID_QWEN_7B = "qwen/qwen-7b-chat"

@dataclass
class ModelTemplateConfig:
    id: str
    name: str
    template_system: str
    template_user: str
    template_history: str
    template_context: str
    template_cot: str
    bos_token: str
    eos_token: str
    pad_token: str

class TemplateManager:
    def __init__(self):
        self._template_registry = {
            # DeepSeek-7B
            MODEL_ID_DEEPSEEK_7B: ModelTemplateConfig(
                    id=MODEL_ID_DEEPSEEK_7B,
                    name="DeepSeek-7B",
                    template_system="System: You are an AI assistant.",
                    template_cot="Show the reasoning steps before giving the final answer: \n{cot_reasoning}\n",
                    template_user="Current Instruction: {instruction}",
                    template_history="Historical Dialogue:\n{history}\n",
                    template_context="Context:\n{context}\n",
                    bos_token="<|begin▁of▁sentence|>",
                    eos_token="<|end▁of▁sentence|>",
                    pad_token="[PAD]"
                ),

                # LLaMA-2
                MODEL_ID_LLAMA_2: ModelTemplateConfig(
                    id=MODEL_ID_LLAMA_2,
                    name="LLaMA-2",
                    template_system="You are a helpful assistant.",
                    template_cot="Show the reasoning steps before giving the final answer: \n{cot_reasoning}\n",
                    template_history="Consider chat history: \n{history}\n",
                    template_context="Consider context:\n{context}\n",
                    template_user="{instruction}",
                    bos_token="[INST]",
                    eos_token="</s>",
                    pad_token="<pad>"
                ),

                # Qwen-7B
                MODEL_ID_QWEN_7B: ModelTemplateConfig(
                    id=MODEL_ID_QWEN_7B,
                    name="Qwen-7B",
                    template_system="system\n",
                    template_cot="Show the reasoning steps before giving the final answer: \n{cot_reasoning}\n",
                    template_history="history: \n{history}\n",
                    template_context="context: \n{context}\n",
                    template_user="user:{instruction}",
                    bos_token="<|im_start|>",
                    eos_token="<|im_end|>",
                    pad_token="<|extra_0|>"
                )
        }

    def _get_default_token(self, tokenizer: AutoTokenizer) -> str:
        bos_token = tokenizer.bos_token
        eos_token = tokenizer.eos_token
        pad_token = tokenizer.pad_token
        return bos_token, eos_token, pad_token
    
    def get_template(self, model_id: str, history=False, context=False, cot=False) -> str:
        config = self._template_registry[model_id]
        if config is None:
            config = self._default_template_config()

        if config.id == MODEL_ID_DEEPSEEK_7B:
            # template=(
            #     "<|begin▁of▁sentence|>System: You are an AI assistant."
            #     "Show the reasoning steps before giving the final answer: \n{cot_reasoning}\n"
            #     "Context:\n{context}\n\n"
            #     "Historical Dialogue:\n{history}\n\n"
            #     "Current Instruction: {instruction}\n"
            #     "<|end▁of▁sentence|>\n"
            #     "Assistant:"
            # ),
            return f"{config.bos_token}\n{config.template_system} {config.template_cot if cot else ''}{config.template_context if context else ''}{config.template_history if history else ''}{config.template_user}\n{config.eos_token}\nAssistant:"
        elif config.id == MODEL_ID_LLAMA_2:
            # template=(
            #     "[INST] <<SYS>>\n"
            #     "You are a helpful assistant. Consider chat history:\n{history}\n Consider context:\n{context}\n Show the reasoning steps before giving the final answer: \n{cot_reasoning}\n \n"
            #     "<</SYS>>\n\n"
            #     "{instruction} [/INST]"
            # ),
            return f"{config.bos_token} <<SYS>>\n{config.template_system} {config.template_history if history else ''} {config.template_context if context else ''} {config.template_cot if cot else ''} \n<</SYS>>\n\n{config.template_user} {config.bos_token}"
        elif config.id == MODEL_ID_QWEN_7B:
            # template=(
            #     "<|im_start|>system\n"
            #     "history: \n{history}\n"
            #     "context: {context}<|im_end|>\n"
            #     "Show the reasoning steps before giving the final answer: \n{cot_reasoning}\n"
            #     "<|im_start|>user\n"
            #     "{instruction}<|im_end|>\n"
            #     "<|im_start|>assistant\n"
            # ),
            return f"{config.bos_token}{config.template_system}{config.template_history if history else ''}{config.template_context if context else ''}{config.template_cot if cot else ''}\n{config.eos_token}\n{config.bos_token}{config.template_user}\n{config.eos_token}\n{config.bos_token}assistant:"
        else:
            # default template
            return f"{config.template_system}{config.template_context if context else ''}{config.template_history if history else ''}{config.template_cot if cot else ''}{config.template_user}"

    def _default_template_config(self) -> str:
        return ModelTemplateConfig(
            name="Default",
            template_system="System: \n",
            template_history="History: {history}\n",
            template_context="Context: {context}\n",
            template_cot="Show the reasoning steps before giving the final answer.\n",
            template_user="User: {instruction}\n",
        )

    def get_template_config(self, model_id: str) -> ModelTemplateConfig:
        for key in self._template_registry:
            if model_id == key:
                return self._template_registry[key]
        
        return self._default_template_config()

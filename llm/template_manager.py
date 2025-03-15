from dataclasses import dataclass
from typing import Dict, Any
from transformers import AutoTokenizer

MODEL_ID_DEEPSEEK_7B = "deepseek-ai/deepseek-llm-7b-chat"
MODEL_ID_LLAMA_2 = "meta-llama/Llama-2"

@dataclass
class ModelTemplateConfig:
    name: str
    template: str
    bos_token: str
    eos_token: str
    pad_token: str

class TemplateManager:
    def __init__(self):
        self._template_registry = {
            # DeepSeek-7B
            MODEL_ID_DEEPSEEK_7B: ModelTemplateConfig(
                    name="DeepSeek-7B",
                    template=(
                        "<|begin▁of▁sentence|>System: You are an AI assistant.\n\n"
                        "Historical Dialogue:\n{history}\n\n"
                        "Current Instruction: {instruction}\n"
                        "<|end▁of▁sentence|>\n"
                        "Assistant:"
                    ),
                    bos_token="<|begin▁of▁sentence|>",
                    eos_token="<|end▁of▁sentence|>",
                    pad_token="[PAD]"
                ),

                # LLaMA-2
                MODEL_ID_LLAMA_2: ModelTemplateConfig(
                    name="LLaMA-2",
                    template=(
                        "[INST] <<SYS>>\n"
                        "You are a helpful assistant. Consider chat history:\n{history}\n"
                        "<</SYS>>\n\n"
                        "{instruction} [/INST]"
                    ),
                    bos_token="[INST]",
                    eos_token="</s>",
                    pad_token="<pad>"
                ),

                # GPT-3.5 Turbo
                "openai-community/gpt-3.5-turbo": ModelTemplateConfig(
                    name="GPT-3.5 Turbo",
                    template=(
                        "<|im_start|>system\n"
                        "history: {history}<|im_end|>\n"
                        "<|im_start|>user\n"
                        "{instruction}<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    ),
                    bos_token="<|im_start|>",
                    eos_token="<|im_end|>",
                    pad_token="<|endoftext|>"
                ),

                # Qwen-7B
                "Qwen/Qwen-7B-Chat": ModelTemplateConfig(
                    name="Qwen-7B",
                    template=(
                        "<|im_start|>system\n"
                        "history: \n{history}<|im_end|>\n"
                        "<|im_start|>user\n"
                        "{instruction}<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    ),
                    bos_token="<|im_start|>",
                    eos_token="<|im_end|>",
                    pad_token="<|extra_0|>"
                )
        }

    def _get_default_token(self, tokenizer: AutoTokenizer) -> str:
        bos_token = tokenizer.bos_token
        eos_token = tokenizer.eos_token
        pad_token = tokenizer.pad_token or tokenizer.eos_token
        return bos_token, eos_token, pad_token

    def get_template_config(self, model_id: str, tokenizer: AutoTokenizer) -> ModelTemplateConfig:
        for key in self._template_registry:
            if model_id == key:
                return self._template_registry[key]
        
        bos_token, eos_token, pad_token = self._get_default_token(tokenizer)
        return ModelTemplateConfig(
            template="{instruction}",
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token
        )

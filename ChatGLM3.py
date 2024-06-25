import ast
import json
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import List, Optional


class ChatGLM3(LLM):
    max_token: int = 16000
    do_sample: bool = True
    temperature: float = 0.8
    top_p = 0.8
    tokenizer: object = None
    model: object = None
    history: List = []
    has_search: bool = False

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM3"

    def load_model(self, model_name_or_path=None):
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            # model_name_or_path, config=model_config, trust_remote_code=True, device_map="auto").eval()
            model_name_or_path, config=model_config, trust_remote_code=True).cuda()

    
    def qa(self,prompt):
        # print(f'传入的prompt为：{prompt}')
        response = self.model.chat(
        self.tokenizer,
        query = prompt,
        do_sample=self.do_sample,
        max_length=self.max_token,
        temperature=self.temperature
        )
        return response

    def count_param(self):
        return self.model.parameters()

    def _call(self, prompt: str):
        pass
from typing import List, Dict, Union
from langchain_community.embeddings import OllamaEmbeddings
import requests
import json
import aiohttp


class LLM():
    """ Класс-обертка для запуска локальных моделей """
    # Модели: llama3.2:3b или llama3.2:1b и mxbai-embed-large
    # todo добавить бд и индексацию (возможно)
    # Добавлять изображения можно добавив в json поле "images": List[Base64]
    
    def __init__(self, host: str, port: Union[int, str], model: str, stream: bool=False) -> None:
        self.model = model
        self.host = host
        self.port = port
        self.stream = stream
    
    async def generate(self, promt: str, images: List[str], **params):
        """ Ответ одним сообщением (ответ) """
        # Добавляем параметры для ускорения генерации
        generation_params = {
            "model": self.model, 
            "prompt": promt, 
            "stream": self.stream, 
            "images": images,
            "temperature": 0.1,  # Низкая температура = более предсказуемые ответы
            "top_p": 0.9,        # Ограничивает выбор токенов
            "top_k": 40,         # Ограничивает выбор токенов
            "max_tokens": 150,   # Ограничивает длину ответа
            "repeat_penalty": 1.1  # Предотвращает повторения
        }
        
        # Обновляем параметрами из kwargs
        generation_params.update(params)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"http://{self.host}:{self.port}/api/generate", json=generation_params) as response:
                data = await response.json()
                return data["response"]
    
    async def chat(self, messages: List[Dict], images: List[str], **params):
        """ Ответ на основе множества сообщений (чат) 
        - messages: пример сообщения [{"role": "user", "content": "question"}]
        """
        json_data={"model": self.model, "messages": messages, "stream": self.stream, "images": images, **params}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"http://{self.host}:{self.port}/api/chat", json=json_data) as response:
                data = await response.json()
                return data["message"]["content"]
    
    def get_embeding_model(self, model: str="nomic-embed-text"):
        """ Ембеддинги текстов """
        return OllamaEmbeddings(model=model, base_url=f"http://{self.host}:{self.port}")
    
    def send(self, json, path):
        """ Метод отправки данных в модель """
        return requests.post(f"http://{self.host}:{self.port}/api/{path}", json=json)

    
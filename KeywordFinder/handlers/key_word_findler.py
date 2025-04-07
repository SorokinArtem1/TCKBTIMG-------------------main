import os
import json
import logging
import aiohttp
# Для ембеддинга и модели

from KeywordFinder.handlers.llm import LLM

from aiogram import Bot, Dispatcher, Router, F, types
from aiogram.types import Message, FSInputFile
from aiogram.client.bot import DefaultBotProperties
from aiogram.fsm.state import StatesGroup, State, default_state
from aiogram.enums import ParseMode
from keyboards import reply
from aiogram.fsm.context import FSMContext
from KeywordFinder.utils.states import FSMAdmin

import pandas as pd # если нет импорта пандаса, то остальной код не имеет смысла
from pandas import DataFrame

from langchain_community.document_loaders import DataFrameLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

import pytesseract

from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Убедитесь, что путь корректный

# Настройка логирования
logging.basicConfig(level=logging.INFO)

from config import bot  # Импортируйте bot отсюда


img1 = FSInputFile(path=r'D:\Downloads\Screenshot_52.jpg')


knowledge_base_router = Router()

documents1 = [
    {"id": 1,
        "question": "Как восстановить пароль?",
        "answer": "Для восстановления пароля перейдите по ссылке 'Забыли пароль?' на странице входа. Введите свой адрес электронной почты, и мы вышлем вам инструкции по восстановлению пароля. Если вы не получили письмо с инструкциями, проверьте папку со спамом или повторите запрос через несколько минут. Если у вас по-прежнему возникают проблемы, свяжитесь со службой поддержки по адресу support@company.com.",
        "url": "https://example.com/confluence/recover-password",
        "image_irl": None}, 
    {"id": 2,
        "question": "Как настроить двухфакторную аутентификацию?",
        "answer": "Для настройки двухфакторной аутентификации перейдите в раздел 'Настройки безопасности' вашего аккаунта и следуйте инструкциям. Выберите метод двухфакторной аутентификации, который вам удобен, например, SMS или приложение для аутентификации. Введите код подтверждения, который вы получите на ваш телефон или в приложении. После успешной настройки двухфакторной аутентификации вам будет необходимо вводить код подтверждения каждый раз при входе в систему. Это значительно повышает безопасность вашего аккаунта.",
        "url": "https://example.com/confluence/2fa-setup",
        "image_irl": None},
    {"id": 3,
        "question": "Как связаться с поддержкой?",
        "answer": "Контактные данные службы поддержки:\nТелефон: +7 (495) 123-45-67\nЭлектронная почта: support@company.com\nФорма обратной связи на сайте: https://company.com/support\n\nВремя работы службы поддержки:\nНаша служба поддержки работает круглосуточно, без выходных. Вы можете обратиться к нам в любое удобное для вас время.\n\nКак оставить заявку на техническую помощь:\nЧтобы оставить заявку на техническую помощь, выполните следующие действия:\n1. Перейдите на страницу технической поддержки на нашем сайте: https://company.com/support.\n2. Заполните форму заявки, указав необходимую информацию:\n   - Ваше имя и контактные данные.\n   - Описание проблемы.\n   - Желаемое время для связи.\n3. Нажмите кнопку 'Отправить'.\nМы постараемся решить вашу проблему как можно быстрее.",
        "url": "https://company.com/support",
        "image_irl": None},
    {"id": 4,
        "question": "Как использовать корпоративный мессенджер?",
        "answer": "Основные функции мессенджера:\nНаш корпоративный мессенджер предоставляет следующие основные функции:\n- Обмен текстовыми сообщениями в режиме реального времени.\n- Создание групповых чатов для общения с коллегами.\n- Обмен файлами и документами.\n- Видеозвонки и аудиозвонки.\n- Интеграция с другими корпоративными сервисами.\n\nКак добавить контакт:\nЧтобы добавить контакт в корпоративный мессенджер, выполните следующие действия:\n1. Откройте приложение мессенджера.\n2. Нажмите на кнопку 'Контакты'.\n3. Нажмите на кнопку 'Добавить контакт'.\n4. Введите имя пользователя или адрес электронной почты контакта.\n5. Нажмите на кнопку 'Добавить'.\n\nКак создать групповой чат:\nЧтобы создать групповой чат, выполните следующие действия:\n1. Откройте приложение мессенджера.\n2. Нажмите на кнопку 'Создать чат'.\n3. Выберите 'Групповой чат'.\n4. Добавьте участников чата, выбрав их из списка контактов.\n5. Придумайте название чата.\n6. Нажмите на кнопку 'Создать'.",
        "url": "https://company.com/messenger",
        "image_irl": None}, 
    {"id": 5,
        "question": "Как оформить командировку?",
        "answer": "Пошаговая инструкция по оформлению командировки:\nЧтобы оформить командировку, выполните следующие действия:\n1. Получите согласование командировки у вашего руководителя.\n2. Заполните заявление на командировку.\n3. Предоставьте необходимые документы в отдел кадров.\n4. Получите командировочное удостоверение и другие необходимые документы.\n5. Оформите билеты и бронирование проживания.\n6. Предоставьте отчет о командировке по возвращении.\n\nНеобходимые документы для командировки:\nДля оформления командировки вам потребуются следующие документы:\n- Заявление на командировку.\n- Командировочное удостоверение.\n- Билеты на транспорт.\n- Документы, подтверждающие бронирование проживания.\n- Другие документы, предусмотренные внутренними правилами компании.\n\nКонтактные данные отдела кадров:\nДля связи с отделом кадров вы можете использовать следующие контактные данные:\nТелефон: +7 (495) 123-45-68\nЭлектронная почта: hr@company.com\nАдрес: Москва, ул. Ленина, д. 1, офис 101",
        "url": "https://company.com/hr",
        "image_irl": None}
]

all_documents = documents1
# todo создать интерфейс для загрузки документов (парсер например)

class VectorStore():
    """ Векторная база данных """
    
    def __init__(self, embedding_model, path=None, name="VectorStore") -> None:
        self.embedding_model = embedding_model
        self.is_loaded = False
        self.name = name
        self.store = None 
        self.load_storage(path)

    def load_storage(self, path):
        if path is not None and os.path.exists(path):
            self.store = FAISS.load_local(self.name, self.embedding_model)
            self.is_loaded = True
        else:
            # Создаем пустое хранилище только при наличии документов
            self.store = None  # Инициализируем как None



    def add(self, docs: pd.DataFrame):
        # Создаем документы из ответов, разбивая их на чанки
        all_documents = []
        
        for _, row in docs.iterrows():
            # Разбиваем ответ на предложения
            sentences = row['answer'].split('.')
            
            # Группируем предложения в чанки по 3-5 предложений
            chunks = []
            current_chunk = []
            
            for sentence in sentences:
                if sentence.strip():
                    current_chunk.append(sentence)
                    if len(current_chunk) >= 3:
                        chunks.append('. '.join(current_chunk) + '.')
                        current_chunk = []
            
            # Добавляем последний чанк, если он не пустой
            if current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
            
            # Создаем документы для каждого чанка, сохраняя ID
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=f"{row['question']} {chunk}",
                    metadata={"id": row['id'], "chunk": i}
                )
                all_documents.append(doc)
        
        # Создаем хранилище
        if not self.is_loaded:
            self.store = FAISS.from_documents(all_documents, self.embedding_model)
        else:
            self.store.add_documents(all_documents)
        
        # Обновляем
        self.store.save_local(self.name)


# Далее ваш код


df = pd.DataFrame(all_documents)


model = LLM(model="llama3.2-vision", host="127.0.0.1", port=11434)
db = VectorStore(embedding_model=model.get_embeding_model())

# Добавляем документы в хранилище
db.add(df)


@knowledge_base_router.message(F.text.lower() == "ответ по базе знаний")
async def menu_cmd(message: types.Message, state:FSMAdmin):
    await message.answer("Задайте мне вопрос, и я найду релевантные документы.", reply_markup=reply.start_kb)
    await state.set_state(FSMAdmin.input)

@knowledge_base_router.message(FSMAdmin.input)
async def process_message(message: types.Message, state: FSMContext):
    if message.content_type == 'photo':
        # Если сообщение содержит изображение, получаем фото с наибольшим разрешением
        photo = message.photo[-1]
        
        # Получаем объект файла
        file = await bot.get_file(photo.file_id)
        
        # Определяем путь для сохранения фото
        photo_path = f"{file.file_id}.jpg"
        
        # Скачиваем файл в локальную систему
        await bot.download_file(file.file_path, destination=photo_path)
        
        # Открываем изображение и извлекаем текст
        img = Image.open(photo_path)
        text = pytesseract.image_to_string(img, lang='rus')
        

        if not text.strip():  # Проверка на пустой текст
            await message.reply("На фото плохо видно текст")
            return 
        

        query = text
        await message.reply(query)
    else:
        query = message.text

    # Поиск релевантного документа
    print(f"\n=== Новый запрос ===")
    print(f"Вопрос пользователя: {query}")
    
    # Получаем несколько результатов вместо одного (k=5 или больше, в зависимости от количества документов)
    results = db.store.similarity_search_with_score(query, k=len(all_documents))
    print(f"Результаты поиска: {results}")
    
    # Выводим оценки схожести для всех найденных документов
    print("\n=== Оценки схожести для всех документов ===")
    for i, (doc, score) in enumerate(results):
        doc_id = doc.metadata['id']
        doc_question = next((d['question'] for d in all_documents if d['id'] == doc_id), "Неизвестный вопрос")
        print(f"ID: {doc_id}, Оценка: {score}, Вопрос: {doc_question}")
    
    # Выбираем документ с наилучшей оценкой (наименьшим расстоянием)
    if results:
        # Сортируем по оценке (меньше = лучше)
        sorted_results = sorted(results, key=lambda x: x[1])
        best_match = sorted_results[0]
        document_id = best_match[0].metadata['id']
        similarity_score = best_match[1]
        
        print(f"\n=== Выбран документ ===")
        print(f"Найден документ ID: {document_id}")
        print(f"Оценка схожести: {similarity_score}")
        
        full_answer = next((doc['answer'] for doc in all_documents if doc['id'] == document_id), None)
        image_path = next((doc['image_irl'] for doc in all_documents if doc['id'] == document_id), None)
        if full_answer is not None:
            # Формируем промпт для Ollama
            prompt = f"""Система: Ты - русскоязычный ассистент. Отвечай кратко, только на основе контекста.

Контекст: {full_answer}

Вопрос: {query}

Инструкции:
1. Используй ТОЛЬКО информацию из контекста
2. Ответ должен быть 1-5 предложения
3. Если в контексте нет ответа, скажи "Извините, в контексте нет ответа на этот вопрос"
4. Отвечай ТОЛЬКО на вопрос пользователя
5. Ответы должны быть вежливыми, структурированными.
6. Избегать излишне разговорных или неформальных выражений.
7. В конце каждого ответа должна быть добавлена фраза: "Благодарим за обращение! Если у вас остались дополнительные вопросы, мы готовы на них ответить!"
8. В начале добавить "Спасибо за ваше обращение!"
"""
            
            # Получаем ответ от Ollama
            try:
                response = await model.generate(prompt, images=[])
            except KeyError:
                # Прямой запрос к API
                async with aiohttp.ClientSession() as session:
                    json_data = {"model": model.model, "prompt": prompt, "stream": model.stream, "images": []}
                    async with session.post(f"http://{model.host}:{model.port}/api/generate", json=json_data) as resp:
                        data = await resp.json()
                        # Для llama3.3:70b часто используется ключ "completion"
                        response = data.get("completion", str(data))
            print(f"Ответ модели: {response}")
            await message.answer(response)
            # Проверяем наличие изображения и анализируем его
            
            if image_path:
                for image_row in image_path:

                    img = Image.open(image_row)
                    extracted_text = pytesseract.image_to_string(img, lang='rus')
        
                    text_analysis_prompt = f"""
Твоя задача проанализировать текст с изображения и дать ответ 'да', если текст изображения отвечает на вопрос пользователя и 'нет' если текст изображения не отвечает на вопрос пользователя.

Текст с изображение: {extracted_text} 

Вопрос пользователя: {query}

Инстррукции:
1. Отвечай "да" только если текст дословно отвечает на вопрос пользователя
2. Наличие общей темы текста изображения и вопроса пользователя не является причиной говорить "да"
3. В случае если текст изображения не отвечает точно на вопрос пользователя, но и вопрос и текст изображения имеют общую тему - Отвечай "нет"
4. Ты можешь в своем ответе писать только 'да' или 'нет' в зависимости от анализа
"""
                    extracted_text_meaning = await model.generate(text_analysis_prompt, images=[])
                    
                    print (f"Вывод: {extracted_text_meaning}")

                    if "да" in extracted_text_meaning.lower():
                        img_file = FSInputFile(path=image_row)
                        await message.answer_photo(photo=img_file)

                    else:
                        await message.answer("Извините, изображение не содержит ответа на ваш вопрос.")

    print ("The End")


                # # Извлекаем текст из изображения
                # img = Image.open(image_path)
                # extracted_text = pytesseract.image_to_string(img, lang='rus')

                # text_analysis_prompt = f"Проанализируй этот текст и опиши его смысл: {extracted_text}"
                # text_analysis_response = await model.generate(text_analysis_prompt, images=[])
                # print (f"Смысл текста: {text_analysis_response}")

                # # Проверяем, содержит ли извлеченный текст ответ на вопрос
                # if query.lower() in text_analysis_response.lower():  # Сравниваем текст вопроса с извлеченным текстом
                #     img1 = FSInputFile(path=image_path)
                #     await message.answer_photo(photo=img1)  # Отправляем изображение





#    else:
#      await message.answer("Извините, я не смог найти ответ на ваш вопрос.")
#             if image_path:
#                 # Извлекаем текст из изображения
#                 img = Image.open(image_path)
#                 # Формируем запрос для анализа изображения
#                 image_analysis_prompt = f"Проанализируй это изображение и опиши его смысл: {img}"
#                 image_analysis_response = await model.generate(image_analysis_prompt, images=img)

#                 # Проверяем, соответствует ли смысл изображения запросу пользователя
#                 if query.lower() in image_analysis_response.lower():   # Сравниваем смысл изображения с запросом
#                     img1 = FSInputFile(path=image_path)
#                     await message.answer_photo(photo=img1)  # Отправляем изображение
#                 else:
#                     await message.answer("Извините, изображение не содержит ответа на ваш вопрос.")
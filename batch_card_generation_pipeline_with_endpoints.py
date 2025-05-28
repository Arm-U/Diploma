import openai
from openai import AzureOpenAI
import os
from dotenv import load_dotenv, find_dotenv
import json
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Any, TypedDict
import time
from requests.exceptions import ConnectionError, Timeout
import logging
import datetime
import traceback
from dataclasses import dataclass
from enum import Enum

# Настройка логирования
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, "batch_card_generation.log")

# Настройка форматирования логов
log_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Настройка обработчика файла
file_handler = logging.FileHandler(log_filename, mode='a')
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

# Настройка логгера
logger = logging.getLogger("batch_card_generation")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Отключаем логи от других модулей
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Отключаем вывод логов в консоль
logging.getLogger().handlers = []

GPT_3_5_TURBO = "gpt-3.5-turbo"
GPT_4_TURBO_PREVIEW = "gpt-4-turbo-preview"
GPT_4 = 'gpt-4'
GPT_4o = 'gpt-4o'

class TranslationQuality(Enum):
    PERFECT = 1.0
    IMPERFECT = 0.0

@dataclass
class Card:
    word: str
    sentence: str
    id: str

class EvaluationResult(TypedDict):
    sentenceCorrectness: Dict[str, Any]
    wordUsage: Dict[str, Any]
    wordTranslationAccuracy: Dict[str, Any]
    sentenceTranslationAccuracy: Dict[str, Any]
    id: str

@dataclass
class CardProcessingState:
    source_card: Dict
    secondary_card: Dict
    generated_card: Optional[Dict] = None
    evaluation_result: Optional[Dict] = None
    is_successful: bool = False
    improvement_attempts: int = 0

class CardQueue:
    def __init__(self):
        self.successful_cards: List[CardProcessingState] = []
        self.pending_cards: List[CardProcessingState] = []
        self.failed_cards: List[CardProcessingState] = []
    
    def add_to_pending(self, card_state: CardProcessingState):
        self.pending_cards.append(card_state)
    
    def mark_as_successful(self, card_state: CardProcessingState):
        card_state.is_successful = True
        self.successful_cards.append(card_state)
        if card_state in self.pending_cards:
            self.pending_cards.remove(card_state)
    
    def mark_as_failed(self, card_state: CardProcessingState):
        if card_state in self.pending_cards:
            self.pending_cards.remove(card_state)
        self.failed_cards.append(card_state)
    
    def get_batch(self, batch_size: int) -> List[CardProcessingState]:
        return self.pending_cards[:batch_size]
    
    def is_empty(self) -> bool:
        return len(self.pending_cards) == 0
    
    def get_stats(self) -> Dict:
        return {
            "successful": len(self.successful_cards),
            "pending": len(self.pending_cards),
            "failed": len(self.failed_cards)
        }

def get_openai_api_key():
    _ = load_dotenv(find_dotenv())
    return os.getenv("OPENAI_API_KEY")

def get_azure_endpoints():
    _ = load_dotenv(find_dotenv())
    endpoints = {}
    for i in range(1, 6):  # Предполагаем, что у нас есть 5 endpoint'ов
        endpoint = os.getenv(f"AZURE_OPENAI_ENDPOINT_{i}")
        if endpoint:
            # Извлекаем имя модели из имени эндпоинта
            endpoint_name = endpoint.split('/')[-1]  # Получаем последнюю часть URL
            endpoints[f"endpoint_{i}"] = {
                "url": endpoint,
                "model": endpoint_name  # Используем имя эндпоинта как имя модели
            }
    return endpoints

def create_azure_client(endpoint: str):
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-07-01-preview",
        azure_endpoint=endpoint
    )

def generate_batch_unified_cards(
    source_cards: List[Dict],
    secondary_cards: List[Dict],
    source_lang: str,
    target_lang: str,
    secondary_lang: str,
    model: str = GPT_4o,
    system_prompt: Optional[str] = None,
    endpoint: Optional[str] = None
) -> List[Dict]:
    """
    Generates multiple cards in a single API call.
    
    Args:
        source_cards: List of cards in source language
        secondary_cards: List of cards in secondary language
        source_lang: Source language
        target_lang: Target language
        secondary_lang: Secondary language
        model: GPT model to use
        system_prompt: Optional custom system prompt
        endpoint: Optional Azure OpenAI endpoint
        
    Returns:
        List[Dict]: List of generated cards
    """
    if system_prompt is None:
        system_prompt = f'''You are a multilingual assistant who is proficient in {source_lang}, {secondary_lang} and {target_lang}.'''

    # Create client with specified endpoint
    client = create_azure_client(endpoint) if endpoint else create_azure_client(os.getenv("AZURE_OPENAI_ENDPOINT"))

    # Prepare the batch of cards for the prompt
    cards_batch = []
    for source_card, secondary_card in zip(source_cards, secondary_cards):
        source_word, source_sentence, _ = source_card.items()
        secondary_word, secondary_sentence, _ = secondary_card.items()
        
        cards_batch.append({
            "source_word": source_word,
            "source_sentence": source_sentence,
            "secondary_word": secondary_word,
            "secondary_sentence": secondary_sentence,
            "id": source_card['id']
        })

    user_prompt = f"""
    **Translate the given batch of {source_lang} words or phrases along with their {secondary_lang} translations into {target_lang}, 
    and then translate the provided {source_lang} sentences, incorporating the {target_lang} translations of the words or phrases. 
    Use synonyms or related terms where necessary to convey the intended meaning and maintain naturalness in {target_lang}.**  

    ### Cards to translate:
    {json.dumps(cards_batch, ensure_ascii=False, indent=2)}

    ### Important notes:
    - If the input sentence is empty (''), the output sentence should also be empty ('')
    - If the input sentence contains content, translate it naturally while incorporating the translated word/phrase
    - Maintain consistency in translation style across all cards

    ### Response structure:  

    Respond in JSON format with the following structure:
    {{
        "translations": [
            {{
                "id": "card_id",
                "translatedWord": "Translated word in {target_lang}",
                "translatedSentence": "Translated sentence in {target_lang} (empty string if input sentence was empty)"
            }},
            ...
        ]
    }}
    """

    response = client.chat.completions.create(
        model=model,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ]
    )

    response_dict = json.loads(response.choices[0].message.content.strip())
    translations = response_dict['translations']

    generated_cards = []
    for translation in translations:
        card = {
            "word": translation['translatedWord'].strip() if translation['translatedWord'] is not None else "",
            "sentence": translation['translatedSentence'].strip() if translation['translatedSentence'] is not None else "",
            "id": translation['id']
        }
        generated_cards.append(card)

    return generated_cards

def evaluate_batch_unified_cards(
    source_cards: List[Dict],
    secondary_cards: List[Dict],
    target_cards: List[Dict],
    source_lang: str,
    target_lang: str,
    secondary_lang: str,
    model: str = GPT_4o,
    endpoint: Optional[str] = None
) -> List[Dict]:
    """
    Evaluates multiple cards in a single API call.
    
    Args:
        source_cards: List of cards in source language
        secondary_cards: List of cards in secondary language
        target_cards: List of cards in target language
        source_lang: Source language
        target_lang: Target language
        secondary_lang: Secondary language
        model: GPT model to use
        endpoint: Optional Azure OpenAI endpoint
        
    Returns:
        List[Dict]: List of evaluation results
    """
    system_prompt = f'''You are a multilingual assistant who is proficient in {source_lang}, {secondary_lang} and {target_lang}.'''

    # Create client with specified endpoint
    client = create_azure_client(endpoint) if endpoint else create_azure_client(os.getenv("AZURE_OPENAI_ENDPOINT"))

    # Prepare the batch of cards for evaluation
    cards_batch = []
    for source_card, secondary_card, target_card in zip(source_cards, secondary_cards, target_cards):
        source_word, source_sentence, _ = source_card.items()
        secondary_word, secondary_sentence, _ = secondary_card.items()
        target_word, target_sentence, _ = target_card.items()
        
        cards_batch.append({
            "id": source_card['id'],
            "source_word": source_word,
            "source_sentence": source_sentence,
            "secondary_word": secondary_word,
            "secondary_sentence": secondary_sentence,
            "target_word": target_word,
            "target_sentence": target_sentence
        })

    user_prompt = f"""
    **Evaluate the correctness of a batch of {target_lang} words and sentences based on their translations from {source_lang} and {secondary_lang}. 
    You will receive words and sentences in {source_lang}, {secondary_lang}, and their translations in {target_lang}. 
    Your task is to assess the quality of each {target_lang} sentence, the usage of each {target_lang} word in its sentence, 
    and the accuracy of the translations from {source_lang} and {secondary_lang} to {target_lang}.**  

    ### Cards to evaluate:
    {json.dumps(cards_batch, ensure_ascii=False, indent=2)}

    ### Important notes:
    - If the input sentence is empty (''), consider the sentence correctness, word usage in sentence, and sentence translation accuracy as correct (true)
    - If the input sentence contains content, evaluate it as usual
    - Provide detailed explanations and suggested fixes for each card

    ### Response structure:  

    Respond in JSON format with the following structure:
    {{
        "evaluations": [
            {{
                "id": "card_id",
                "sentenceCorrectness": {{
                    "isCorrect": true/false,
                    "explanation": "Detailed explanation if there is an issue or why it's correct.",
                    "suggestedFix": "Suggested corrected sentence if there is an issue, or null if not applicable."
                }},
                "wordUsage": {{
                    "isCorrect": true/false,
                    "explanation": "Detailed explanation if there is an issue or why it's correct.",
                    "suggestedFixSentence": "Suggested corrected sentence if the word usage is incorrect, or null if not applicable.",
                    "suggestedFixWord": "Suggested corrected word if the word usage is incorrect, or null if not applicable."
                }},
                "wordTranslationAccuracy": {{
                    "isCorrect": true/false,
                    "explanation": "Detailed explanation if there is an issue or why it's correct.",
                    "suggestedFix": "Suggested correction for translation issues, or null if not applicable."
                }},
                "sentenceTranslationAccuracy": {{
                    "isCorrect": true/false,
                    "explanation": "Detailed explanation if there is an issue or why it's correct.",
                    "suggestedFix": "Suggested correction for translation issues, or null if not applicable."
                }}
            }},
            ...
        ]
    }}
    """

    response = client.chat.completions.create(
        model=model,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ]
    )

    response_dict = json.loads(response.choices[0].message.content.strip())
    return response_dict['evaluations']

def process_cards_batch(
    source_cards: List[Dict],
    secondary_cards: List[Dict],
    source_lang: str,
    target_lang: str,
    secondary_lang: str,
    model: str = GPT_4o,
    batch_size: int = 3,
    max_improvement_attempts: int = 3,
    max_retries: int = 5,
    retry_delay: int = 5,
    use_moa: bool = False,
    endpoint: Optional[str] = None
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Processes cards with improved error handling and logging.
    First generates translations for all cards, then improves unsuccessful ones.
    Optionally uses Mixture of Agents (MoA) for improvement.
    
    Args:
        source_cards: List of cards in source language
        secondary_cards: List of cards in secondary language
        source_lang: Source language
        target_lang: Target language
        secondary_lang: Secondary language
        model: GPT model to use (ignored when endpoint is specified)
        batch_size: Number of cards to process in each batch
        max_improvement_attempts: Maximum number of improvement attempts per card
        max_retries: Maximum number of retries for each batch
        retry_delay: Delay in seconds between retries
        use_moa: Whether to use Mixture of Agents for improvement
        endpoint: Optional Azure OpenAI endpoint name (e.g. "endpoint_1")
        
    Returns:
        Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]: Tuple containing:
            - List of successfully generated cards
            - List of evaluation results for successful cards
            - List of failed cards
            - List of evaluation results for failed cards
    """
    logger.info(f"Starting card processing with {len(source_cards)} cards")
    logger.info(f"Using MoA: {use_moa}")
    
    # Получаем информацию об эндпоинте и модели
    endpoints = get_azure_endpoints()
    endpoint_info = endpoints.get(endpoint, {"url": os.getenv("AZURE_OPENAI_ENDPOINT"), "model": model})
    endpoint_url = endpoint_info["url"]
    endpoint_model = endpoint_info["model"]
    
    logger.info(f"Using endpoint: {endpoint_url}")
    logger.info(f"Using model: {endpoint_model}")
    
    # Initialize card queue
    card_queue = CardQueue()
    
    # Create initial card states
    for source_card, secondary_card in zip(source_cards, secondary_cards):
        card_state = CardProcessingState(
            source_card=source_card,
            secondary_card=secondary_card
        )
        card_queue.add_to_pending(card_state)
    
    # First pass: generate translations for all cards
    while not card_queue.is_empty():
        batch = card_queue.get_batch(batch_size)
        if not batch:
            break
            
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Generate translations for the batch
                source_batch = [state.source_card for state in batch]
                secondary_batch = [state.secondary_card for state in batch]
                
                generated_cards = generate_batch_unified_cards(
                    source_batch,
                    secondary_batch,
                    source_lang,
                    target_lang,
                    secondary_lang,
                    endpoint_model,  # Используем модель из эндпоинта
                    endpoint=endpoint_url  # Используем URL эндпоинта
                )
                
                # Evaluate translations
                evaluation_results = evaluate_batch_unified_cards(
                    source_batch,
                    secondary_batch,
                    generated_cards,
                    source_lang,
                    target_lang,
                    secondary_lang,
                    endpoint_model,  # Используем модель из эндпоинта
                    endpoint=endpoint_url  # Используем URL эндпоинта
                )
                
                # Update card states
                for card_state, generated_card, eval_result in zip(batch, generated_cards, evaluation_results):
                    card_state.generated_card = generated_card
                    card_state.evaluation_result = eval_result
                    
                    quality = calculate_quality_score(eval_result)
                    if quality == TranslationQuality.PERFECT.value:
                        card_queue.mark_as_successful(card_state)
                    else:
                        card_state.improvement_attempts += 1
                        if card_state.improvement_attempts >= max_improvement_attempts:
                            card_queue.mark_as_failed(card_state)
                
                break  # Success, exit retry loop
                
            except Exception as e:
                retry_count += 1
                error_message = f"Error processing batch (attempt {retry_count}/{max_retries}):\n"
                error_message += f"Error type: {type(e).__name__}\n"
                error_message += f"Error message: {str(e)}\n"
                error_message += f"Traceback:\n{traceback.format_exc()}"
                logger.error(error_message)
                
                if retry_count >= max_retries:
                    for card_state in batch:
                        card_queue.mark_as_failed(card_state)
                    logger.error(f"Failed to process batch after {max_retries} retries")
                else:
                    time.sleep(retry_delay)
    
    # Prepare final results
    successful_cards = [state.generated_card for state in card_queue.successful_cards]
    successful_evals = [state.evaluation_result for state in card_queue.successful_cards]
    failed_cards = [state.generated_card for state in card_queue.failed_cards]
    failed_evals = [state.evaluation_result for state in card_queue.failed_cards]
    
    stats = card_queue.get_stats()
    logger.info(f"Card processing completed. Stats: {stats}")
    
    return successful_cards, successful_evals, failed_cards, failed_evals

def calculate_quality_score(evaluation_result: EvaluationResult) -> float:
    """
    Calculates a quality score from evaluation results.
    
    Args:
        evaluation_result: Evaluation results dictionary
        
    Returns:
        float: Quality score (1.0 if all criteria are correct, 0.0 otherwise)
    """
    criteria = [
        evaluation_result.get('sentenceCorrectness', {}).get('isCorrect', False),
        evaluation_result.get('wordUsage', {}).get('isCorrect', False),
        evaluation_result.get('wordTranslationAccuracy', {}).get('isCorrect', False),
        evaluation_result.get('sentenceTranslationAccuracy', {}).get('isCorrect', False)
    ]
    
    return TranslationQuality.PERFECT.value if all(criteria) else TranslationQuality.IMPERFECT.value

def write_cards_to_tmp_file(successful_cards, successful_evals, failed_cards, failed_evals, output_dir="output"):
    """
    Записывает все карточки (успешные и неудачные) в один файл, отсортированные по id.
    
    Args:
        successful_cards: Список успешно сгенерированных карточек
        successful_evals: Список результатов оценки для успешных карточек
        failed_cards: Список неудачно сгенерированных карточек
        failed_evals: Список результатов оценки для неудачных карточек
        output_dir: Директория для сохранения файла
        
    Returns:
        str: Путь к созданному файлу
    """
    # Создаем директорию, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Формируем имя файла с текущей датой и временем
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"cards_{timestamp}.json")
    
    # Объединяем все карточки и их оценки
    all_cards = []
    for card, eval_result in zip(successful_cards, successful_evals):
        all_cards.append({
            "card": card,
            "evaluation": eval_result,
            "status": "successful"
        })
    
    for card, eval_result in zip(failed_cards, failed_evals):
        all_cards.append({
            "card": card,
            "evaluation": eval_result,
            "status": "failed"
        })
    
    # Сортируем по id
    all_cards.sort(key=lambda x: x["card"]["id"])
    
    # Записываем в файл
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump({
            "cards": all_cards,
            "stats": {
                "total": len(all_cards),
                "successful": len(successful_cards),
                "failed": len(failed_cards)
            }
        }, f, ensure_ascii=False, indent=2)
    
    return file_path 
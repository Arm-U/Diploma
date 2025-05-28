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
GPT_4o_1 = 'gpt-4o.1'

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

OPENAI_API_KEY = get_openai_api_key()
azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_4o-1")
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2025-01-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_4o-1")
)

def create_cards_list(base_cards: List[Dict], source_key: str, secondary_key: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Creates lists of cards for source and secondary languages.
    
    Args:
        base_cards: List of base cards
        source_key: Key for source language
        secondary_key: Key for secondary language
        
    Returns:
        Tuple[List[Dict], List[Dict]]: Tuple of two card lists
    """
    source_cards = []
    secondary_cards = []

    for item in tqdm(base_cards, desc="Creating card lists"):
        source_card = item['translations'][source_key]
        source_card['id'] = item['id']

        secondary_card = item['translations'][secondary_key]
        secondary_card['id'] = item['id']

        source_cards.append(source_card)
        secondary_cards.append(secondary_card)
    
    return source_cards, secondary_cards

def generate_batch_unified_cards(
    source_cards: List[Dict],
    secondary_cards: List[Dict],
    source_lang: str,
    target_lang: str,
    secondary_lang: str,
    model: str = GPT_4o_1,
    system_prompt: Optional[str] = None
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
        
    Returns:
        List[Dict]: List of generated cards
    """
    if system_prompt is None:
        system_prompt = f'''You are a multilingual assistant who is proficient in {source_lang}, {secondary_lang} and {target_lang}.'''

    # Prepare the batch of cards for the prompt
    cards_batch = []
    for source_card, secondary_card in zip(source_cards, secondary_cards):
        # Safely get values from dictionaries
        source_word = source_card.get('word', '')
        source_sentence = source_card.get('sentence', '')
        secondary_word = secondary_card.get('word', '')
        secondary_sentence = secondary_card.get('sentence', '')
        
        cards_batch.append({
            "source_word": source_word,
            "source_sentence": source_sentence,
            "secondary_word": secondary_word,
            "secondary_sentence": secondary_sentence,
            "id": source_card.get('id', '')
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

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ]
        )

        response_content = response.choices[0].message.content
        if not response_content:
            logger.error("Empty response from API")
            return []

        response_dict = json.loads(response_content)
        translations = response_dict.get('translations', [])

        generated_cards = []
        for translation in translations:
            if not isinstance(translation, dict):
                logger.error(f"Invalid translation format: {translation}")
                continue
                
            translated_word = translation.get('translatedWord', '')
            translated_sentence = translation.get('translatedSentence', '')
            card_id = translation.get('id', '')
            
            if translated_word is None:
                translated_word = ''
            if translated_sentence is None:
                translated_sentence = ''
                
            card = {
                "word": str(translated_word).strip(),
                "sentence": str(translated_sentence).strip(),
                "id": str(card_id).strip()
            }
            generated_cards.append(card)

        return generated_cards
        
    except Exception as e:
        logger.error(f"Error in generate_batch_unified_cards: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def evaluate_batch_unified_cards(
    source_cards: List[Dict],
    secondary_cards: List[Dict],
    target_cards: List[Dict],
    source_lang: str,
    target_lang: str,
    secondary_lang: str,
    model: str = GPT_4o_1
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
        
    Returns:
        List[Dict]: List of evaluation results
    """
    system_prompt = f'''You are a multilingual assistant who is proficient in {source_lang}, {secondary_lang} and {target_lang}.'''

    # Prepare the batch of cards for evaluation
    cards_batch = []
    for source_card, secondary_card, target_card in zip(source_cards, secondary_cards, target_cards):
        # Safely get values from dictionaries
        source_word = source_card.get('word', '')
        source_sentence = source_card.get('sentence', '')
        secondary_word = secondary_card.get('word', '')
        secondary_sentence = secondary_card.get('sentence', '')
        target_word = target_card.get('word', '')
        target_sentence = target_card.get('sentence', '')
        
        cards_batch.append({
            "id": source_card.get('id', ''),
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

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ]
        )

        response_content = response.choices[0].message.content
        if not response_content:
            logger.error("Empty response from API")
            return []

        response_dict = json.loads(response_content)
        evaluations = response_dict.get('evaluations', [])

        # Validate and clean up evaluation results
        cleaned_evaluations = []
        for evaluation in evaluations:
            if not isinstance(evaluation, dict):
                logger.error(f"Invalid evaluation format: {evaluation}")
                continue

            # Ensure all required fields exist and have proper types
            cleaned_evaluation = {
                "id": str(evaluation.get('id', '')).strip(),
                "sentenceCorrectness": {
                    "isCorrect": bool(evaluation.get('sentenceCorrectness', {}).get('isCorrect', True)),
                    "explanation": str(evaluation.get('sentenceCorrectness', {}).get('explanation', '')).strip(),
                    "suggestedFix": evaluation.get('sentenceCorrectness', {}).get('suggestedFix')
                },
                "wordUsage": {
                    "isCorrect": bool(evaluation.get('wordUsage', {}).get('isCorrect', True)),
                    "explanation": str(evaluation.get('wordUsage', {}).get('explanation', '')).strip(),
                    "suggestedFixSentence": evaluation.get('wordUsage', {}).get('suggestedFixSentence'),
                    "suggestedFixWord": evaluation.get('wordUsage', {}).get('suggestedFixWord')
                },
                "wordTranslationAccuracy": {
                    "isCorrect": bool(evaluation.get('wordTranslationAccuracy', {}).get('isCorrect', True)),
                    "explanation": str(evaluation.get('wordTranslationAccuracy', {}).get('explanation', '')).strip(),
                    "suggestedFix": evaluation.get('wordTranslationAccuracy', {}).get('suggestedFix')
                },
                "sentenceTranslationAccuracy": {
                    "isCorrect": bool(evaluation.get('sentenceTranslationAccuracy', {}).get('isCorrect', True)),
                    "explanation": str(evaluation.get('sentenceTranslationAccuracy', {}).get('explanation', '')).strip(),
                    "suggestedFix": evaluation.get('sentenceTranslationAccuracy', {}).get('suggestedFix')
                }
            }
            cleaned_evaluations.append(cleaned_evaluation)

        return cleaned_evaluations

    except Exception as e:
        logger.error(f"Error in evaluate_batch_unified_cards: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def generate_batch_cards_with_evaluation_feedback(
    source_cards: List[Dict],
    secondary_cards: List[Dict],
    evaluation_results: List[Dict],
    source_lang: str,
    target_lang: str,
    secondary_lang: str,
    model: str = GPT_4o_1,
    system_prompt: Optional[str] = None
) -> List[Dict]:
    """
    Generates multiple cards with feedback from evaluation results in a single API call.
    """
    if system_prompt is None:
        system_prompt = f'''You are a multilingual assistant who is proficient in {source_lang}, {secondary_lang} and {target_lang}. 
        You are tasked with improving translations based on specific feedback.'''

    # Prepare the batch of cards with feedback
    cards_batch = []
    for source_card, secondary_card, eval_result in zip(source_cards, secondary_cards, evaluation_results):
        # Safely get values from dictionaries
        source_word = source_card.get('word', '')
        source_sentence = source_card.get('sentence', '')
        secondary_word = secondary_card.get('word', '')
        secondary_sentence = secondary_card.get('sentence', '')
        
        feedback = []
        
        sentence_correctness = eval_result.get('sentenceCorrectness', {})
        word_usage = eval_result.get('wordUsage', {})
        word_translation = eval_result.get('wordTranslationAccuracy', {})
        sentence_translation = eval_result.get('sentenceTranslationAccuracy', {})
        
        if not sentence_correctness.get('isCorrect', True):
            feedback.append(f"Sentence correctness issue: {sentence_correctness.get('explanation', '')}")
            if sentence_correctness.get('suggestedFix'):
                feedback.append(f"Suggested fix: {sentence_correctness.get('suggestedFix')}")
        
        if not word_usage.get('isCorrect', True):
            feedback.append(f"Word usage issue: {word_usage.get('explanation', '')}")
            if word_usage.get('suggestedFixSentence'):
                feedback.append(f"Suggested sentence fix: {word_usage.get('suggestedFixSentence')}")
            if word_usage.get('suggestedFixWord'):
                feedback.append(f"Suggested word fix: {word_usage.get('suggestedFixWord')}")
        
        if not word_translation.get('isCorrect', True):
            feedback.append(f"Word translation issue: {word_translation.get('explanation', '')}")
            if word_translation.get('suggestedFix'):
                feedback.append(f"Suggested fix: {word_translation.get('suggestedFix')}")
        
        if not sentence_translation.get('isCorrect', True):
            feedback.append(f"Sentence translation issue: {sentence_translation.get('explanation', '')}")
            if sentence_translation.get('suggestedFix'):
                feedback.append(f"Suggested fix: {sentence_translation.get('suggestedFix')}")
        
        cards_batch.append({
            "id": source_card.get('id', ''),
            "source_word": source_word,
            "source_sentence": source_sentence,
            "secondary_word": secondary_word,
            "secondary_sentence": secondary_sentence,
            "feedback": feedback
        })

    user_prompt = f"""
    **Improve the translations of the given batch of {source_lang} words or phrases and their {secondary_lang} translations into {target_lang}, 
    and then improve the translations of the provided {source_lang} sentences, incorporating the {target_lang} translations of the words or phrases. 
    Use the feedback provided to make the necessary corrections.**  

    ### Cards to improve:
    {json.dumps(cards_batch, ensure_ascii=False, indent=2)}

    ### Important notes:
    - If the input sentence is empty (''), the output sentence should also be empty ('')
    - If the input sentence contains content, translate it naturally while incorporating the translated word/phrase
    - Address all the feedback points in your improved translations
    - Maintain consistency in translation style across all cards

    ### Response structure:  

    Respond in JSON format with the following structure:
    {{
        "translations": [
            {{
                "id": "card_id",
                "translatedWord": "Improved translated word in {target_lang}",
                "translatedSentence": "Improved translated sentence in {target_lang} (empty string if input sentence was empty)"
            }},
            ...
        ]
    }}
    """

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ]
        )

        response_content = response.choices[0].message.content
        if not response_content:
            logger.error("Empty response from API")
            return []

        response_dict = json.loads(response_content)
        translations = response_dict.get('translations', [])

        improved_cards = []
        for translation in translations:
            if not isinstance(translation, dict):
                logger.error(f"Invalid translation format: {translation}")
                continue
                
            translated_word = translation.get('translatedWord', '')
            translated_sentence = translation.get('translatedSentence', '')
            card_id = translation.get('id', '')
            
            if translated_word is None:
                translated_word = ''
            if translated_sentence is None:
                translated_sentence = ''
                
            card = {
                "word": str(translated_word).strip(),
                "sentence": str(translated_sentence).strip(),
                "id": str(card_id).strip()
            }
            improved_cards.append(card)

        return improved_cards

    except Exception as e:
        logger.error(f"Error in generate_batch_cards_with_evaluation_feedback: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def get_system_prompts(source_lang: str, target_lang: str, secondary_lang: str) -> List[str]:
    """
    Returns a list of specialized system prompts for different translation approaches.
    
    Args:
        source_lang: Source language
        target_lang: Target language
        secondary_lang: Secondary language
        
    Returns:
        List[str]: List of specialized system prompts
    """
    return [
        f'''You are a multilingual assistant who is proficient in {source_lang}, {secondary_lang} and {target_lang}. 
        You specialize in precise technical translations with a focus on maintaining exact terminology and technical accuracy.''',
        
        f'''You are a multilingual assistant who is proficient in {source_lang}, {secondary_lang} and {target_lang}. 
        You specialize in creative literary translations that capture the artistic essence and emotional depth of the original text.''',
        
        f'''You are a multilingual assistant who is proficient in {source_lang}, {secondary_lang} and {target_lang}. 
        You specialize in cultural adaptation, ensuring translations are culturally appropriate and resonate with {target_lang} speakers.''',
        
        f'''You are a multilingual assistant who is proficient in {source_lang}, {secondary_lang} and {target_lang}. 
        You specialize in academic and research translations, maintaining scholarly precision and citation accuracy.''',
        
        f'''You are a multilingual assistant who is proficient in {source_lang}, {secondary_lang} and {target_lang}. 
        You specialize in marketing and advertising translations that effectively convey brand messages and marketing intent.'''
    ]

def combine_translations_with_regressor(
    agent_translations: List[Dict],
    source_card: Dict,
    secondary_card: Dict,
    card_id: str,
    source_lang: str,
    target_lang: str,
    secondary_lang: str,
    model: str = GPT_4o_1
) -> Dict:
    """
    Combines multiple translations using a regressor to select the best one.
    
    Args:
        agent_translations: List of translations from different agents
        source_card: Card in source language
        secondary_card: Card in secondary language
        card_id: Card ID
        source_lang: Source language
        target_lang: Target language
        secondary_lang: Secondary language
        model: GPT model to use
        
    Returns:
        Dict: Best combined translation
    """
    source_word, source_sentence, _ = source_card.items()
    secondary_word, secondary_sentence, _ = secondary_card.items()
    
    translations_json = json.dumps(agent_translations, ensure_ascii=False)
    
    system_prompt = f'''You are a multilingual assistant who is proficient in {source_lang}, {secondary_lang} and {target_lang}. 
    You are tasked with selecting the best translation from multiple options or combining them into an optimal solution.'''

    user_prompt = f"""
    **Select the best translation or combine multiple translations of the given {source_lang} word or phrase and its {secondary_lang} translation into {target_lang}, 
    and then select or combine the translations of the provided {source_lang} sentence, incorporating the {target_lang} translation of the word or phrase.**  

    Given word or phrase ({source_lang}): '{source_word}'  
    Given word or phrase ({secondary_lang}): '{secondary_word}'  

    Given sentence ({source_lang}): '{source_sentence}'  
    Given sentence ({secondary_lang}): '{secondary_sentence}'  

    ### Available translations:
    {translations_json}

    ### Important notes:
    - If the input sentence is empty (''), the output sentence should also be empty ('')
    - If the input sentence contains content, select or combine the best translation
    - Consider accuracy, naturalness, and cultural appropriateness in your selection or combination

    ### Response structure:  

    Respond in JSON format with the following structure:
    {{
        "translatedWord": "Best or combined translated word in {target_lang}",
        "translatedSentence": "Best or combined translated sentence in {target_lang} (empty string if input sentence was empty)",
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
    tr_word = response_dict['translatedWord']
    tr_sentence = response_dict['translatedSentence']

    card = {
        "word": tr_word.strip(),
        "sentence": tr_sentence.strip(),
        "id": card_id,
    }

    return card

def generate_card_with_moa(
    source_card: Dict,
    secondary_card: Dict,
    card_id: str,
    source_lang: str,
    target_lang: str,
    secondary_lang: str,
    evaluation_result: Dict,
    model: str = GPT_4o_1
) -> Dict:
    """
    Generates a card using the mixture-of-agents approach.
    
    Args:
        source_card: Card in source language
        secondary_card: Card in secondary language
        card_id: Card ID
        source_lang: Source language
        target_lang: Target language
        secondary_lang: Secondary language
        evaluation_result: Evaluation results from previous generation
        model: GPT model to use
        
    Returns:
        Dict: Generated card using mixture-of-agents approach
    """
    system_prompts = get_system_prompts(source_lang, target_lang, secondary_lang)
    
    agent_translations = []
    for system_prompt in system_prompts:
        translation = generate_card_with_evaluation_feedback(
            source_card,
            secondary_card,
            card_id,
            source_lang,
            target_lang,
            secondary_lang,
            evaluation_result,
            model,
            system_prompt
        )
        agent_translations.append(translation)
    
    combined_card = combine_translations_with_regressor(
        agent_translations,
        source_card,
        secondary_card,
        card_id,
        source_lang,
        target_lang,
        secondary_lang,
        model
    )
    
    return combined_card

def process_cards_batch(
    source_cards: List[Dict],
    secondary_cards: List[Dict],
    source_lang: str,
    target_lang: str,
    secondary_lang: str,
    model: str = GPT_4o_1,
    batch_size: int = 3,
    max_improvement_attempts: int = 3,
    max_retries: int = 5,
    retry_delay: int = 5,
    use_moa: bool = False
) -> Tuple[List[Dict], List[Dict]]:
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
        model: GPT model to use
        batch_size: Number of cards to process in each batch
        max_improvement_attempts: Maximum number of improvement attempts per card
        max_retries: Maximum number of retries for each batch
        retry_delay: Delay in seconds between retries
        use_moa: Whether to use Mixture of Agents for improvement
        
    Returns:
        Tuple[List[Dict], List[Dict]]: Tuple of lists containing generated cards and evaluation results
    """
    logger.info(f"Starting card processing with {len(source_cards)} cards")
    logger.info(f"Using MoA: {use_moa}")
    
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
                    model
                )
                
                # Evaluate translations
                evaluation_results = evaluate_batch_unified_cards(
                    source_batch,
                    secondary_batch,
                    generated_cards,
                    source_lang,
                    target_lang,
                    secondary_lang,
                    model
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
                error_message = str(e)
                logger.error(f"Error processing batch (attempt {retry_count}/{max_retries}): {error_message}")
                
                if retry_count >= max_retries:
                    for card_state in batch:
                        card_queue.mark_as_failed(card_state)
                    logger.error(f"Failed to process batch after {max_retries} retries")
                else:
                    time.sleep(retry_delay)
    
    # Second pass: improve unsuccessful cards
    while not card_queue.is_empty():
        batch = card_queue.get_batch(batch_size)
        if not batch:
            break
            
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Get cards that need improvement
                source_batch = [state.source_card for state in batch]
                secondary_batch = [state.secondary_card for state in batch]
                current_cards = [state.generated_card for state in batch]
                current_evals = [state.evaluation_result for state in batch]
                
                # Generate improved translations
                improved_cards = []
                if use_moa:
                    # Use MoA for improvement
                    for source_card, secondary_card, current_card, current_eval in zip(source_batch, secondary_batch, current_cards, current_evals):
                        improved_card = generate_card_with_moa(
                            source_card,
                            secondary_card,
                            current_card['id'],
                            source_lang,
                            target_lang,
                            secondary_lang,
                            current_eval,
                            model
                        )
                        improved_cards.append(improved_card)
                else:
                    # Use regular feedback-based improvement
                    improved_cards = generate_batch_cards_with_evaluation_feedback(
                        source_batch,
                        secondary_batch,
                        current_evals,
                        source_lang,
                        target_lang,
                        secondary_lang,
                        model
                    )
                
                # Evaluate improved translations
                improved_evals = evaluate_batch_unified_cards(
                    source_batch,
                    secondary_batch,
                    improved_cards,
                    source_lang,
                    target_lang,
                    secondary_lang,
                    model
                )
                
                # Update card states
                for card_state, improved_card, improved_eval in zip(batch, improved_cards, improved_evals):
                    card_state.generated_card = improved_card
                    card_state.evaluation_result = improved_eval
                    
                    quality = calculate_quality_score(improved_eval)
                    if quality == TranslationQuality.PERFECT.value:
                        card_queue.mark_as_successful(card_state)
                    else:
                        card_state.improvement_attempts += 1
                        if card_state.improvement_attempts >= max_improvement_attempts:
                            card_queue.mark_as_failed(card_state)
                
                break  # Success, exit retry loop
                
            except Exception as e:
                retry_count += 1
                error_message = str(e)
                logger.error(f"Error improving batch (attempt {retry_count}/{max_retries}): {error_message}")
                
                if retry_count >= max_retries:
                    for card_state in batch:
                        card_queue.mark_as_failed(card_state)
                    logger.error(f"Failed to improve batch after {max_retries} retries")
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
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
log_filename = os.path.join(log_dir, "card_generation.log")

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
logger = logging.getLogger("card_generation")
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

def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = get_openai_api_key()
azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-07-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
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

def generate_unified_card(
    source_card: Dict,
    secondary_card: Dict,
    card_id: str,
    source_lang: str,
    target_lang: str,
    secondary_lang: str,
    model: str = GPT_4o,
    system_prompt: Optional[str] = None
) -> Dict:
    """
    Universal function for card generation, supporting both simple and full cards.
    
    Args:
        source_card: Card in source language
        secondary_card: Card in secondary language
        card_id: Card ID
        source_lang: Source language
        target_lang: Target language
        secondary_lang: Secondary language
        model: GPT model to use
        system_prompt: Optional custom system prompt
        
    Returns:
        Dict: Generated card
    """
    source_word, source_sentence, _ = source_card.items()
    secondary_word, secondary_sentence, _ = secondary_card.items()

    if system_prompt is None:
        system_prompt = f'''You are a multilingual assistant who is proficient in {source_lang}, {secondary_lang} and {target_lang}.'''

    user_prompt = f"""
    **Translate the given {source_lang} word or phrase along with its {secondary_lang} translation into {target_lang}, and then translate the provided {source_lang} sentence, incorporating the {target_lang} translation of the word or phrase. Use synonyms or related terms where necessary to convey the intended meaning and maintain naturalness in {target_lang}.**  

    Given word or phrase ({source_lang}): '{source_word}'  
    Given word or phrase ({secondary_lang}): '{secondary_word}'  

    Given sentence ({source_lang}): '{source_sentence}'  
    Given sentence ({secondary_lang}): '{secondary_sentence}'  

    ### Important notes:
    - If the input sentence is empty (''), the output sentence should also be empty ('')
    - If the input sentence contains content, translate it naturally while incorporating the translated word/phrase

    ### Response structure:  

    Respond in JSON format with the following structure:
    {{
        "translatedWord": "Translated word in {target_lang}",
        "translatedSentence": "Translated sentence in {target_lang} (empty string if input sentence was empty)"
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
        "id": card_id
    }

    return card

def evaluate_unified_card(
    source_card: Dict,
    secondary_card: Dict,
    target_card: Dict,
    card_id: str,
    source_lang: str,
    target_lang: str,
    secondary_lang: str,
    model: str = GPT_4o
) -> Dict:
    """
    Universal function for card evaluation, supporting both simple and full cards.
    
    Args:
        source_card: Card in source language
        secondary_card: Card in secondary language
        target_card: Card in target language
        card_id: Card ID
        source_lang: Source language
        target_lang: Target language
        secondary_lang: Secondary language
        model: GPT model to use
        
    Returns:
        Dict: Evaluation results
    """
    source_word, source_sentence, _ = source_card.items()
    secondary_word, secondary_sentence, _ = secondary_card.items()
    target_word, target_sentence, _ = target_card.items()

    system_prompt = f'''You are a multilingual assistant who is proficient in {source_lang}, {secondary_lang} and {target_lang}.'''

    user_prompt = f"""
    **Evaluate the correctness of a {target_lang} word and sentence based on their translations from {source_lang} and {secondary_lang}. You will receive a word in {source_lang}, {secondary_lang}, and its translation in {target_lang}, as well as a sentence in {source_lang}, {secondary_lang}, and its translation in {target_lang}. Your task is to assess the quality of the {target_lang} sentence, the usage of the {target_lang} word in the sentence, and the accuracy of the translations from {source_lang} and {secondary_lang} to {target_lang}. For each evaluation point, provide a detailed explanation of your judgment and suggest fixes where applicable, either to the {target_lang} word, the {target_lang} sentence, or both.**  

    Please ensure that the {target_lang} sentence is grammatically correct and natural. Suggest a corrected version if necessary. Verify that the {target_lang} sentence contains the {target_lang} word in some form and suggest using synonyms or related terms if the word is missing. Prioritize naturalness and correctness. Ensure that the translations of both the word and sentence from {source_lang} and {secondary_lang} to {target_lang} are accurate and provide corrections if necessary.  

    ### Important notes:
    - If the input sentence is empty (''), the output sentence should also be empty ('')
    - If the input sentence is empty (''), consider the sentence correctness, word usage in sentence, and sentence translation accuracy as correct (true) without any explanations or suggested fixes
    - If the input sentence contains content, evaluate it as usual

    Here are the provided word and sentence in {source_lang}, {secondary_lang}, and {target_lang}:  

    - Word in {source_lang}: {source_word}  
    - Word in {secondary_lang}: {secondary_word}  
    - Word in {target_lang}: {target_word}  
    - Sentence in {source_lang}: {source_sentence}  
    - Sentence in {secondary_lang}: {secondary_sentence}  
    - Sentence in {target_lang}: {target_sentence}  

    Respond in JSON format with the following structure:  
    {{
        "sentenceCorrectness": {{
            "isCorrect": true/false,  If input sentence is empty (''), this should be true
            "explanation": "Detailed explanation if there is an issue or why it's correct.",  If input sentence is empty (''), this should be null
            "suggestedFix": "Suggested corrected sentence if there is an issue, or null if not applicable."  If input sentence is empty (''), this should be null
        }},
        "wordUsage": {{
            "isCorrect": true/false,  If input sentence is empty (''), this should be true
            "explanation": "Detailed explanation if there is an issue or why it's correct.",  If input sentence is empty (''), this should be null
            "suggestedFixSentence": "Suggested corrected sentence if the word usage is incorrect, or null if not applicable.",  If input sentence is empty (''), this should be null
            "suggestedFixWord": "Suggested corrected word if the word usage is incorrect, or null if not applicable."  If input sentence is empty (''), this should be null
        }},
        "wordTranslationAccuracy": {{
            "isCorrect": true/false,
            "explanation": "Detailed explanation if there is an issue or why it's correct.",
            "suggestedFix": "Suggested correction for translation issues, or null if not applicable."
        }},
        "sentenceTranslationAccuracy": {{
            "isCorrect": true/false,  If input sentence is empty (''), this should be true
            "explanation": "Detailed explanation if there is an issue or why it's correct.",  If input sentence is empty (''), this should be null
            "suggestedFix": "Suggested correction for translation issues, or null if not applicable."  If input sentence is empty (''), this should be null
        }}
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

    res = json.loads(response.choices[0].message.content.strip())
    res['id'] = card_id

    return res

def generate_card_with_evaluation_feedback(
    source_card: Dict,
    secondary_card: Dict,
    card_id: str,
    source_lang: str,
    target_lang: str,
    secondary_lang: str,
    evaluation_result: Dict,
    model: str = GPT_4o,
    system_prompt: Optional[str] = None
) -> Dict:
    """
    Generates a card with feedback from evaluation results.
    
    Args:
        source_card: Card in source language
        secondary_card: Card in secondary language
        card_id: Card ID
        source_lang: Source language
        target_lang: Target language
        secondary_lang: Secondary language
        evaluation_result: Evaluation results from previous generation
        model: GPT model to use
        system_prompt: Optional custom system prompt
    Returns:
        Dict: Improved card based on evaluation feedback
    """
    source_word, source_sentence, _ = source_card.items()
    secondary_word, secondary_sentence, _ = secondary_card.items()
    
    sentence_correctness = evaluation_result.get('sentenceCorrectness', {})
    word_usage = evaluation_result.get('wordUsage', {})
    word_translation = evaluation_result.get('wordTranslationAccuracy', {})
    sentence_translation = evaluation_result.get('sentenceTranslationAccuracy', {})
    
    feedback = []
    
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
    
    system_prompt = system_prompt if system_prompt is not None else f'''You are a multilingual assistant who is proficient in {source_lang}, {secondary_lang} and {target_lang}. 
    You are tasked with improving a translation based on specific feedback.'''

    user_prompt = f"""
    **Improve the translation of the given {source_lang} word or phrase and its {secondary_lang} translation into {target_lang}, 
    and then improve the translation of the provided {source_lang} sentence, incorporating the {target_lang} translation of the word or phrase. 
    Use the feedback provided to make the necessary corrections.**  

    Given word or phrase ({source_lang}): '{source_word}'  
    Given word or phrase ({secondary_lang}): '{secondary_word}'  

    Given sentence ({source_lang}): '{source_sentence}'  
    Given sentence ({secondary_lang}): '{secondary_sentence}'  

    ### Feedback for improvement:
    {chr(10).join(feedback) if feedback else "No specific issues found, but please ensure the translation is natural and accurate."}

    ### Important notes:
    - If the input sentence is empty (''), the output sentence should also be empty ('')
    - If the input sentence contains content, translate it naturally while incorporating the translated word/phrase
    - Address all the feedback points in your improved translation

    ### Response structure:  

    Respond in JSON format with the following structure:
    {{
        "translatedWord": "Improved translated word in {target_lang}",
        "translatedSentence": "Improved translated sentence in {target_lang} (empty string if input sentence was empty)"
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
        "id": card_id
    }

    return card

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
    model: str = GPT_4o
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
    model: str = GPT_4o
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
    model: str = GPT_4o,
    max_iterations: int = 3,
    max_retries: int = 5,
    retry_delay: int = 5
) -> Tuple[List[Dict], List[Dict]]:
    """
    Processes a batch of cards with improved error handling and logging.
    
    Args:
        source_cards: List of cards in source language
        secondary_cards: List of cards in secondary language
        source_lang: Source language
        target_lang: Target language
        secondary_lang: Secondary language
        model: GPT model to use
        max_iterations: Maximum number of improvement iterations
        max_retries: Maximum number of retries for each card
        retry_delay: Delay in seconds between retries
        
    Returns:
        Tuple[List[Dict], List[Dict]]: Tuple of lists containing generated cards and evaluation results
    """
    logger.info(f"Starting batch processing with {len(source_cards)} cards")
    logger.info(f"Parameters: source_lang={source_lang}, target_lang={target_lang}, secondary_lang={secondary_lang}, model={model}")
    
    generated_cards: List[Dict] = []
    evaluation_results: List[Dict] = []
    failed_cards: List[Dict] = []
    
    cards_to_process = list(zip(source_cards, secondary_cards))
    
    for idx, (source_card, secondary_card) in enumerate(tqdm(cards_to_process, total=len(cards_to_process), desc="Processing cards")):
        card_id = source_card.get('id', f"unknown_{idx}")
        retry_count = 0
        success = False
        
        while not success and retry_count <= max_retries:
            try:
                if retry_count > 0:
                    logger.info(f"Retry {retry_count}/{max_retries} for card {card_id}")
                    time.sleep(retry_delay)
                
                current_card = generate_unified_card(
                    source_card,
                    secondary_card,
                    card_id,
                    source_lang,
                    target_lang,
                    secondary_lang,
                    model
                )
                
                current_eval = evaluate_unified_card(
                    source_card,
                    secondary_card,
                    current_card,
                    card_id,
                    source_lang,
                    target_lang,
                    secondary_lang,
                    model
                )
                
                current_quality = calculate_quality_score(current_eval)
                
                if current_quality == TranslationQuality.PERFECT.value:
                    logger.info(f"Card {card_id} has perfect quality")
                    generated_cards.append(current_card)
                    evaluation_results.append(current_eval)
                    success = True
                    continue
                
                best_card = current_card
                best_eval = current_eval
                best_quality = current_quality
                
                for iteration in range(max_iterations):
                    logger.info(f"Attempting to improve card {card_id} (iteration {iteration+1}/{max_iterations})")
                    
                    improved_card = generate_card_with_moa(
                        source_card,
                        secondary_card,
                        card_id,
                        source_lang,
                        target_lang,
                        secondary_lang,
                        best_eval,
                        model
                    )
                    
                    improved_eval = evaluate_unified_card(
                        source_card,
                        secondary_card,
                        improved_card,
                        card_id,
                        source_lang,
                        target_lang,
                        secondary_lang,
                        model
                    )
                    
                    improved_quality = calculate_quality_score(improved_eval)
                    
                    if improved_quality > best_quality:
                        best_card = improved_card
                        best_eval = improved_eval
                        best_quality = improved_quality
                        logger.info(f"Card {card_id} improved: {best_quality:.2f} (iteration {iteration+1})")
                    
                    if best_quality == TranslationQuality.PERFECT.value:
                        logger.info(f"Card {card_id} reached perfect quality")
                        break
                
                generated_cards.append(best_card)
                evaluation_results.append(best_eval)
                success = True
                
            except Exception as e:
                error_message = str(e)
                error_traceback = traceback.format_exc()
                
                is_timeout = (
                    (hasattr(e, 'status_code') and e.status_code in [408, 429, 500, 502, 503, 504]) or
                    isinstance(e, (ConnectionError, Timeout)) or
                    error_message == 'Connection error.'
                )
                
                if is_timeout:
                    logger.warning(f"Timeout error for card {card_id}: {error_message}")
                    logger.debug(f"Traceback: {error_traceback}")
                    time.sleep(retry_delay)
                else:
                    retry_count += 1
                    logger.error(f"Error processing card {card_id} (attempt {retry_count}/{max_retries}): {error_message}")
                    logger.error(f"Traceback: {error_traceback}")
                    
                    if retry_count > max_retries:
                        failed_cards.append({
                            "index": idx,
                            "card_id": card_id,
                            "error": error_message,
                            "retries": retry_count - 1,
                            "traceback": error_traceback
                        })
                        logger.error(f"Failed to process card {card_id} after {retry_count - 1} retries")
    
    if failed_cards:
        logger.error(f"Failed to process {len(failed_cards)} cards out of {len(source_cards)}")
        for failed in failed_cards:
            logger.error(f"Card {failed['card_id']} (index {failed['index']}): {failed['error']} after {failed['retries']} retries")
            logger.debug(f"Traceback: {failed['traceback']}")
    
    logger.info(f"Batch processing completed. Generated {len(generated_cards)} cards, {len(failed_cards)} failed.")
    return generated_cards, evaluation_results

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
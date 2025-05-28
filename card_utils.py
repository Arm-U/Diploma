import json
import os
from pathlib import Path

# Constants for file paths
RU_FINN_FOLDER_WITH_JSON = '../../data/russian-finnish/cards/curated_platform_cards/'
ENG_FINN_OUTPUT_FOLDER = '../../data/english-finnish/cards/test_cards/eng_finn_'
OUTPUT_FOLDER = '../../data/english-finnish/cards/test_cards/eng_finn_'
EVAL_FOLDER = '../../data/english-finnish/cards/eval_results/eng_finn_'

def get_ru_finn_cards_from_file(file_name):
    """
    Получает все карточки из указанного файла.
    
    Args:
        file_name (str): Имя файла с карточками
        
    Returns:
        list: Список карточек
    """
    data = []
    file_path = Path(RU_FINN_FOLDER_WITH_JSON) / file_name
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return [card for card in data if 'isMarkedDeleted' not in card.keys()]

def get_ru_finn_cards_by_ids(file_name, card_ids):
    """
    Получает русско-финские карточки по списку ID из указанного файла.
    
    Args:
        file_name (str): Имя файла с карточками
        card_ids (list): Список ID карточек для поиска
        
    Returns:
        list: Список найденных карточек
    """
    all_cards = get_ru_finn_cards_from_file(file_name)
    return [card for card in all_cards if card['id'] in card_ids]

def get_eng_finn_cards_from_file(file_name):
    """
    Получает англо-финские карточки из указанного файла.
    
    Args:
        file_name (str): Имя файла с карточками
        
    Returns:
        list: Список карточек
    """
    data = []
    file_path = Path(ENG_FINN_OUTPUT_FOLDER) / file_name
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def write_cards_to_file(file_name, cards):
    """
    Записывает карточки в файл.
    
    Args:
        file_name (str): Имя файла для записи
        cards (list): Список карточек для записи
    """
    file_path = os.path.join(OUTPUT_FOLDER, file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)

def write_eval_results_to_file(file_name, results):
    """
    Записывает результаты оценки в файл.
    
    Args:
        file_name (str): Имя файла для записи
        results (dict): Результаты оценки для записи
    """
    file_path = os.path.join(EVAL_FOLDER, file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def get_english_cards():
    """
    Returns a list of all English cards from all chapters.
    
    Returns:
        List[Dict]: List of English cards
    """
    all_cards = []
    
    file_names = [f'sm1_new_kap{i}.json' for i in range(1, 10)]
    file_names.extend([f'sm2_new_kap{i}.json' for i in range(1, 9)])
    file_names.append('sm2_new_puhekieli.json')
    file_names.extend([f'sm3_kap{i}.json' for i in range(1, 9)])
    file_names.extend([f'sm4_kap{i}.json' for i in range(1, 6)])
    
    for file_name in file_names:
        cards = get_eng_finn_cards_from_file(file_name)
        all_cards.extend(cards)
    
    return all_cards

def get_russian_cards():
    """
    Returns a list of all Russian cards from all chapters.
    
    Returns:
        List[Dict]: List of Russian cards
    """
    all_cards = []
    
    file_names = [f'sm1_new_kap{i}.json' for i in range(1, 10)]
    file_names.extend([f'sm2_new_kap{i}.json' for i in range(1, 9)])
    file_names.append('sm2_new_puhekieli.json')
    file_names.extend([f'sm3_kap{i}.json' for i in range(1, 9)])
    file_names.extend([f'sm4_kap{i}.json' for i in range(1, 6)])
    
    for file_name in file_names:
        cards = get_ru_finn_cards_from_file(file_name)
        all_cards.extend(cards)
    
    return all_cards 
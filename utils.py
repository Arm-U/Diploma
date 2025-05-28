import os
import json
from datetime import datetime

def write_cards_to_tmp_file(cards, evals, output_dir="output"):
    """
    Записывает карточки и результаты оценки во временный файл.
    
    Args:
        cards: Список сгенерированных карточек
        evals: Список результатов оценки
        output_dir: Директория для сохранения файлов
        
    Returns:
        str: Путь к созданному файлу
    """
    # Создаем директорию, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Формируем имя файла с текущей датой и временем
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"cards_{timestamp}.json")
    
    # Записываем карточки в файл
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump({
            "cards": cards,
            "evaluations": evals
        }, f, ensure_ascii=False, indent=2)
    
    return file_path 
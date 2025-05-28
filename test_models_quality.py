import sys
import os
from pathlib import Path
from datetime import datetime
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple

# Добавляем путь к модулю в PYTHONPATH
module_path = str(Path.cwd().parent)
if module_path not in sys.path:
    sys.path.append(module_path)

from vocab_preparation.batch_card_generation_pipeline_with_endpoints import (
    process_cards_batch,
    calculate_quality_score,
    TranslationQuality,
    write_cards_to_tmp_file
)

def get_ru_fi_cards():
    all_cards = list()
    
    file_names = [f'sm1_new_kap{i}.json' for i in range(1, 10)]
    file_names.extend([f'sm2_new_kap{i}.json' for i in range(1, 9)])
    file_names.append('sm2_new_puhekieli.json')
    file_names.extend([f'sm3_kap{i}.json' for i in range(1, 9)])
    file_names.extend([f'sm4_kap{i}.json' for i in range(1, 6)])
    
    for file_name in file_names:
        cards = get_ru_finn_cards_from_file(file_name)
        all_cards.extend(cards)
    
    return all_cards

def test_model(model_config: Dict, source_cards: List[Dict], secondary_cards: List[Dict], 
              source_lang: str, target_lang: str, secondary_lang: str) -> Dict:
    """
    Тестирует модель и возвращает результаты
    """
    start_time = time.time()
    
    # Обработка карточек
    successful_cards, successful_evals, failed_cards, failed_evals = process_cards_batch(
        source_cards=source_cards,
        secondary_cards=secondary_cards,
        source_lang=source_lang,
        target_lang=target_lang,
        secondary_lang=secondary_lang,
        model=model_config['model_name'],
        endpoint=model_config['endpoint'],
        batch_size=3,
        max_improvement_attempts=3,
        max_retries=5,
        retry_delay=5
    )
    
    end_time = time.time()
    
    # Подсчет метрик
    total_cards = len(source_cards)
    successful_count = len(successful_cards)
    failed_count = len(failed_cards)
    
    # Вычисление среднего качества для успешных карточек
    quality_scores = [calculate_quality_score(result) for result in successful_evals]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    # Время обработки
    processing_time = end_time - start_time
    
    # Сохраняем результаты в файл
    output_file = write_cards_to_tmp_file(
        successful_cards=successful_cards,
        successful_evals=successful_evals,
        failed_cards=failed_cards,
        failed_evals=failed_evals,
        output_dir=f"output/{model_config['model_name']}"
    )
    
    return {
        'model': model_config['model_name'],
        'total_cards': total_cards,
        'successful_cards': successful_count,
        'failed_cards': failed_count,
        'success_rate': successful_count / total_cards,
        'avg_quality': avg_quality,
        'processing_time': processing_time,
        'cards_per_second': (successful_count + failed_count) / processing_time,
        'output_file': output_file
    }

def main():
    source_lang = 'Finnish'
    secondary_lang = 'Russian'
    target_lang = 'English'
    
    # Получаем случайную выборку из карточек
    ru_fi_cards_sample = np.random.choice(list(get_ru_fi_cards()), size=300, replace=False)
    
    MODELS = {
        'gpt-3.5-turbo': {
            'endpoint': os.getenv('AZURE_OPENAI_ENDPOINT_4o-1-mini'),
            'model_name': 'gpt-3.5-turbo'
        },
        'gpt-4': {
            'endpoint': os.getenv('AZURE_OPENAI_ENDPOINT_4o-1'),
            'model_name': 'gpt-4'
        },
        'gpt-4-turbo-preview': {
            'endpoint': os.getenv('AZURE_OPENAI_ENDPOINT_4o-3'),
            'model_name': 'gpt-4-turbo-preview'
        }
    }
    
    results = []
    
    # Тестируем каждую модель
    for model_name, model_config in tqdm(MODELS.items()):
        print(f"\nТестирование модели: {model_name}")
        result = test_model(
            model_config=model_config,
            source_cards=ru_fi_cards_sample,
            secondary_cards=ru_fi_cards_sample,  # В данном случае исходные и вторичные карточки одинаковые
            source_lang=source_lang,
            target_lang=target_lang,
            secondary_lang=secondary_lang
        )
        results.append(result)
    
    # Преобразуем результаты в DataFrame
    results_df = pd.DataFrame(results)
    
    # Создаем график для качества
    plt.figure(figsize=(15, 5))
    sns.barplot(data=results_df, x='model', y='avg_quality')
    plt.title('Среднее качество генерации карточек')
    plt.ylabel('Среднее качество')
    plt.xlabel('Модель')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('quality_plot.png')
    plt.close()
    
    # График для успешности
    plt.figure(figsize=(15, 5))
    sns.barplot(data=results_df, x='model', y='success_rate')
    plt.title('Процент успешно сгенерированных карточек')
    plt.ylabel('Процент успеха')
    plt.xlabel('Модель')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('success_rate_plot.png')
    plt.close()
    
    # График для скорости
    plt.figure(figsize=(15, 5))
    sns.barplot(data=results_df, x='model', y='cards_per_second')
    plt.title('Скорость генерации карточек')
    plt.ylabel('Карточек в секунду')
    plt.xlabel('Модель')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('speed_plot.png')
    plt.close()
    
    # Сохраняем результаты в CSV
    results_df.to_csv('model_quality_results.csv', index=False)
    
    print("\nРезультаты сохранены в файлах:")
    print("- quality_plot.png")
    print("- success_rate_plot.png")
    print("- speed_plot.png")
    print("- model_quality_results.csv")

if __name__ == "__main__":
    main()

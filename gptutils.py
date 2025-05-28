#!/usr/bin/env python
# coding: utf-8
import openai
from tqdm.notebook import tqdm
import os
from dotenv import load_dotenv, find_dotenv
import requests
import base64


# create ".env" file and put the line with the key in it
# OPENAI_API_KEY="<get_your_key_from_platform.openai.com"
def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = get_openai_api_key()
client = openai.OpenAI(api_key=OPENAI_API_KEY)  # Replace "YOUR_API_KEY" with your actual OpenAI API key

GPT_3_5_TURBO = "gpt-3.5-turbo"
GPT_4_TURBO_PREVIEW = "gpt-4-turbo-preview"
GPT_4 = 'gpt-4'

def generate_finnish_sentences(words, model=GPT_3_5_TURBO):
    word_to_sentence = {}
    for word in tqdm(words, desc="Generating sentece examplesn"):
        #         prompt = f"Generate a simple, useful, beginner-friendly sentence in Finnish using the word or phrase: '{word}'. The sentence should be useful and easy to understand for someone learning Finnish."
        # prompt = f"Generate a simple, useful, beginner-friendly sentence in Finnish using the word or phrase: '{word}'. The sentence should be useful and easy to understand for someone learning Finnish. Provide only the Finnish sentence, without any translation or additional information."
        prompt = f'''Generate an idiomatic, simple, useful sentence in Finnish using the word or phrase:
        
        === 
        '{word}'
        ===
         
        The sentence should be useful and easy to understand for someone learning Finnish. Provide only the Finnish sentence, without any translation or additional information.
        '''

        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        #         print(response.choices[0].message)

        # Correctly accessing the response data
        # Ensure you're correctly interpreting the structure of the response
        sentence = response.choices[0].message.content if response.choices else 'No response generated.'

        # Map the word or phrase to the generated sentence
        word_to_sentence[word] = sentence.strip()
    return word_to_sentence


# system_prompt = "You are a professional russian-finnish translator. Use 'ё' in russian written language instead of 'е', when it is grammatically more correct."  # When 'е' suits better use it." #" When 'е' suits better use it."
system_prompt = '''You are a professional russian-finnish translator. '''


def translate_word_or_phrase_to_russian(phrase, model=GPT_3_5_TURBO):
    # prompt = f"Translate the following Finnish phrase to Russian: '{phrase}'. Provide only the translation to russian sentence, without any additional information."
    prompt = f'''Translate the following Finnish phrase to Russian: 
    
    ===
    '{phrase}'
    === 
    
    Provide only the translation to russian sentence, without any additional information.'''
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "system",
            "content": system_prompt
        },
            {
                "role": "user",
                "content": prompt
            }],
        temperature=0.0
    )

    # Correctly accessing the response data
    # Ensure you're correctly interpreting the structure of the response
    translation = response.choices[0].message.content if response.choices else 'No response generated.'

    translation = translation.strip()
    return translation


def translate_word_or_phrase_to_russian_with_context(phrase, context, model=GPT_3_5_TURBO):
    prompt = f"""Translate the following Finnish word to Russian: '{phrase}'. \n 
                To understand tranlsation context for words with different variants of translation use the following sentence with the word: '{context}'. 
                Provide only the translation to russian word, without any additional information."""
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "system",
            "content": system_prompt
        },
            {
                "role": "user",
                "content": prompt
            }],
        temperature=0.0
    )

    # Correctly accessing the response data
    # Ensure you're correctly interpreting the structure of the response
    translation = response.choices[0].message.content if response.choices else 'No response generated.'

    translation = translation.strip()
    return translation


def translate_to_russian(data, model=GPT_3_5_TURBO):
    translations = []
    for finnish_word, finnish_sentence in tqdm(data, desc="Translating to russian"):
        # Construct the prompt for translation
        if finnish_sentence:
            russian_sentence = translate_word_or_phrase_to_russian(finnish_sentence, model=model)
            russian_word = translate_word_or_phrase_to_russian_with_context(finnish_word, finnish_sentence, model=model)
            translations.append((finnish_word, finnish_sentence, russian_word, russian_sentence))
        else:
            russian_word = translate_word_or_phrase_to_russian(finnish_word, model=model)
            translations.append((finnish_word, finnish_sentence, russian_word, ""))
    return translations


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Define the prompt
IMAGE_DEFAULT_PROMPT = "Create a list of Finnish words and phrases from the screenshot. Use plain text, output row by row. Use plain text without bullets, so I can copy it to Excel."

def process_image(image_path, prompt = IMAGE_DEFAULT_PROMPT):
    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    # Extract the text from the response
    response_json = response.json()
    try:
        result_text = response_json['choices'][0]['message']['content'].strip()
    except (KeyError, IndexError):
        result_text = "Error: Could not parse response."

    return result_text
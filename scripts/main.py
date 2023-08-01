#importing dependencies

import re
from youtube_transcript_api import YouTubeTranscriptApi
import torch
import torchaudio
import openai
import textwrap
from transformers import pipeline, AutoTokenizer
from transformers import pipeline

def extract_video_id(youtube_url):
    match = re.search(r"v=([A-Za-z0-9_-]+)", youtube_url)
    if match:
        video_id = match.group(1)
        return video_id
    else:
        raise ValueError("Invalid YouTube URL")

def get_youtube_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text = ""
    for segment in transcript:
        transcript_text += segment["text"] + " "
    return transcript_text

# def translate_transcript(transcript_text, model_checkpoint="Helsinki-NLP/opus-mt-en-es"):
#     translator = pipeline("translation", model=model_checkpoint)
#     max_length = 512
#     segments = [transcript_text[i:i+max_length] for i in range(0, len(transcript_text), max_length)]
#     translated_text = ""
#     for segment in segments:
#         result = translator(segment)
#         translated_text += result[0]['translation_text']
#     return translated_text

def summarize_transcript(transcript_text, model_checkpoint='stevhliu/my_awesome_billsum_model', chunk_size=200):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    summarizer = pipeline("summarization", model=model_checkpoint, tokenizer=tokenizer)
    words = transcript_text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
        summary_text = summary[0]['summary_text']
        summaries.append(summary_text)
    final_summary = ' '.join(summaries)
    return final_summary

def split_text_into_chunks(text, max_chunk_size):
    return textwrap.wrap(text, max_chunk_size)

def generate_chat_summaries(transcript_text):
    max_chunk_size = 4000
    transcript_chunks = split_text_into_chunks(transcript_text, max_chunk_size)
    summaries = ""
    for chunk in transcript_chunks:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{chunk}\n\nCreate short concise summary"}
            ],
            max_tokens=250,
            temperature=0.5
        )
        summaries += response['choices'][0]['message']['content'].strip() + " "
    return summaries

def generate_steps_from_text(transcript_text, api_key):
    # Provide the API key before making the API call
    openai.api_key = api_key
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a technical instructor."},
            {"role": "user", "content": transcript_text},
            {"role": "user", "content": "Generate steps to follow from text."},
        ]
    )
    guide = response['choices'][0]['message']['content']
    return guide

def generate_quiz_questions(transcript_text, api_key):
    # Provide the API key before making the API call
    openai.api_key = api_key
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates questions."},
            {"role": "user", "content": transcript_text},
            {"role": "user", "content": "Generate 10 quiz questions based on the text with multiple choices."},
        ]
    )
    quiz_questions = response['choices'][0]['message']['content']
    return quiz_questions
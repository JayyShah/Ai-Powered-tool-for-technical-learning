# Introduction 

This repository leverages the power of Large Language Models along with Python functions to process YouTube videos and generate summaries, steps, and quiz questions based on the video's transcript.

The project was initially made Configuring a free ML development environment using Amazon SageMaker Studio Lab, integrate it with open-source ML models from Hugging Face, and use OpenAI's ChatGPT APIs.

# Problem Statement: 
According to a recent survey most respondents view videos 2-to-4 times per week, and in a 2019 Google study, 86% of U.S. viewers said they used YouTube to learn new things.

This shouldn't come as a surprise: YouTube has educational content on pretty much any topic, from academic subjects like math and programming to hands-on projects, tutorials, and preparation for professional certifications.

But as a learning tool, YouTube isn't perfect. Some videos have too many ads and sponsorship interruptions, some are slowed down by non-essential information, while others require viewers to pause frequently just to follow the steps.

# Solution :

Configuring a free ML development environment using Amazon SageMaker Studio Lab, integrate it with open-source ML models from Hugging Face, and use OpenAI's ChatGPT APIs.


Another way is creating a Streamlit Web App where the user can add their Youtube URL and OpenAI API keys and can Generate Summary, using Hugging Face Translation models they can translate into different languages and to test their technical skills they can Create quiz using Open AI.

# Table of Contents
1.Introduction

This project aims to automate the process of summarizing YouTube video content and generating quizzes based on the video's transcript. It utilizes the YouTube Transcript API to extract the transcript of a specified video URL. Additionally, it employs the OpenAI GPT-3 model through the transformers library to generate summaries, steps, and quiz questions based on the extracted transcript. The summaries and quiz questions are designed to provide key information and assess the user's understanding of the video's content.

# 2. Usage
To use the functions provided by this project, follow these steps:

* Clone the Repo. 
* Create a virtual environment - `(python -m venv <name/of/your/virtualenv>)`
* Activate the environment - `(<name/of/your/virtualenv/Scripts/activate)`
* Install the dependencies - `(pip install -r requirements.txt)`
* Run Streamlit app - `(streamlit run app.py)`
* Enter the Youtube URL, For optimal performance keep the limit of the video till `~30 min`.
* Enter the OpenAI API keys.  

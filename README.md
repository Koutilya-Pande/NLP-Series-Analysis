# Situation
In the era of content consumption, analyzing and understanding narratives in series has become increasingly important. Fans and researchers alike want to dive deeper into character interactions, themes, and plot developments. However, existing tools often lack the capability to comprehensively analyze series content and facilitate interactive engagement with characters.
## Analyze Your Favorite Series with NLP

This project empowers users to analyze their favorite series using Natural Language Processing (NLP) techniques and Large Language Models (LLMs). From scraping data to building chatbots, it provides a comprehensive approach to exploring characters, themes, and textual classifications. By integrating models into an intuitive web interface with Gradio, this project combines cutting-edge NLP technology with interactive tools.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)

---

## Overview

This project leverages modern NLP techniques to analyze series content, such as episode transcripts, character dialogues, and themes. It involves scraping data, building character networks using Named Entity Recognition (NER), training custom text classifiers, extracting themes using zero-shot classification, and creating character-based chatbots using LLMs.

### Objectives:
- **Web Scraping**: Automatically gather and build datasets from series.
- **Character Network**: Build networks between characters using NER models.
- **Theme Classification**: Extract key themes using zero-shot classification.
- **Text Classification**: Train a custom model to classify series-specific content.
- **Character Chatbot**: Interact with series characters using an LLM-based chatbot.
- **Web Interface**: Deploy all models in a user-friendly Gradio web interface.

---

## Features

- **Web Scraping**: Collects data from online sources using Scrapy.
- **Named Entity Recognition**: Uses Spacy to extract characters and visualize their relationships with NetworkX and PyViz.
- **Zero-Shot Classification**: Classifies series themes without needing labeled training data.
- **Custom Text Classifier**: Trained to classify text specific to the series.
- **LLM-Based Chatbot**: Chat with characters using Large Language Models (LLMs).
- **Gradio Web Interface**: Access all models through a simple web app.

---

## Tech Stack

- **Languages**: Python
- **Libraries**: Spacy, Hugging Face Transformers, NLTK, Scrapy
- **Models**: Llama 3.2-3B-instruct, BERT, DistilBert, bart-large-mnli(Zero-Shot Classification)
- **Visualization**: NetworkX, PyViz
- **Web Framework**: Gradio (for GUI)

---
## Project Structure
```
/Naruto NLP/
│
├── character_chatbot/
│   ├── character_chatbot.py
│   ├── chatbot_developement.ipynb
│   ├── __init__.py
│
├── character_network/
│   ├── character_network_generation.py
│   ├── character_network_generatir.ipynb
│   ├── named_entity_recognizer.py
│   ├── naruto.html
│   ├── __init__.py
│
├── gradio_app.py
├── requirements.txt
│
├── text_classification/
│   ├── cleaner.py
│   ├── jutsu_classifier.ipynb
│   ├── jutsu_classifier.py
│   ├── trainer.py
│   ├── training_utils.py
│   ├── __init__.py
│
├── theme_classifier/
│   ├── theme_classification_developement.ipynb
│   ├── theme_classifier.py
│   ├── __init__.py
│
├── utils/
│   ├── data_loader.py
│   ├── __init__.py
│
├── web_scrape/
│   ├── jujutsu.py
│
└── README.md
```
## Installation

Clone the repository:

```
git clone https://github.com/Koutilya-Pande/NLP-Series-Analysis.git
cd your_project_directory
 ```
Create a virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate 
```
Install dependencies:

Make sure you have all the necessary libraries installed by running:

```pip install -r requirements.txt```

Usage

To run the Gradio web application:


```python gradio_app.py```
Navigate to the local web URL provided in the terminal to use the web interface.

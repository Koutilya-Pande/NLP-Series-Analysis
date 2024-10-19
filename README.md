# Analyze Your Favorite Series with NLP

This project empowers users to analyze their favorite series using Natural Language Processing (NLP) techniques and Large Language Models (LLMs). From scraping data to building chatbots, it provides a comprehensive approach to exploring characters, themes, and textual classifications. By integrating models into an intuitive web interface with Gradio, this project combines cutting-edge NLP technology with interactive tools.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)

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

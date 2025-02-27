import gradio as gr
from theme_classifier import ThemeClassifier
from utils.data_loader import load_subtitles_dataset
import pandas as pd
import os
from character_network import CharacterNetworkGenerator, NamedEntityRecognizer
from text_classification import JutsuClassifier
from dotenv import load_dotenv
import os
from character_chatbot import CharacterChatbot
load_dotenv()

def get_themes(theme_list_str, subtitles_path, save_path):
    # Removed print statements for terminal output
    try:
        # Split the theme list and create the classifier
        print(f"Subtitles path: {subtitles_path}")  # Print the subtitles path
        print(f"NER save path: {save_path}")
        theme_list = theme_list_str.split(',')
        theme_classifier = ThemeClassifier(theme_list)
        output_df = theme_classifier.get_themes(subtitles_path, save_path)
        # Check if the output DataFrame is empty
        if output_df.empty:
            return pd.DataFrame(columns=['Theme', 'Score'])  # Return an empty DataFrame

        # Remove dialogue from the theme list
        theme_list = [theme for theme in theme_list if theme != 'dialogue']
        output_df = output_df[theme_list]
        
        # Sum the scores for each theme
        output_df = output_df.sum().reset_index()
        output_df.columns = ['Theme', 'Score']
        
        # Removed terminal output

        return output_df  # Return the DataFrame to be displayed
    except Exception as e:
        return pd.DataFrame(columns=['Theme', 'Score'])  # Return an empty DataFrame on error
    
def get_character_network(subtitles_path,ner_path):
    ner = NamedEntityRecognizer()
    print("Function Called")
    ner_df = ner.get_ners(subtitles_path,ner_path)
    print("NERs obtained")

    character_network_generator = CharacterNetworkGenerator()
    relationship_df = character_network_generator.generate_character_network(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)

    print("Done")  # Debugging step to check the generated HTML

    return html

def classify_jutsu(text_classification_model,data_path,text_to_classify):
    jutsu_classifier = JutsuClassifier(model_path = text_classification_model,
                                       data_path = data_path,
                                       huggingface_token = os.getenv("huggingface_token") )
    output = jutsu_classifier.classify_jutsu(text_to_classify)
    output = output[0]
    return output

def chatbot_character(message,history):
    character_chatbot = CharacterChatbot("Koutilya/Naruto-Chatbot-Llama-3.2-3B-Instruct",
                                         huggingface_token = os.getenv("huggingface_token"))
    output = character_chatbot.chat(message,history)
    output = output['content'].strip()
    return output

def main():

    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Naruto Theme Classifier (Zero-Shot Classification)</h1>")
                with gr.Row():
                    with gr.Column():
                        # Output table for theme scores
                        output_table = gr.Dataframe(label="Theme Scores")
                    with gr.Column():
                        theme_list_str = gr.Textbox(label="Theme List", placeholder="Enter themes separated by commas")
                        subtitles_path = gr.Textbox(label="Subtitles Path / Script Path", placeholder="Enter the path to subtitles")
                        save_path = gr.Textbox(label="Save Path", placeholder="Enter the path to save results")
                        theme_classifier_button = gr.Button("Classify Themes")
                        theme_classifier_button.click(get_themes, inputs=[theme_list_str, subtitles_path, save_path], outputs=[output_table])

            # Character Network Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Network (NERs and Graphs)</h1>")
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                    with gr.Column():
                        subtitles_path = gr.Textbox(label="Subtitles or Script Path")
                        ner_path = gr.Textbox(label="NERs save path")
                        get_network_graph_button = gr.Button("Get Character Network")
                        get_network_graph_button.click(get_character_network, inputs=[subtitles_path,ner_path], outputs=[network_html])

        # Jutsu Classification with LLMs
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Jutsu Classification with LLMs</h1>")
                with gr.Row():
                    with gr.Column():
                        text_classification_output = gr.Textbox(label="Jutsu Classification")
                    with gr.Column():
                        text_classification_model = gr.Textbox(label="Model Path")
                        data_path = gr.Textbox(label="Data Path")
                        text_to_classify = gr.Textbox(label="Text to Classify")
                        classify_text_button = gr.Button("Classify Jutsu")
                        classify_text_button.click(classify_jutsu, inputs=[text_classification_model,data_path,text_to_classify], outputs=[text_classification_output])

        # Chatbot Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Naruto Chatbot</h1>")
                gr.ChatInterface(chatbot_character)

    iface.launch(share=True)


if __name__ == "__main__":
    main()

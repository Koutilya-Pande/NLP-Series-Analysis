import gradio as gr
from theme_classifier import ThemeClassifier
from utils.data_loader import load_subtitles_dataset
import pandas as pd
import os


def get_themes(theme_list_str, subtitles_path, save_path):
    # Removed print statements for terminal output
    try:
        # Split the theme list and create the classifier
        theme_list = theme_list_str.split(',')
        theme_classifier = ThemeClassifier(theme_list)
        
        # Removed terminal output
        output_df = theme_classifier.get_themes(subtitles_path, save_path)
        
        # Check if the output DataFrame is empty
        if output_df.empty:
            return pd.DataFrame(columns=['Theme', 'Score'])  # Return an empty DataFrame

        # Removed terminal output
        
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

    iface.launch(share=True)


if __name__ == "__main__":
    main()

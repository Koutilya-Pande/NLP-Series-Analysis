import gradio as gr
from theme_classifier import ThemeClassifier
from utils.data_loader import load_subtitles_dataset
import pandas as pd
import os


def get_themes(theme_list_str, subtitles_path, save_path):
    print(f"get_themes called with: {theme_list_str}, {subtitles_path}, {save_path}")
    try:
        # Split the theme list and create the classifier
        theme_list = theme_list_str.split(',')
        theme_classifier = ThemeClassifier(theme_list)
        
        print("Loading and processing subtitles...")
        output_df = theme_classifier.get_themes(subtitles_path, save_path)
        
        # Check if the output DataFrame is empty
        if output_df.empty:
            print("Output DataFrame is empty.")
            return pd.DataFrame(columns=['Theme', 'Score'])  # Return an empty DataFrame

        print(f"Output DataFrame shape: {output_df.shape}")
        print(f"Output DataFrame columns: {output_df.columns}")
        
        # Remove dialogue from the theme list
        theme_list = [theme for theme in theme_list if theme != 'dialogue']
        output_df = output_df[theme_list]
        
        # Sum the scores for each theme
        output_df = output_df.sum().reset_index()
        output_df.columns = ['Theme', 'Score']
        
        print("\nFinal theme scores:")
        print(output_df)

        return output_df  # Return the DataFrame to be displayed
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return pd.DataFrame(columns=['Theme', 'Score'])  # Return an empty DataFrame on error


def test_get_themes():
    theme_list_str = "Adventure,Comedy,Action,Drama,Friendship,Betrayal,Revenge,Love,Power,Growth,Sacrifice,Hope"
    subtitles_path = "C:/Users/kouti/OneDrive/Desktop/Naruto_NLP/data/Subtitles"
    save_path = "C:/Users/kouti/OneDrive/Desktop/Naruto_NLP/output/theme_analysis.csv"

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    return get_themes(theme_list_str, subtitles_path, save_path)


def main():
    print("Starting Gradio interface...")
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
                        
                        # Test case button
                        test_case_button = gr.Button("Run Test Case")
                        test_case_button.click(test_get_themes, inputs=[], outputs=[output_table])

    print("Launching Gradio interface...")
    iface.launch(share=True)


if __name__ == "__main__":
    main()

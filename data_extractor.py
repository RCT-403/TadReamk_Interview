# Extract the data from the npy files in the folder vector_colors and folder vector_design_codes
# and save them in a csv file

import numpy as np
import pandas as pd
import os

def check_unique_shapes(data_list):
    unique_shapes = set()
    for data in data_list:
        unique_shapes.add(data.shape)
    return unique_shapes

# Load the data from the npy files
def load_data():
    # Initialize lists to hold the data
    data_color_list = []
    data_design_list = []
    all_phrases = []
    
    # Load the data from the npy files in vector_colors folder
    for file_name in os.listdir("vector_colors"):
        if file_name.endswith(".npy"):
            file_path = os.path.join("vector_colors", file_name)
            data_color = np.load(file_path, allow_pickle=True)
            data_color_list.append(data_color)

    # Load the data from the npy files in vector_design_codes folder
    for file_name in os.listdir("vector_design_codes"):
        if file_name.endswith(".npy"):
            file_path = os.path.join("vector_design_codes", file_name)
            data_design = np.load(file_path, allow_pickle=True)
            if data_design.ndim == 1:
                data_design = np.zeros(1586).reshape(1, 1586)
                print(file_name)
            data_design_list.append(data_design)
    
    for file_name in os.listdir("words"):
        if file_name.endswith('.txt'):
            file_path = os.path.join("words", file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                    # Read the entire content of the file as a single phrase
                    phrase = f.read().strip()
                    # Append the phrase to the all_phrases list
                    all_phrases.append(phrase)
            
            

    # Concatenate all the data into single arrays
    data_color = np.concatenate(data_color_list, axis=0)
    data_design = np.concatenate(data_design_list, axis=0)

    print(data_color.shape)
    print(data_design.shape)

    return data_color, data_design, all_phrases

# Save the data in a csv file
def save_data(data_color, data_design, all_phrases):
    # Save the data in a csv file
    df_color = pd.DataFrame(data_color)
    df_design = pd.DataFrame(data_design)
    df_color.to_csv("data_colors.csv", index=False)
    df_design.to_csv("data_designs.csv", index=False)
    
    with open('words.txt', 'w', encoding='utf-8') as f:
        for phrase in all_phrases:
            f.write(phrase + '\n')
    
    
    print("Data saved in data_colors.csv and data_designs.csv")
    
data_color, data_design, all_phrases = load_data()
save_data(data_color, data_design, all_phrases) 





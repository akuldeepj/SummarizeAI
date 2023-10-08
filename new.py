import os
import random

# Set the path to the folder containing the stories
data_dir = "D:\SummarizeAI\cnn_stories\cnn\stories"

# Set the percentage of stories to use for each set
train_percent = 0.8
val_percent = 0.1
test_percent = 0.1

# Get a list of all the story files in the folder
all_stories = [f for f in os.listdir(data_dir) if f.endswith(".story")]

# Shuffle the list of story files
random.shuffle(all_stories)

# Split the stories into train, validation, and test sets
train_stories = all_stories[:int(len(all_stories) * train_percent)]
val_stories = all_stories[int(len(all_stories) * train_percent):int(len(all_stories) * (train_percent + val_percent))]
test_stories = all_stories[int(len(all_stories) * (train_percent + val_percent)):]

# Print the number of stories in each set
print("Number of train stories:", len(train_stories))
print("Number of validation stories:", len(val_stories))
print("Number of test stories:", len(test_stories))

# Amazon Review Sentiment Analysis

This repository contains a Python-based project for scraping Amazon product reviews and performing sentiment analysis on them. The project involves training a sentiment analysis model on a dataset of reviews and using this model to classify the sentiment of reviews scraped from Amazon.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Overview
The project aims to classify Amazon product reviews as positive, negative, or neutral based on their textual content. It involves the following steps:

1. *Preprocessing Text Data*: Cleaning and preparing text data by removing punctuation, converting to lowercase, and removing stopwords.
2. *Training a Model*: Using a decision tree classifier trained on preprocessed review data.
3. *Scraping Amazon Reviews*: Extracting reviews from Amazon product pages.
4. *Analyzing Reviews*: Applying the trained model to classify the sentiment of scraped reviews.
5. *Summarizing Sentiment*: Aggregating sentiment scores to determine the overall sentiment of the reviews.

## Installation

To get started with this project, follow these steps:

1. *Clone the Repository*:
    sh
    git clone https://github.com/yourusername/amazon-review-sentiment-analysis.git
    cd amazon-review-sentiment-analysis
    

2. *Install Dependencies*:
    Make sure you have Python installed, then install the required Python packages:
    sh
    
## Usage

Follow these steps to run the sentiment analysis:

1. *Prepare the Training Data*:
   Place your training data CSV file (e.g., flipkart.csv) in the project directory. This file should contain at least two columns: review and rating.

2. *Run the Main Script*:
   Execute the main_process function with the URL of the Amazon product reviews you want to analyze:
    python
    from sentiment_analysis import main_process

    url = "https://www.amazon.in/dp/product-id"
    result = main_process(url)
    print(result)
    

## Project Structure

- sentiment_analysis.py: Contains all the functions for preprocessing, training, scraping, and analyzing reviews.
- requirements.txt: Lists all the Python dependencies required for the project.
- README.md: Provides an overview and instructions for the project (this file).

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to fork the repository and submit a pull request. Please make sure to update tests as appropriate.

Feel free to clone and develop this project further. Your contributions and feedback are appreciated!
![WhatsApp Image 2024-06-04 at 09 16 21_cf90056e](https://github.com/Shanchana/Review-Sentiment-Analyzer/assets/137145340/f968f00b-7ce3-4187-a11e-cf297a146805)
![WhatsApp Image 2024-06-04 at 09 16 21_d52419db](https://github.com/Shanchana/Review-Sentiment-Analyzer/assets/137145340/dfd2f21b-1de4-4e64-b10b-264fa5a8b20a)

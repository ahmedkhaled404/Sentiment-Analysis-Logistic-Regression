<h1>Sentiment Analysis on movie reviews using NLP</h1>

**1. Introduction**

*Objective:* This project aims to develop a sentiment analysis pipeline for movie reviews on IMDB. This involves classifying sentiment into positive and negative categories using Natural Language Processing (NLP) techniques and machine learning models. The project aims to build an efficient model for sentiment classification, compare the performance of traditional methods, and provide insights into the effectiveness of these methods.

**2. Data Description**

*Dataset Source:*

-   IMDB Movie Ratings Sentiment Analysis
-   [Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis)

*Data Overview:*

-   **Number of Instances:** 39,723
-   **Features:**
    -   text: The review text.
    -   label: Sentiment label (0 for negative, 1 for positive).

*Data Splits:*

-   **Train + Validation Set:** 85%
-   **Test Set:** 15%
-   **Validation Set:** 15%

**3. Baseline Experiments**

*Initial Steps:*

1.  **Data Import and Inspection:**
    -   Loaded data from CSV file.
    -   Checked for null values and data types.
2.  **Data Exploration (EDA):**
    -   Verified the shape and distribution of data.
    -   Checked for duplicates and removed them.
    -   Analyzed class imbalance and word count distribution.

*Baseline Analysis Results:*

-   **Class Distribution:** Balanced (almost equal number of positive and negative reviews).

    ![image](https://github.com/user-attachments/assets/0555d4a0-e6ca-43e2-9468-6aad4ac5b38d)

-   **Word Count Distribution:** Right-skewed with few anomalies.

    ![image](https://github.com/user-attachments/assets/679eac5a-14b4-486a-9f1e-50f3890e8031)

**4. Advanced Experiments**

*4.1 Data Preprocessing*

**Text Cleaning:**

-   Converted text to lowercase and removed punctuation.

**Tokenization and Lemmatization:**

-   Used NLTK (Natural Language Tool Kit) for tokenization and lemmatization.
-   Implemented part-of-speech tagging for accurate lemmatization.

**Stop Words Removal:**

-   Removed common stop words using NLTK’s stop words list.

**Feature Extraction:**

-   Used TF-IDF vectorizer to transform text data into feature vectors.

**Text before and after applying analysis:**

| Even though I have great interest in Biblical movies, I was bored to death every minute of the movie. Everything is bad. The movie is too long, the acting is most of the time a Joke and the script is horrible. I did not get the point in mixing the story about Abraham and Noah together. So if you value your time and sanity stay away from this horror. |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| even though great interest biblical movie bore death every minute movie everything bad movie long acting time joke script horrible get point mix story abraham noah together value time sanity stay away horror                                                                                                                                                 |

*4.2 Model Development and Tuning*

**Train-Test Split:**

-   Split the dataset into training, validation, and test sets.

**Models Developed:**

1.  **Logistic Regression:**
    -   **Parameter Tuning:** Used GridSearchCV to find the best parameters.
    -   **Best Parameters:** C values of [0.01, 0.1, 1, 10, 100] and solvers 'liblinear' and 'saga'.
    -   **Validation Accuracy:** 88.67%
    -   **Test Accuracy:** 89.31%
2.  **Naive Bayes:**
    -   **Parameter Tuning:** GridSearchCV was used to find the best alpha values.
    -   **Best Parameters:** alpha values of [0.01, 0.1, 1, 10, 100].
    -   **Validation Accuracy:** 86.04%
    -   **Test Accuracy:** 86.02%

**Performance Comparison:**

**Comparative Analysis:**

Compared performance of Logistic Regression and Naive Bayes using metrics such as accuracy, confusion matrix, and F1 score.


-   **Classification Report:**
  
    ![Screenshot 2024-07-21 184711](https://github.com/user-attachments/assets/bdec8fea-d751-4b82-bd27-5284a96a04b1)

   
-   **Confusion Matrix Comparison:**

![image](https://github.com/user-attachments/assets/489bc90c-569c-420e-8930-48379cb268be)

**5. Overall Conclusion**

*Best Model:*

-   Logistic Regression is identified as the best-performing model with the highest accuracy and ROC AUC score.

*Learning Curves:*

-   Generated learning curves for Logistic Regression to analyze model performance with varying training sizes.

    ![](media/a6cb39aca95e745595f5b46cdae5cef3.png)

*ROC Curve:*

-   **ROC AUC Score:** 0.95
-   **ROC Curve Plot:** Displays the trade-off between true positive rate and false positive rate for the Logistic Regression model

    ![](media/ecb533f3ee94cdfe7ee9c028ec6f24d5.png)

**Insights:**

-   Logistic Regression outperforms Naive Bayes in both validation and test accuracy.
-   Learning curves indicate that the model benefits from more training data.
-   The ROC curve confirms that the Logistic Regression model provides an excellent trade-off between sensitivity and specificity.

**Additional Information**

*Report of Resources*

**1. Libraries and Tools Used**

| Library/Tool                    | Description                                                                                                                                          | Link                                                    |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| Python                          | The primary programming language used for the project.                                                                                               | [Python](https://www.python.org/)                       |
| pandas                          | A data manipulation library used for loading, processing, and analyzing data.                                                                        | pandas Documentation                                    |
| numpy                           | A library for numerical operations used in data manipulation and analysis.                                                                           | [numpy Documentation](https://numpy.org/)               |
| matplotlib                      | A plotting library used for visualizing data and model performance metrics.                                                                          | [matplotlib Documentation](https://matplotlib.org/)     |
| seaborn                         | A data visualization library based on Matplotlib, used for creating attractive and informative statistical graphics.                                 | seaborn Documentation                                   |
| NLTK (Natural Language Toolkit) | A suite of libraries and programs for natural language processing, used for text preprocessing, tokenization, lemmatization, and stop words removal. | [NLTK Documentation](https://www.nltk.org/)             |
| scikit-learn                    | A machine learning library used for model development, tuning, and evaluation.                                                                       | [scikit-learn Documentation](https://scikit-learn.org/) |
| GridSearchCV                    | A module in scikit-learn used for hyperparameter tuning of machine learning models.                                                                  | GridSearchCV Documentation                              |
| TF-IDF Vectorizer               | A tool in scikit-learn for transforming text data into feature vectors.                                                                              | TF-IDF Vectorizer Documentation                         |

*2. External Resources*

| Resource             | Description                                                                                                     | Link                                                                                            |
|----------------------|-----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| Dataset              | IMDB Movie Ratings Sentiment Analysis dataset, which includes movie reviews and corresponding sentiment labels. | [Kaggle Dataset](https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis) |
| NLTK Stop Words List | A list of common stop words used for text preprocessing and removing irrelevant words from the dataset.         | NLTK Stop Words List                                                                            |


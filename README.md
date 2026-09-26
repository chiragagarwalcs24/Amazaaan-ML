# 🚀 AmazaanML Team — Business Entity Resolution

Welcome to the **AmazaanML Team** repository! This project is our complete, end-to-end Machine Learning pipeline for the Amazon ML Challenge 2026.

Our goal was to solve **Business Entity Resolution**: given a massive list of 1.73 million businesses (Source 1), we had to search through a database of 10 million other businesses (Source 2 and Source 3) to find exactly which ones are the *same* real-world business.

---

## 📊 The "Percentage" Explained (Our Score)
If someone asks you "what percentage did we get?", they are asking about our **F0.5 Score**. 
We achieved a staggering **99.59% (0.9959)** on our training validation! 

**What does 99.59% mean?**
In Machine Learning, we measure two things:
1. **Precision:** When our model says two businesses are a match, how often is it right?
2. **Recall:** Out of all the true matches that exist, how many did we successfully find?

The competition values Precision heavily (they don't want us guessing randomly). Our 99.59% score means our AI model is incredibly accurate at deciding if two businesses are the same, barely making any mistakes!

---

## 🛠️ How It Works (The 4-Step Process)
We divided the project into four main tasks. Here is exactly what the code does in plain English:

### Step 1: Data Cleaning & Normalization (Person 2's Task)
Raw data is messy. People write "Pvt. Ltd.", "Private Limited", or make spelling mistakes. 
We wrote code to aggressively clean the data:
- Converting everything to lowercase.
- Stripping out weird punctuation and accents.
- Removing generic words like "the", "inc", and "corp".
*This ensures the AI doesn't get confused by formatting differences.*

### Step 2: Blocking & Candidate Generation (Person 3's Task)
Comparing 1.7 million businesses against 10 million businesses equals **17 Trillion** comparisons. That would take years to compute!
To fix this, we created a **"Blocking Index"**. 
Instead of checking every business, the code instantly groups businesses that share similar words or are in the same country. It narrows the 10 million choices down to the **top 50 best guesses**. This turned a 50-year task into an 8-hour task!

### Step 3: Feature Engineering 
How does a computer know if "Reliance Fresh" and "Reliance Supermart" are similar? We wrote mathematical rules (features). 
For the top 50 guesses, we calculate 23 different math scores, such as:
- **Levenshtein Distance:** How many letters do you have to change to make the words match?
- **Jaccard Similarity:** What percentage of words overlap between the two addresses?
- **TF-IDF:** Giving higher importance to rare words and ignoring common words like "store".

### Step 4: The Machine Learning Model (Person 1's Task)
We took those 23 math scores and fed them into a powerful AI called **LightGBM**. The AI learned the patterns and now acts as the final judge, scoring the probability of a match from 0% to 100%. We set a strict threshold: if the AI is over 92.5% confident, we merge the businesses!

---

## 💻 Our Tech Stack
Here are the tools we used to build this:
- **Python (3.10+):** The core programming language.
- **Pandas:** Used for loading, organizing, and manipulating the giant Excel-like tables of data.
- **NumPy:** Used for lightning-fast mathematical calculations.
- **RapidFuzz:** A specialized C++ library that calculates string differences (like Levenshtein distance) 50x faster than normal Python.
- **LightGBM:** A state-of-the-art Gradient Boosting Machine Learning algorithm created by Microsoft. It builds hundreds of "Decision Trees" to vote on whether two businesses match.
- **Scikit-Learn:** Used to calculate TF-IDF (Text Frequency) vectors and evaluate our precision/recall metrics.

---

## 🏁 Current Status
✅ **ALL TASKS COMPLETED!**
- The data is cleaned.
- The blocking index is optimized.
- The Machine Learning model is trained.
- All 1.73 million predictions have been generated.
- Unnecessary temporary files have been deleted.

**Final Output:** The entire pipeline has successfully compressed the results into a file named `AmazaanML_Team_submission.zip`, which is ready to be submitted to the competition portal!

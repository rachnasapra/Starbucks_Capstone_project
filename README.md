# Starbucks_Capstone_project
 Project Description
Starbucks Offer Dataset is one of the datasets that students can choose from to complete their capstone project for Udacity’s Data Science Nanodegree. The dataset contains simulated data that mimics customers' behavior after they received Starbucks offers. The data is collected via Starbucks rewards mobile apps and the offers were sent out once every few days to the users of the mobile app.

The goal of this project was not defined by Udacity. Thus, it is open-ended. I decided to investigate in the situation where customers used an offer without viewing it. I wanted to understand who are these customers and how we can avoid or minimize the chance of this from happening.

Project Result
The first part of the question was addressed by in-depth data engineering and data science. It turns out that all customers are equally likely to use an offer without viewing it. The demographics do not make a difference. However, the design of the offer makes a big difference, especially its promotion channel and the length of the offer.

I built a machine learning model using Decision Tree and KNN to address the second part of the question.. The model achived a 87% in both accuracy score. I also used a confusion matrix. Further details are in my blog post.

I wrote a blog post to walk through the steps I took to achieve the result. The medium blog post can be accessed here.

Main Files: Project Structure
├── data          
|   ├── portfolio.json
|   ├── profile.json
|   └──transcript.json
|
├── README.md
|
├── Starbucks_Capstone_notebook.ipynb 
|

The data folder contains the 3 datasets provided by Udacity.

The Starbucks_Capstone_notebook.ipynb is where all the analysis is.


Tech Stack
Python3 is the main language.
Numpy, Pandas, matplotlib, seaborn, and sklearn are the main packages that were used.

Jupyter Notebook is where the code is hosted.
Author and Acknowledgement
Rachna Sapra is the only author of this project.

Starbucks provided this dataset.

Udacity provided this Data Science Nanodegree Program and the access to this dataset.

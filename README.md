# Clickbait Filter

This project is an optional challenge of the mandatory course 
"Machine Learning for Natural Language Understanding" of my NLP master's degree
program at Trier University in the winter semester 22/23. 

## Task

The Task was to train a clickbait filter to classify clickbait articles 
by their headline. I could freely decide how to prepare the data and which ML 
model to use for classification.

The challenge was considered passed if the model performs better than professor's 
baseline (a simple classifier; F1 ~0.89). 

## Dataset
The data consists of two files, a text file with clickbait 
headlines and one with headlines from news sources. 
The hold out dataset is organized the same way.

I'm not allowed to publish the train and validation datasets since it is a property of
Computerlinguistik und Digital Humanities Department of the University of Trier.

## Results

I implemented an [LSTM model](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) (Raschka, 2022, p.499)
with [dropouts](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) using PyTorch library (`./utils/models.py`)
It showed a quite good result on the validation set: ___F1-Score = 96.2%___ (`./notebooks/validation_and_examples.ipynb`) which is however can be easily overcome with Transformer architecture.

## How to use
`git clone https://github.com/bourgeois-radical/clickbait-detection.git`

 _ClickbaitClassifier_ class (`./utils/showing_results.py`) 
provides a dunder-method which classifies every English sentence you give. 
 Feel free to check the classifier in the "Showing model results" section 
 (`./notebooks/validation_and_examples.ipynb`).

## References

Aggarwal, C. (2022). _Machine Learning for Text_ (2nd ed.). Springer

Raschka, S., Liu Y., & Mirjalili, V. (2022). _Machine Learning with PyTorch and Scikit-Learn_. Packt


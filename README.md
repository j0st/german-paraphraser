# German Paraphrasing Tool

## Introduction
Modified version of the code used in my bachelor's thesis titled "Der Einfluss von automatisch generierten Paraphrasen auf die NLU-Performance am Beispiel des Miele Alexa Skills" (2021).
In this project, German paraphrases are generated using different methods at the lexical and syntactic levels. The resulting paraphrases can be used for NLU evaluations for example.

## Method
Paraphrases were automatically generated in the following ways:

* __Word Substitution__
* __Pivot Translation__
* __T5 (Transformer Model)__

## Getting Started
1. Clone project
```
git clone https://github.com/j0st/german-paraphraser
```

2. Install requirements (Odenet need to be installed directly from https://github.com/hdaSprachtechnologie/odenet)
```
pip install -r requirements.txt
pip install git+https://github.com/hdaSprachtechnologie/odenet
```

3. Import `paraphraser.py` and create an instance of the paraphraser class
```
paraphraser = Paraphraser()
```

4. Generate paraphrases from your text file
```
paraphraser.generate(YOUR_LIST_OF_INPUT_SENTENCES, OUTPUT_FILE)
```

import csv
import random

from odenet import *
import stanza
import torch
from tqdm import tqdm
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from translators import translate_text

class Paraphraser:
    def __init__(self):
        stanza.download('de')
        self.nlp = stanza.Pipeline(lang="de", processors="tokenize,pos")
        self.unmasker = pipeline("fill-mask", model="bert-base-german-cased")
        self.model = T5ForConditionalGeneration.from_pretrained("seduerr/t5_base_paws_ger")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
    
    def set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def truecasing(self, doc):
        truecased = []
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.xpos in ["NN", "NE"]:
                    noun = word.text.capitalize()
                    truecased.append(noun)
                else:
                    truecased.append(word.text)
        truecased = " ".join(truecased)
        truecased = truecased.replace(" . ", ".\n\n")
        return truecased

    def preprocessing(self, sentences):
        utterances_string = '\n\n'.join(sentences)
        doc = self.nlp(utterances_string)
        truecased = self.truecasing(doc)
        doc = self.nlp(truecased)
        return doc

    def word_substitution(self, doc):
        unused_tokens = [f"[unused_punctuation{i}]" for i in range(29)]
        synonym_sub = ["ADJA", "ADV", "NN", "NE", "PTKANT", "VVINF", "VVIZU", "ITJ"]
        paraphrases = []
        for sentence in tqdm(doc.sentences):
            for word in sentence.words:
                if word.xpos not in synonym_sub:
                    masked = sentence.text.replace(word.parent.text, "[MASK]", 1)
                    if "[MASK]" in masked:
                        for seq in self.unmasker(masked):
                            if seq["token_str"] not in unused_tokens and seq["score"] > 0.01:
                                unmasked = seq["sequence"]
                                paraphrases.extend(unmasked.split())
                else:
                    if synonyms_word(word.text) is None or not synonyms_word(word.text)[0]:
                        masked = sentence.text.replace(word.parent.text, "[MASK]", 1)
                        if "[MASK]" in masked:
                            for seq in self.unmasker(masked):
                                if seq["token_str"] not in unused_tokens and seq["score"] > 0.6:
                                    unmasked = seq["sequence"]
                                    paraphrases.extend(unmasked.split())
                    else:
                        synonym = synonyms_word(word.parent.text)[0][choice(range(len(synonyms_word(word.text)[0])))]
                        synonym_sent = sentence.text.replace(word.text, synonym, 1)
                        paraphrases.extend(synonym_sent.split())
        paraphrases = " ".join(paraphrases)
        paraphrases = paraphrases.split(". ")
        print(str(len(paraphrases)) + " paraphrases were generated through substitution.")
        return list(set(paraphrases))

    def pivot_translation(self, doc):
        lang = ["en", "pt", "es", "pl", "ht", "nl", "it", "ja"]
        paraphrases = []
        for sentence in tqdm(doc.sentences):
            rndm = randint(1,2)
            if rndm == 1:
                pivot_lang = choice(lang)
                pivot = translate_text(sentence.text, translator="bing", from_language="de", to_language=pivot_lang)
                paraphrase = translate_text(pivot, translator="bing", from_language=pivot_lang, to_language="de")
                if paraphrase != sentence.text:
                    paraphrases.append(paraphrase)
                else:
                    pivot_lang_1 = choice(lang)
                    pivot_lang_2 = choice([x for x in lang if x != pivot_lang_1])
                    pivot = translate_text(sentence.text, translator="bing", from_language="de", to_language=pivot_lang_1)
                    paraphrase = translate_text(pivot, translator="bing", from_language=pivot_lang_1, to_language=pivot_lang_2)
                    paraphrase = translate_text(pivot, translator="bing", from_language=pivot_lang_2, to_language="de")
                    paraphrases.append(paraphrase)
            else:
                pivot_lang_1 = choice(lang)
                pivot_lang_2 = choice([x for x in lang if x != pivot_lang_1])
                pivot = translate_text(sentence.text, translator="bing", from_language="de", to_language=pivot_lang_1)
                paraphrase = translate_text(pivot, translator="bing", from_language=pivot_lang_1, to_language=pivot_lang_2)
                paraphrase = translate_text(pivot, translator="bing", from_language=pivot_lang_2, to_language="de")
                paraphrases.append(paraphrase)
        print(str(len(paraphrases)) + " paraphrases were generated through pivot translation.")
        return list(set(paraphrases))

    def t5_paraphrase(self, doc):
        paraphrases = []
        self.set_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)
        for sentence in tqdm(doc.sentences):
            sentence = sentence.text
            text = "paraphrase: " + sentence + " </s>"
            max_len = 256
            encoding = self.tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
            input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
            beam_outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_masks,
                do_sample=True,
                max_length=256,
                top_k=120,
                top_p=0.98,
                early_stopping=True,
                num_return_sequences=3
            )
            for i, line in enumerate(beam_outputs):
                paraphrase = self.tokenizer.decode(line, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                paraphrases.append(paraphrase)
        print(str(len(paraphrases)) + " paraphrases were generated through T5.")
        return list(set(paraphrases))

    def paraphrase(self, sentences):
        doc = self.preprocessing(sentences)
        paraphrases_sub = self.word_substitution(doc)
        paraphrases_pivot = self.pivot_translation(doc)
        paraphrases_t5 = self.t5_paraphrase(doc)
        paraphrases = paraphrases_sub + paraphrases_pivot + paraphrases_t5
        paraphrases = list(set(paraphrases))
        return paraphrases

    def save_paraphrases(self, paraphrases, output_file):
        with open(output_file, "w", encoding="utf-8") as file:
            file.write("\n".join(paraphrases))
        print('Paraphrases saved in the text file:', output_file)

    def process_sample_utterances(self, sentences, output_file):
        paraphrases = self.paraphrase(sentences)
        self.save_paraphrases(paraphrases, output_file)

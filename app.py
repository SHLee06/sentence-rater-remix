#importing the libraries
# 
import streamlit as st
from classifier.predictor import SentenceLevelPredictor

clr = SentenceLevelPredictor.from_path('models/sent_level_bert_ce_6levels.tar.gz', 'sentence_level_predictor')
labels = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}

def probs2level(probs):
    idx = probs.index(max(probs))
    level = labels[idx]
    return level


# good_sentence = 'Town meetings should be held to discuss issues.'
ex_sentence = 'In this article, I will first provide an overview of the system of accreditation and then discuss issues of accreditation as they apply to these contemporary American educational programs in Japan.'

# Designing the interface
st.title("Sentence Rating Demo")
# For newline
('\n')

form = st.form(key='my-form')
sent = form.text_area('Test it out!!', value=ex_sentence, help = ('Enter your sentence here'))
submit = form.form_submit_button('Enter')

if submit:
    sent_probs = clr.predict_probs({'text': sent})
    sent_probs = [round(p,3) for p in sent_probs]
    st.write(f'Probs: {sent_probs}')
    # for i, p in enumerate(sent_probs):
    #     st.write(f'{labels[i]}: {round(p, 3)}')
    st.write(f'Level: {probs2level(sent_probs)}')



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
og_sent = form.text_area('Test it out!!', value=ex_sentence, help = ('Enter your sentence here'))
submit = form.form_submit_button('Enter')

if submit:
    mix_sent = " ".join(sorted(og_sent.split(), key=lambda v: (v.upper(), v[0].islower())))

    sent_probs = clr.predict_probs({'text': og_sent})
    sent_probs = [round(p,3) for p in sent_probs]
    st.write(f'Original sentence: {og_sent}')
    st.write(f'Probs: {sent_probs}')
    # for i, p in enumerate(sent_probs):
    #     st.write(f'{labels[i]}: {round(p, 3)}')
    st.write(f'Level: {probs2level(sent_probs)}')

    sent_probs = clr.predict_probs({'text': mix_sent})
    sent_probs = [round(p,3) for p in sent_probs]
    st.write(f'Remixed sentence: {mix_sent}')
    st.write(f'Probs: {sent_probs}')
    # for i, p in enumerate(sent_probs):
    #     st.write(f'{labels[i]}: {round(p, 3)}')
    st.write(f'Level: {probs2level(sent_probs)}')



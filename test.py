import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Attention, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

st.set_page_config(page_title="English-Hindi Translator", layout="centered")

# -------------------------------------------------------------------
# Set up paths based on your local directory
# -------------------------------------------------------------------
BASE_DIR = r"C:\Users\shaha\Downloads\Lab6\model\Translator_Project"

FILES = {
    'eng_tok': os.path.join(BASE_DIR, 'eng_tokenizer.pkl'),
    'hin_tok': os.path.join(BASE_DIR, 'hin_tokenizer.pkl'),
    'config': os.path.join(BASE_DIR, 'model_config.pkl'),
    'weights': os.path.join(BASE_DIR, 'model_weights.weights.h5')
}

# Verify files exist before trying to load them
missing_files = [path for path in FILES.values() if not os.path.exists(path)]
if missing_files:
    st.error("⚠️ Missing required model files in your directory:")
    for f in missing_files:
        st.write(f"- `{f}`")
    st.info("Please ensure you have downloaded the files from Colab and placed them in the correct folder.")
    st.stop()

# -------------------------------------------------------------------
# Load Files and Build Inference Models (Cached for speed)
# -------------------------------------------------------------------
@st.cache_resource
def load_translator_system():
    # 1. Load Tokenizers and Config
    with open(FILES['eng_tok'], 'rb') as f:
        eng_tok = pickle.load(f)
    with open(FILES['hin_tok'], 'rb') as f:
        hin_tok = pickle.load(f)
    with open(FILES['config'], 'rb') as f:
        config = pickle.load(f)

    eng_vocab_size = config['eng_vocab_size']
    hin_vocab_size = config['hin_vocab_size']
    max_eng_len = config['max_eng_len']
    latent_dim = config['latent_dim']

    # 2. Rebuild the exact Architecture
    encoder_inputs = Input(shape=(None,), name="encoder_input")
    enc_emb = Embedding(eng_vocab_size, latent_dim, name="encoder_embedding")(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="encoder_lstm")
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,), name="decoder_input")
    dec_emb_layer = Embedding(hin_vocab_size, latent_dim, name="decoder_embedding")
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

    attention_layer = Attention(name='attention_layer')
    attention_output = attention_layer([decoder_outputs, encoder_outputs])

    concat_layer = Concatenate(axis=-1, name='concat_layer')
    decoder_combined_context = concat_layer([decoder_outputs, attention_output])

    decoder_dense = Dense(hin_vocab_size, activation='softmax', name="decoder_dense")
    decoder_outputs_final = decoder_dense(decoder_combined_context)

    # 3. Compile the base model and Load Weights
    full_model = Model([encoder_inputs, decoder_inputs], decoder_outputs_final)
    full_model.load_weights(FILES['weights'])

    # 4. Construct Inference Models using the loaded layers
    # Encoder Inference Model
    encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

    # Decoder Inference Model
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_hidden_state_input = Input(shape=(max_eng_len, latent_dim))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    dec_emb2 = dec_emb_layer(decoder_inputs)
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)

    attention_output2 = attention_layer([decoder_outputs2, decoder_hidden_state_input])
    decoder_combined_context2 = concat_layer([decoder_outputs2, attention_output2])
    decoder_outputs2 = decoder_dense(decoder_combined_context2)

    decoder_model = Model(
        [decoder_inputs, decoder_hidden_state_input] + decoder_states_inputs,
        [decoder_outputs2, state_h2, state_c2]
    )

    return eng_tok, hin_tok, max_eng_len, encoder_model, decoder_model

# Initialize models
with st.spinner("Loading AI model into memory..."):
    eng_tok, hin_tok, max_eng_len, enc_model, dec_model = load_translator_system()

# -------------------------------------------------------------------
# Translation Function
# -------------------------------------------------------------------
def translate(input_text):
    seq = eng_tok.texts_to_sequences([input_text])
    
    # Check if the word is in our vocabulary
    if not seq or not seq[0]:
         return "Error: Words not recognized by the model's vocabulary."
         
    padded_seq = pad_sequences(seq, maxlen=max_eng_len, padding='post')
    
    enc_outs, h, c = enc_model.predict(padded_seq, verbose=0)

    start_token = hin_tok.word_index.get('start', hin_tok.word_index.get('<start>', 1))
    end_token = hin_tok.word_index.get('end', hin_tok.word_index.get('<end>', -1))

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = start_token

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = dec_model.predict([target_seq, enc_outs, h, c], verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = hin_tok.index_word.get(sampled_token_index, '')

        # Stop if end token is hit, or sentence gets suspiciously long
        if sampled_token_index == end_token or sampled_word in ['end', '<end>'] or len(decoded_sentence.split()) > 50:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

    return decoded_sentence.strip()

# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.title("🌐 English to Hindi Translator")
st.write("Powered by an LSTM Encoder-Decoder with Attention")

user_input = st.text_input("Enter English text:", placeholder="Type a sentence here...")

if st.button("Translate", type="primary"):
    if user_input:
        with st.spinner("Translating..."):
            translation = translate(user_input.lower())
            st.success(f"**Translation:** {translation}")
    else:
        st.warning("Please enter some text to translate.")
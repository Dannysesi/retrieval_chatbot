import streamlit as st
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import time
from datetime import date

st.set_page_config(
     page_title='Infectous disease ChatBot',
    #  layout="wide",
     initial_sidebar_state="expanded",
)


with open("intents.json", 'r', encoding='utf-8') as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
	for pattern in intent["patterns"]:
		wrds = nltk.word_tokenize(pattern)
		words.extend(wrds)
		docs_x.append(wrds)
		docs_y.append(intent["tag"])

	if intent["tag"] not in labels:
		labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))


labels = sorted(labels)


training = []
output = []

out_empty = [0 for _ in range(len(labels))]
for x, doc in enumerate(docs_x):
	bag = []

	wrds = [stemmer.stem(w) for w in doc]

	for w in words:
		if w in wrds:
			bag.append(1)
		else:
			bag.append(0)
	output_row = out_empty[:]
	output_row[labels.index(docs_y[x])] = 1

	training.append(bag)
	output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

with open('data.pickle','wb') as f:
	pickle.dump((words, labels, training, output), f)


tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
	model.load("model.tflearn")
except:
	model = tflearn.DNN(net)
	model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
	model.save("model.tflearn")

def bag_of_words(s,words):
	bag = [0 for _ in range(len(words))]


	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1

	return numpy.array(bag)

from requests import Response


unique_key = 0
def chat():
	unique_key = 0
	st.title("Multi-Lingual Healthcare Chatbot for Infectious Disease Diagnosis🤖")
	st.write('---')

	if 'user_info' not in st.session_state:
		st.session_state['user_info'] = None

	if st.session_state['user_info'] is None:
		# Display the form to collect user details
		with st.form("user_details_form"):
			name = st.text_input("Name")
			age = st.number_input("Age", min_value=16, max_value=120, step=1)
			gender = st.selectbox("Gender", ["Male", "Female", "Other"])

			# Submit button
			submit = st.form_submit_button("Submit")

			if submit:
				# Store user details in session state
				st.session_state['user_info'] = {
					"name": name,
					"age": age,
					"gender": gender
				}
				st.rerun()

	else:
		st.write(f"<span style='font-size: 24px;'>Welcome, {st.session_state['user_info']['name']}!🎉</span>", unsafe_allow_html=True)


		with st.form("chat_input", clear_on_submit=True):
			a, b = st.columns([4, 1])
			inp = a.text_input("User_Input: ", key=unique_key, placeholder='What are your symptoms?💭', label_visibility="collapsed")
				# if inp.lower() == "quit":
			if inp.strip():
				pass
			results = model.predict([bag_of_words(inp,words)])[0]
			results_index = numpy.argmax(results)
			tag = labels[results_index]
			dail = '+2349025629246'
			submitted = b.form_submit_button("Send", use_container_width=True)
			# st.session_state.messages.append({"role": "user", "content": inp})

				
			if submitted:
				if results[results_index] > 0.5:
					for tg in data["intents"]:
						if tg['tag'] == tag:
							responses = tg['responses']
					with st.chat_message("assistant"):
						resp = random.choice(responses)
						st.markdown(resp + "▌")
						st.markdown("\n")
						if inp.lower() in ['hi', 'bawo ni']:
							st.write('---')
						else:
							st.markdown("""
							Please visit the nearest hospital to see a doctor and get tested soonest.
							Also, here are a few self-care practices you can follow pending when you see a doctor:

							1.) Isolate yourself  
							2.) Drink lots of water  
							3.) Get plenty of rest  
							4.) Eat a balanced diet  
							5.) Take painkillers and fever relievers  
							6.) Avoid alcohol and smoking  
							7.) Keep a healthy hygiene  
							8.) See a doctor and get tested as soon as possible.
							""")
				else:
					z = "Sorry I didn't get that, please describe your symptoms in more detail"
					st.warning("Sorry I didn't get that, please describe your symptoms in more detail")
					st.session_state.messages.append({"role": "assistant", "content": z})
			st.write('---')
			st.markdown('For more information or urgent assistance, contact us at: +2349025629246')
			st.info('This chatbot provides preliminary guidance and is not a substitute for professional medical advice. Always seek professional help for serious conditions.', icon="ℹ️")
			st.write('---')
			# st.subheader('Chat History')
			# st.write('---')
			# # Display chat messages from history on app rerun
			# for message in st.session_state.messages:
			# 	with st.chat_message(message["role"]):
			# 		st.markdown(message["content"])


chat()	


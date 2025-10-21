import os
import json
import pathway as pw
import numpy as np
import pickle as pk
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sentence_transformers import SentenceTransformer


## User Query Schema

class UserQuerySchema(pw.Schema):
    customer_id: str
    message: str


MAX_LEN = 20

# loading the static models
try:
    intent_model = tf.keras.models.load_model("support_model.keras")

    with open("tokenizer.pickle", "rb") as handle:
        intent_tokenizer = pk.load(handle)

    with open("encoder.pickle", "rb") as handle:
        intent_encoder = pk.load(handle)

except Exception as e:
    print(f"Error in loading models {e}")
    exit(1)

embedding_model_name = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(embedding_model_name)


#=========== HELPERS =================

def predict_intent(raw_text: str) -> str:
    try:
        sequences = intent_tokenizer.texts_to_sequences([raw_text])
        padded_sequence = pad_sequences(
            sequences, maxlen=MAX_LEN, padding="post", truncating="post"
        )

        prediction = intent_model.predict(padded_sequence, verbose=0)
        predicted_idx = np.argmax(prediction, axis=1)

        predicted_intent = intent_encoder.inverse_transform(predicted_idx)[0]

        return predicted_intent

    except Exception as e:
        print(f"Error during prediction {e}")
        return "ERROR_PREDICTING_INTENT"


#Used ANY(pw.apply) -> INT casting
def get_one(_: str) -> int:
    return 1

#Used for ANY(pw.apply) -> STR casting
def add_source(text: str, source_name: str) -> str:
    return source_name

def embed_text(text):
    embedding = embedder.encode([text])[0]
    return embedding.tolist()


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))



# Sentence pair similarities
def calc_similarity(query_vec, doc_vec):
    return cosine_similarity(query_vec, doc_vec)


def process_query_once(customer_id, message):

    intent = predict_intent(message)
    query_vector = embed_text(message)

    print(f"\n--- Processing query from {customer_id}: '{message}' ---")
    print(f"Predicted intent: {intent}")
    
    return (intent, query_vector)

#=================================================




#============ RAG PIPELINE ======================

# Load documents

faq_data = pw.io.fs.read(
    "./policy_docs/faq.md",
    mode="streaming",
    format="plaintext",
)

status_data = pw.io.fs.read(
    "./status.txt",
    mode="streaming",
    format="plaintext",
)

history_data = pw.io.fs.read(
    "./chat_history.jsonl",
    mode="streaming",
    format="plaintext",
)





## Using the ANY -> STR helper during Source Assignment 
faq_data = faq_data.select(
    doc_content=pw.this.data, 
    source=pw.apply(add_source, pw.this.data, "faq")
)
status_data = status_data.select(
    doc_content=pw.this.data, 
    source=pw.apply(add_source, pw.this.data, "status")
)
history_data = history_data.select(
    doc_content=pw.this.data, 
    source=pw.apply(add_source, pw.this.data, "history")
)





# Concatenate all documents
all_documents = pw.Table.concat_reindex(faq_data, status_data, history_data)




## Embed documents
embedded_docs = all_documents.select(
    doc_content=pw.this.doc_content,
    source=pw.this.source,
    vector=pw.apply(embed_text, pw.this.doc_content),
)





















#=============== QUERY PROCESSING ==================

query_stream = pw.io.jsonlines.read(
    "./queries.jsonl",
    schema=UserQuerySchema,
    mode="streaming",
    autocommit_duration_ms=1000,
)


print("Analysing queries written in queries.jsonl")



# To exploit the pathways efficient data stream we will first store the tuple(intent, query_vector) in same var
processed_queries_temp = query_stream.select(
    customer_id=pw.this.customer_id,
    message=pw.this.message,
    processed=pw.apply(process_query_once, pw.this.customer_id, pw.this.message),
)


# we are using predict_intent and embed_text in same function to increase redundancy so seperate them here(process_query_once) 2 times
processed_queries = processed_queries_temp.select(
    customer_id=pw.this.customer_id,
    message=pw.this.message,
    intent=pw.apply(lambda x: x[0], pw.this.processed),
    query_vector=pw.apply(lambda x: x[1], pw.this.processed),
)


# Create a cross join by adding a dummy column
processed_queries_with_key = processed_queries.select(
    customer_id=pw.this.customer_id,
    message=pw.this.message,
    intent=pw.this.intent,
    query_vector=pw.this.query_vector,
    join_key=pw.apply(get_one, pw.this.customer_id)
)

embedded_docs_with_key = embedded_docs.select(
    doc_content=pw.this.doc_content,
    source=pw.this.source,
    vector=pw.this.vector,
    join_key=pw.apply(get_one, pw.this.doc_content)
)

# Join queries with all documents (cross join using dummy key)
query_doc_pairs = processed_queries_with_key.join(
    embedded_docs_with_key,
    pw.left.join_key == pw.right.join_key,
).select(
    customer_id=pw.left.customer_id,
    message=pw.left.message,
    intent=pw.left.intent,
    query_vector=pw.left.query_vector,
    doc_content=pw.right.doc_content,
    source=pw.right.source,
    doc_vector=pw.right.vector,
)








pairs_with_sim = query_doc_pairs.select(
    customer_id=pw.this.customer_id,
    message=pw.this.message,
    intent=pw.this.intent,
    query_vector=pw.this.query_vector,
    doc_content=pw.this.doc_content,
    source=pw.this.source,
    doc_vector=pw.this.doc_vector,
    similarity=pw.apply(calc_similarity, pw.this.query_vector, pw.this.doc_vector)
)


# Get best match per source type - using cast to ensure type compatibility
faq_pairs = pairs_with_sim.filter(pw.this.source == pw.cast(str, "faq"))
status_pairs = pairs_with_sim.filter(pw.this.source == pw.cast(str, "status"))

best_faq = faq_pairs.groupby(
    pw.this.customer_id, pw.this.message
).reduce(
    customer_id=pw.this.customer_id,
    message=pw.this.message,
    intent=pw.reducers.any(pw.this.intent),
    faq_content=pw.reducers.argmax(pw.this.similarity, pw.this.doc_content),
)

best_status = status_pairs.groupby(
    pw.this.customer_id, pw.this.message
).reduce(
    customer_id=pw.this.customer_id,
    message=pw.this.message,
    status_content=pw.reducers.argmax(pw.this.similarity, pw.this.doc_content),
)


# Join back together
results = best_faq.join(
    best_status,
    pw.left.customer_id == pw.right.customer_id,
    pw.left.message == pw.right.message,
    how=pw.JoinMode.LEFT,
).select(
    customer_id=pw.left.customer_id,
    message=pw.left.message,
    intent=pw.left.intent,
    faq_content=pw.left.faq_content,
    status_content=pw.right.status_content,
)










# ================= SUGAR COATING =======================
## Helpers for sugar coating the (intent, faq, status) vector

def sugarcoat_with_llm(raw_response, intent):
   
   #### We can sugar coat the intend using any light weight LLM for now I simulated it using Templates
    templates = {
        "ORDER_STATUS": f"Hi there!  {raw_response} Let me know if you need anything else!",
        "PASSWORD_RESET": f"No worries! {raw_response} You'll be back in your account in no time. ",
        "LOGIN_ISSUE": f"I understand how frustrating that can be. {raw_response} Feel free to reach out if this doesn't resolve it!",
        "REFUND_REQUEST": f"I'm sorry to hear that. {raw_response} We're here to help make this right",
        "TECHNICAL_ISSUE": f"Thanks for reporting this! {raw_response} Our team is on it.",
    }
    return templates.get(intent, f"Raw Response: {raw_response}")


def generate_reply(customer_id, message, intent, faq_content, status_content):
    raw_reply = "Sorry, I couldn't find an answer."
    source = "default"
    
    # Check if status.txt is non empty and relevant (cruicial for the status.txt to override the RAG Live Idx Response)
    if status_content is not None and status_content.strip().lower() != "all systems normal.":
        status_lower = status_content.lower()
        is_relevant = False
        
        if ("login" in status_lower or "server" in status_lower or "outage" in status_lower) and \
           (intent in ["LOGIN_ISSUE", "TECHNICAL_ISSUE"]):
            is_relevant = True ## Override
        elif ("promo" in status_lower or "sale" in status_lower) and (intent == "ORDER_STATUS"):
            is_relevant = True ## Override
        
        if is_relevant:
            raw_reply = f"LIVE STATUS UPDATE: {status_content}"
            source = "status"
            print(f"Agent [{customer_id}]: Using status alert for intent '{intent}'")
        elif faq_content is not None:
            raw_reply = faq_content
            source = "rag"
            print(f"Agent [{customer_id}]: Using RAG answer for intent '{intent}'")
            
    elif faq_content is not None:
        raw_reply = faq_content
        source = "rag"
        print(f"Agent [{customer_id}]: Using RAG answer for intent '{intent}'")

    # Sugarcoat with LLM
    final_reply = sugarcoat_with_llm(raw_reply, intent)
    print(f"REPLY [{customer_id}] ({source}): {final_reply[:80]}...")
    return final_reply
#==============================================================






#================== GENERATE REPIES ==================
final_replies = results.select(
    customer_id=pw.this.customer_id,
    message=pw.this.message,
    reply=pw.apply(generate_reply, pw.this.customer_id, pw.this.message, pw.this.intent, pw.this.faq_content, pw.this.status_content),
)
#======================================================



#========== DEBUGS AND WRITE THE REPLIES INTO chat_history.jsonl================
print("\nWriting outputs to: ./chat_history.jsonl")
if os.path.exists("./chat_history.jsonl"):
    os.remove("./chat_history.jsonl")

print("\nWriting outputs to: ./chat_history.jsonl")
pw.io.jsonlines.write(final_replies, "./chat_history.jsonl")




pw.run()


import numpy as np
import joblib
from joblib import dump,load

import pandas as pd
import os
from flask import Flask,request
import pika
import json


app = Flask(__name__)
vectorizer = load('vectorizer.joblib') 
model = load("model.joblib")

rabbitmq_host = '192.168.1.42'
rabbitmq_port = 5672
rabbitmq_queue = 'text_queue'


def connect_to_rabbitmq():
    #create connection to rabbitmq
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_host,
        port=rabbitmq_port))
    return connection
def consume_messages():
    #create connection  
    connection=connect_to_rabbitmq()
    # Create a channel
    channel = connection.channel()
    # Declare the queue
    channel.queue_declare(queue=rabbitmq_queue)
     # Set up the consumer and start consuming messages
    channel.basic_consume(queue=rabbitmq_queue, on_message_callback=callback, auto_ack=True)
    channel.start_consuming()

def callback(ch, method, properties, body):
    # Process the received message
    # print(f"Received message: {body}")
    text=json.loads(body)
    predict(text)


def predict(texts):
    """Predict texts array"""
    

    vectorizer = load('vectorizer.joblib') 
    model = load("model.joblib")

    l=len(texts)
    print(l)
    texts=list(texts.values())
    print(texts)
    off=[]  
    for i in range (0,l):
        val=model.predict(vectorizer.transform([texts[i]]))
        off.append(int(val))
    # print(off)
    x=sum(off)
    # print(x)
    score=sum(off)/l
    score=score*100
    connection = connect_to_rabbitmq()

    # Create a channel
    channel = connection.channel()

    #Declare the queue
    channel.queue_declare(queue='score_sent', durable=True)


    channel.basic_publish(
        exchange='',
        routing_key='score_sent',
        body=str(score),
        properties=pika.BasicProperties(
            delivery_mode=2,  # make the message persistent
        )
    )

    print(f"Score {score} sent to RabbitMQ")

    # Close the connection
    connection.close()
    








if __name__ == '__main__':
    # Run the consumer in a separate thread
    import threading
    consumer_thread = threading.Thread(target=consume_messages)
    consumer_thread.start()

    # Run the Flask app
    app.run(debug=True)





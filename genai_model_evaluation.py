# Databricks notebook source
import openai
from openai import OpenAI

import pandas as pd
import os
import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using mlflow evaluation metrics 

# COMMAND ----------

eval_df = pd.DataFrame(
    {
        "inputs": [
            "How does useEffect() work?",
            "What does the static keyword in a function mean?",
            "What does the 'finally' block in Python do?",
            "What is the difference between multiprocessing and multithreading?",
        ],
        "ground_truth": [
            "The useEffect() hook tells React that your component needs to do something after render. React will remember the function you passed (we’ll refer to it as our “effect”), and call it later after performing the DOM updates.",
            "Static members belongs to the class, rather than a specific instance. This means that only one instance of a static member exists, even if you create multiple objects of the class, or if you don't create any. It will be shared by all objects.",
            "'Finally' defines a block of code to run when the try... except...else block is final. The finally block will be executed no matter if the try block raises an error or not.",
            "Multithreading refers to the ability of a processor to execute multiple threads concurrently, where each thread runs a process. Whereas multiprocessing refers to the ability of a system to run multiple processors in parallel, where each processor can run one or more threads.",
        ],
    }
)

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.tracking.fluent._is_tracking_enabled = False

# COMMAND ----------

system_prompt = "Answer the following question in two sentences"

client = OpenAI(
    api_key=dbutils.secrets.get("shj_scope", "rag_sp_token"),
    base_url="https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints"
)
def get_llm_response(input_question: str):
  response = client.chat.completions.create(
    model="databricks-meta-llama-3-1-70b-instruct",
    messages=[
      {
        "role": "system",
        "content": system_prompt
      },
      {
        "role": "user",
        "content": input_question
      }
    ],
    temperature=0.1,
    max_tokens=128)
  return response.choices[0].message.content

eval_df['response'] = eval_df['inputs'].apply(get_llm_response)

# COMMAND ----------

display(eval_df)

# COMMAND ----------

answer_similarity_metric = mlflow.metrics.genai.answer_similarity(model="endpoints:/databricks-meta-llama-3-70b-instruct")

with mlflow.start_run() as run:    
    results = mlflow.evaluate(
        data=eval_df,
        predictions="response",
        targets="ground_truth",  # specify which column corresponds to the expected output
        model_type="question-answering",  # model type indicates which metrics are relevant for this task
        evaluators="default",
        extra_metrics=[answer_similarity_metric],
    )

results.metrics

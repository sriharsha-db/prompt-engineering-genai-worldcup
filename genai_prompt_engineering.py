# Databricks notebook source
# MAGIC %md
# MAGIC ## Using OpenAI python client example

# COMMAND ----------

import os
import openai
from openai import OpenAI

# COMMAND ----------

client = OpenAI(
    api_key=dbutils.secrets.get("shj_scope", "rag_sp_token"),
    base_url="https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints"
)

SYSTEM_PROMPT = """You are a classifier that will tag input text with a score based upon supplied criteria. 

Please review the text and determine if it meets the criteria for tagging.

Here is the criteria for tagging:
(1) insults
(2) threats
(3) highly negative comments
(4) any Personally Identifiable Information
"""

HUMAN_PROMPT = "Here is the text: {input}"

# COMMAND ----------

input_q = "hey!!! are you mad or what!!"

response = client.chat.completions.create(
    model="databricks-meta-llama-3-1-70b-instruct",
    messages=[
      {
        "role": "system",
        "content": SYSTEM_PROMPT
      },
      {
        "role": "user",
        "content": HUMAN_PROMPT.format(input=input_q)
      }
    ],
    temperature=0.1,
    max_tokens=128
)

print(response.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using Langchain LCEL example

# COMMAND ----------

# MAGIC %pip install langchainhub

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from langchain import hub
from operator import itemgetter
from langchain_community.chat_models import ChatDatabricks
from langchain_core.output_parsers import StrOutputParser

# COMMAND ----------

prompt = hub.pull("rlm/tagging")

chat_model = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct", 
                            temperature=0.1,
                            max_tokens=128)

chain = itemgetter("input") | prompt | chat_model | StrOutputParser()

# COMMAND ----------

import mlflow

with mlflow.start_run() as run:
  print(chain.invoke({"input":"hey!!! are you mad or what!!"}))

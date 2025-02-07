# Databricks notebook source
# MAGIC %pip install -U mlflow langchain_community
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import openai
from openai import OpenAI

import pandas as pd
import os
import mlflow

from operator import itemgetter
from langchain_community.chat_models import ChatDatabricks
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register a model to MLflow

# COMMAND ----------

with open('prompt_template.txt', 'r') as file:
    template_string = file.read()

prompt_template = PromptTemplate.from_template(
    template_string,
    template_format="f-string",
)

chat_model = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", 
                            temperature=0.1,
                            max_tokens=128)

# COMMAND ----------

chain = prompt_template | chat_model | StrOutputParser()

input_example = {"user_question": "What is MLflow?",
                "documentation": "MLflow is an open source platform for managing the end-to-end machine learning lifecycle."}
chain.invoke(input_example)

# COMMAND ----------

mlflow.end_run()
mlflow.start_run()

# COMMAND ----------

model_info = mlflow.langchain.log_model(
    chain,
    artifact_path="model",
    input_example=input_example
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using mlflow evaluation metrics 

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

eval_df = pd.DataFrame(
    {
        "user_question": [
            "How does useEffect() work?",
            "What does the static keyword in a function mean?",
            "What does the 'finally' block in Python do?",
            "What is the difference between multiprocessing and multithreading?",
        ],
        "documentation":[
            "useEffect is a React Hook that lets you synchronize a component with an external system.",
            "static is a reserved word in many programming languages to modify a declaration. The effect of the keyword varies depending on the details of the specific programming language, most commonly used to modify the lifetime (as a static variable) and visibility (depending on linkage), or to specify a class member instead of an instance member in classes.",
            "In Python, the try-finally block is used to ensure that certain code executes, regardless of whether an exception is raised or not. Unlike the try-except block, which handles exceptions, the try-finally block focuses on cleanup operations that must occur, ensuring resources are properly released and critical tasks are completed.",
            "Both multiprocessing and multithreading are used in computer operating systems to increase its computing power. The fundamental difference between multiprocessing and multithreading is that multiprocessing makes the use of two or more CPUs to increase the computing power of the system, while multithreading creates multiple threads of a process to be executed in a parallel fashion to increase the throughput of the system."
        ],
        "ground_truth": [
            "The useEffect() hook tells React that your component needs to do something after render. React will remember the function you passed (we’ll refer to it as our “effect”), and call it later after performing the DOM updates.",
            "Static members belongs to the class, rather than a specific instance. This means that only one instance of a static member exists, even if you create multiple objects of the class, or if you don't create any. It will be shared by all objects.",
            "'Finally' defines a block of code to run when the try... except...else block is final. The finally block will be executed no matter if the try block raises an error or not.",
            "Multithreading refers to the ability of a processor to execute multiple threads concurrently, where each thread runs a process. Whereas multiprocessing refers to the ability of a system to run multiple processors in parallel, where each processor can run one or more threads.",
        ],
    }
)
display(eval_df)

# COMMAND ----------

answer_similarity_metric = mlflow.metrics.genai.answer_similarity(model="endpoints:/databricks-meta-llama-3-3-70b-instruct")

results = mlflow.evaluate(
    data=eval_df,
    model=model_info.model_uri,
    targets="ground_truth",  # specify which column corresponds to the expected output
    model_type="question-answering",  # model type indicates which metrics are relevant for this task
    evaluators="default",
    extra_metrics=[answer_similarity_metric],
    evaluator_config={'col_mapping': {'inputs': 'user_question'}},
)

results.metrics

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

mlflow.register_model(model_info.model_uri, name="uc_demos_sriharsha_jana.test_db.prompt_engg_model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation when having Data Only

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

answer_similarity_metric = mlflow.metrics.genai.answer_similarity(model="endpoints:/databricks-meta-llama-3-3-70b-instruct")

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

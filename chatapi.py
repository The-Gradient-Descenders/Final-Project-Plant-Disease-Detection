import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Give me 5 ideas on how to treat apple scab disease."}
  ]
)    

print(completion.choices[0].message.content)
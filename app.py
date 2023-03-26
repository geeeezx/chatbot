import os
import openai
from flask import Flask, request, jsonify
import tiktoken
from functools import lru_cache

app = Flask(__name__)

# 设置你的 OpenAI API 密钥
openai.api_key = os.getenv("OPENAI_API_KEY", "your_api_key_here")

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
  """Returns the number of tokens used by a list of messages."""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
  if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
      num_tokens = 0
      for message in messages:
          num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
          for key, value in message.items():
              num_tokens += len(encoding.encode(value))
              if key == "name":  # if there's a name, the role is omitted
                  num_tokens += -1  # role is always required and always 1 token
      num_tokens += 2  # every reply is primed with <im_start>assistant
      return num_tokens
  else:
      raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

def query_gpt35(messages, max_tokens, temperature):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    content = response['choices'][0]['message']['content'].strip()
    finish_reason = response['choices'][0]['finish_reason']
    total_tokens = response['usage']['total_tokens']
    return content, finish_reason, total_tokens

def query_gpt35_cached(messages, max_tokens, temperature):
    query_gpt35(messages, max_tokens, temperature)

@app.route('/assistant', methods=['POST'])
def assistant():
    data = request.get_json()
    user_input = data.get('input')

    if not user_input:
        return jsonify({"error": "Invalid input"}), 400

    max_tokens = data.get('max_tokens', 400)
    temperature = data.get('temperature', 0.5)
    messages = [
        {"role": "system", "content": "你是一个湾流生态公司水草养殖与环保工程智能助手"},
        {"role": "assistant",
         "content": "湾流生态是世界首家内容输出型人工智能环境难题解决方案商。对生态环境治理领域“设计难度高、建设成本高、运营管理难”的难题提供全品类系统化自研产品和解决方案。湾流生态，蓝色星球的修复师。"},
        {"role": "user", "content": user_input}
    ]

    prompt_tokens = num_tokens_from_messages(messages)

    if prompt_tokens > 4096:  # 确保回复不会超过模型的最大 token 限制
        return jsonify({"error": "Conversation is too long to process."}), 400
    elif prompt_tokens == 0:
        return jsonify({"error": "Invalid input"}), 400

    response_content, finish_reason, total_tokens = query_gpt35(messages, max_tokens, temperature)
    return jsonify({
        "response": response_content,
        "finish_reason": finish_reason,
        "total_tokens": total_tokens
    })

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8000)

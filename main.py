from flask_ngrok import run_with_ngrok
from flask import Flask, request, jsonify

# Start flask app and set to ngrok
app = Flask(__name__)
run_with_ngrok(app)

# llm code

model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin" # the model is in bin format

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

# GPU
lcpp_llm = None
lcpp_llm = Llama(
    model_path=model_path,
    n_threads=2, # CPU cores
    n_batch=512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=32 # Change this value based on your model and your GPU VRAM pool.
    )

# See the number of layers in GPU
lcpp_llm.params.n_gpu_layers



# model function
def get_llm_Inference(prompt):
  # prompt = "Write a linear regression in python"
  prompt_template=f'''SYSTEM: You are a helpful, respectful and honest assistant. Always answer as helpfully.

  USER: {prompt}

  ASSISTANT:
  '''

  response=lcpp_llm(prompt=prompt_template, max_tokens=256, temperature=0.5, top_p=0.95, repeat_penalty=1.2, top_k=150, echo=True)

  llamResponse = response["choices"][0]["text"]
  # print(llamResponse)
  return llamResponse


# ------------------------------------ Flask App ------------------------------------


@app.route('/chat', methods=['POST'])
def generate_image():
    prompt = request.form['message']
    print(f"Generating an response for {prompt}")
    llm_res = get_llm_Inference(prompt)

    # colorPalate = ["orange", "dark orange", "blue", "red", "white"]

    
    response_data = {
        "message": "Data received successfully",
        "color_select": [1, 0, 0, 0],  # body color, mirror, wheel, caliber
        "chat_response": llm_res,
                    }
    
    return jsonify(response_data)


if __name__ == '__main__':
    app.run()


'''
pip install pyngrok
pip install flask_ngrok

!pip install pyngrok
!pip install flask_ngrok
'''
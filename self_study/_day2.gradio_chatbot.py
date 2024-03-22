import gradio as gr
import os
import time

def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)

def add_file(history, file):
    history = history + [((file.name,), None)]
    return history

def bot(history):
    response = "멋져요!"
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join("avatar.png"))),
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="텍스트를 입력하고 엔터를 치거나 이미지를 업로드하세요",
            container=False,
        )
        btn = gr.UploadButton("Upload", file_types=["image", "video", "audio"])

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

demo.queue()
demo.launch(server_name='0.0.0.0')


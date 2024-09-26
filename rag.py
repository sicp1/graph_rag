import fitz
import re
from embedding import model
import requests
from faiss_controller import faiss_controller
faiss_cont=faiss_controller(1024)
def llm(chat):
    data={"messages":[{"role":"user","content":chat}]}
    url = 'http://127.0.0.1:5000/chat'
    response = requests.post(url, json=data)
    return response
    
def get_pdf_raw_blocks(pdf_name):
    pages_blocks = []
    pattern = re.compile(r'[\n\x0f ]')  # Regex pattern to match newline, \x0f, and space characters
    
    with fitz.open(f"/root/autodl-tmp/pdf/{pdf_name}") as doc:
        for page in doc:
            blocks = [pattern.sub("", block[4]) for block in page.get_text("blocks")]
            page_text = "".join(blocks)
            pages_blocks.append(page_text)
            
    return pages_blocks

sentences=get_pdf_raw_blocks("ecs-price-pdf.pdf")
embed_sentences=model.encode(sentences)
faiss_cont.vector_add(embed_sentences)

while True:
    chat=input("输入你的问题: ")
    search_res_index=faiss_cont.vector_search(model.encode([chat]),5,100)
    search_res=[sentences[item] for item in search_res_index]
    nn="\n"
    response=llm(f"""请根据相关信息去回答用户的问题。
用户的问题为:{chat}
相关信息为:
{nn.join(search_res)}
""")
    print("response: ",response.text)
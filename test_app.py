# Offline check on a tiny stand-in manual: fake LLM, no reranker. Run: NVIDIA_API_KEY=dummy python3 test_app.py
import io
import os
import tempfile

import fitz
from PIL import Image
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.prompts import format_document


def png(size):
    buf = io.BytesIO()
    Image.new("RGB", (size, size), "red").save(buf, "PNG")
    return buf.getvalue()


pdf = fitz.open()
p1 = pdf.new_page()
p1.insert_text((72, 72), "The wiring diagram below shows the relay layout.")
p1.insert_image(fitz.Rect(72, 100, 372, 400), stream=png(300))  # real diagram
p1.insert_image(fitz.Rect(500, 20, 520, 40), stream=png(20))  # tiny logo, should be skipped
pdf.new_page().insert_text((72, 72), "The warranty period is 24 months.")
tmp = tempfile.mkdtemp()
os.environ["MANUAL_PDF"], os.environ["DB_DIR"] = os.path.join(tmp, "manual.pdf"), os.path.join(tmp, "db")
pdf.save(os.environ["MANUAL_PDF"])

import app  # builds the index for the stand-in manual

assert os.path.exists(os.path.join(os.environ["DB_DIR"], "done"))
client = TestClient(app.app)
assert client.get("/").status_code == 200

chain = app.qa_chain
chain.retriever = chain.retriever.base_retriever  # skip the cloud reranker
label = format_document(Document(page_content="x", metadata={"page_no": 1, "images": 1}), app.document_prompt)
assert label.startswith("[Page 1 | 1 image(s)"), label

chain.combine_documents_chain.llm_chain.llm = FakeListLLM(responses=["The relay layout is on page 1."])
r = client.post("/ask", json={"query": "show me the wiring diagram"}).json()
assert [i["caption"] for i in r["images"]] == ["Manual page 1 (scanned)"], r["images"]  # one image, logo filtered out

chain.combine_documents_chain.llm_chain.llm = FakeListLLM(responses=["Remove the relay as shown (page 1)."])
r = client.post("/ask", json={"query": "how do I replace the relay?"}).json()
assert [i["caption"] for i in r["images"]] == ["Manual page 1 (scanned)"], "how-to answers show cited pages"

chain.combine_documents_chain.llm_chain.llm = FakeListLLM(responses=[app.REFUSAL])
r = client.post("/ask", json={"query": "show me the diagram of the moon"}).json()
assert r["images"] == [], "refusal must not show images"
app.CHAT_PASSWORD = "s3cret"
assert client.get("/").status_code == 401, "password required when set"
assert client.get("/", auth=("anyone", "wrong")).status_code == 401
assert client.get("/", auth=("anyone", "s3cret")).status_code == 200
app.CHAT_PASSWORD = None
print("all checks passed")

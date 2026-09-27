# Honda Accord 1994–1997 Service Manual chatbot

Chat with the service manual. Answers come only from `manual.pdf`, cite page numbers, and show the scanned pages for diagrams and how-to questions. DeepSeek (via NVIDIA's free API catalog) writes the answers.

## Run

```bash
pip install -r requirements.txt
NVIDIA_API_KEY=nvapi-... python3 -m uvicorn app:app --port 8000
```

Open http://localhost:8000. The first start builds the search index in `chroma_db/` (about 5 minutes); it's committed, so normally startup takes about 10 seconds. Delete `chroma_db/` to rebuild it after changing the PDF.

Get a free key at https://build.nvidia.com. Never put it in the code: pass it as the `NVIDIA_API_KEY` environment variable (or a host "secret").

## Share it from your computer (free tunnel)

```bash
cp .env.example .env   # then put your key (and a password) in .env
./run-public.sh
```

It starts the server and a Cloudflare quick tunnel, then prints a `https://....trycloudflare.com` link anyone can open. Visitors are asked for `CHAT_PASSWORD` (any username). The link works only while this runs and changes each time. Needs `cloudflared` on your PATH.

## Test

```bash
NVIDIA_API_KEY=dummy python3 test_app.py
```

Runs offline with a fake model and a two-page stand-in PDF.

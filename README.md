# Docs-only support chatbot

A support chat that answers only from one document you give it, such as a product manual, a policy handbook, or an FAQ. It cites page numbers, shows the page when a diagram or how-to question needs it, and refuses anything the document doesn't cover. DeepSeek (via NVIDIA's free API catalog) writes the answers.

## Run

```bash
pip install -r requirements.txt
NVIDIA_API_KEY=nvapi-... python3 -m uvicorn app:app --port 8000
```

Open http://localhost:8000. Get a free key at https://build.nvidia.com. Never put it in the code: pass it as the `NVIDIA_API_KEY` environment variable (or a host "secret").

## Use it for your own document

1. Replace `manual.pdf` with your PDF (or point `MANUAL_PDF` at it).
2. Delete `chroma_db/`. The first start rebuilds the search index (about five minutes for 1,400 pages); later starts take about ten seconds.
3. Edit `config.json`:
   - `name`, `greeting`, `tagline`, `placeholder`: the text on the chat page.
   - `document`: how the prompt names the source, e.g. "the Acme Router X2 user guide".
   - `logo`: a single-colour SVG or transparent PNG in `static/`. The page paints it in the theme colour.
   - `suggestions`: starter questions as `[question, note]` pairs. Pick ones the document answers.
   - `min_relevance`: questions whose best-matching passage scores below this are refused before the model sees them. `-4.0` suited the test manual (real questions scored -2.9 or higher, off-topic ones -4.5 or lower); check a few of your own questions and adjust.

Set `CHAT_CONFIG` to use a config file somewhere else.

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

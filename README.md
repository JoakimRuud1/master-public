# Master — Prompt engineering for medical summary generation

Master's thesis project: using prompting strategies to optimize patient
summaries from emergency-room (legevakt) conversations. Done in collaboration
with Helseetaten / Oslo kommune.

> Bruk av prompting for å optimere pasientjournaler ved legevakten i
> samarbeid med helseetaten.

## Setup

1. Clone the repo and create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   # eller: venv\Scripts\activate  (Windows)
   pip install -r requirements.txt
   ```

2. Create a `.env` file at the project root with your API keys:

   ```text
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   ```

   `.env` is git-ignored and must never be committed.

## Data

The project uses ACI-Bench directly from:

```text
data/aci_bench/src_experiment_data_json/
```

Files included in this repo:

```text
train_aci_asrcorr.json
valid_aci_asrcorr.json
test1_aci_asrcorr.json
test2_aci_asrcorr.json
test3_aci_asrcorr.json
```

The code uses `aci_asrcorr` as the transcript variant and maps ACI-Bench rows
to internal IDs in the form:

```text
{split}:aci_asrcorr:{file}
```

Example: `test1:aci_asrcorr:0-aci`.

The first two `train` examples are reserved for one-shot / two-shot prompting
and are excluded from evaluation:

```text
train:aci_asrcorr:0-aci
train:aci_asrcorr:1-aci
```

## Typical usage

Generate summaries on the main split:

```bash
python src/run_generate.py --strategies configs/strategies.json --splits test1
```

Judge the generated summaries:

```bash
python src/run_judge.py --run-dir runs/<run_id>
```

Model, API endpoint, reasoning effort and max output length are configured in:

```text
configs/endpoints.json
```

Default setup:

```text
generator: responses, gpt-5.4, 1500 output tokens
judge:     responses, gpt-5.4, reasoning high, 16000 output tokens
```

For reasoning models, "thinking tokens" count against `max_output_tokens`, so
the judge needs a larger budget than the generator to avoid empty responses at
high reasoning effort.

## Resume on interruption

`run_generate.py` writes each produced row to `summaries.jsonl` immediately
and can be resumed if the run is interrupted:

```bash
python src/run_generate.py --strategies configs/strategies.json --splits test1 --resume runs/<run_id>
```

Already-produced `(conversation_id, strategy_id)` pairs are skipped. Failed
pairs are logged to `summaries_errors.jsonl` and re-generated on the next
resume run.

## Comparing against manual scoring

```bash
python src/compare_judge_manual.py --run-dir runs/<run_id> --manual path/to/manual_scoring.csv
```

Manual scoring files are kept private under `results/` and are not part of
this repo.

## License

TBD — choose a license before going public (MIT is a common choice for
research code). Replace this section with the actual license text.

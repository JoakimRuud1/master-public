# Master — Promptstrategier for KI-genererte kliniske sammendrag

Kode, prompter, datautvalg og analyseverktøy for masteroppgaven

> **Hvordan instruksjons- og promptstrategier påvirker kvaliteten på
> KI-genererte journalsammendrag i klinisk kontekst**
> *(How instruction and prompt strategies affect the quality of AI-generated
> clinical summaries in a clinical context)*

skrevet av **Rajvir Singh Aujla** og **Joakim Otto Ruud**, masteroppgave i
industriell økonomi (60 stp) ved Norges miljø- og biovitenskapelige
universitet (NMBU), våren 2026.

Arbeidet er knyttet til samarbeidet mellom Oslo kommune (Helseetaten) og
Bouvet rundt KI-støttet journalføring.

## Problemstilling

> Hvordan påvirker ulike promptstrategier kvaliteten på KI-genererte
> journalsammendrag fra simulerte kliniske konsultasjoner?

Problemstillingen operasjonaliseres gjennom tre forskningsspørsmål:

1. Hvilke instruksjons- og promptstrategier gir høyest samlet kvalitet på
   KI-genererte journalsammendrag sammenlignet med en baseline-prompt?
2. Hvordan påvirker ulike promptstrategier balansen mellom fullstendighet,
   faktuell forankring og faktiske feil i sammendragene?
3. I hvilken grad gir en begrenset manuell samsvarskontroll grunnlag for å
   bruke LLM-dommeren som evalueringsverktøy i denne studien?

## Hva ligger i dette repoet

Dette er den offentlige delen av kodebasen. Det dekker:

- **Åtte promptstrategier** under `prompts/`: minimal baseline, zero-shot
  strukturert, one-shot, two-shot, og kombinasjoner av decomposition,
  self-criticism og ensemble.
- **Generering** av sammendrag med valgt strategi (`src/run_generate.py`).
- **Automatisk evaluering** med LLM-as-judge mot et kriteriesett tilpasset fra
  PDSQI-9 (`src/run_judge.py`, `src/judge_*.py`).
- **Analyse**: deskriptiv sammenligning, plotting, sammenslåing av kjøringer,
  feilanalyse og sammenligning mot manuell scoring (`src/plot_results.py`,
  `src/report_results.py`, `src/compare_judge_manual.py`,
  `src/pool_test_runs.py`, `src/explore_results.py`,
  `src/extract_hallucination_candidates.py`).
- **Datagrunnlag**: ACI-Bench under `data/aci_bench/` brukes som offentlig
  proxy for kliniske samtaler. Private manuelle vurderinger og resultater fra
  den lokale plausibilitetstesten på Helseetaten-data inngår ikke i repoet.
- **Manuelle vedlegg**: de to Excel-filene i rotmappen er offentlige vedlegg
  brukt til manuell evaluering og dommerkontroll i oppgaven.

Repoet inneholder ikke genererte `runs/`-mapper, resultatmapper eller ferdige
plott fra analysen. Slike artefakter kan regenereres med skriptene over dersom
man har nødvendige API-nøkler og modelltilgang.

## Setup

1. Klon repoet og lag et virtuelt miljø:

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   # eller: venv\Scripts\activate  (Windows)
   pip install -r requirements.txt
   ```

2. Lag en `.env` i prosjektroten med API-nøkkelen din, eventuelt ved å kopiere
   `.env.example`:

   ```text
   OPENAI_API_KEY=sk-...
   ```

   `.env` er git-ignorert og skal aldri committes.

## Data

Prosjektet bruker [ACI-Bench](https://github.com/wyim/aci-bench) direkte fra:

```text
data/aci_bench/src_experiment_data_json/
```

Filer som er inkludert i repoet:

```text
train_aci_asrcorr.json
valid_aci_asrcorr.json
test1_aci_asrcorr.json
test2_aci_asrcorr.json
test3_aci_asrcorr.json
```

Koden bruker `aci_asrcorr` som transkripsjonsvariant og mapper ACI-Bench-rader
til interne IDer på formen:

```text
{split}:aci_asrcorr:{file}
```

Eksempel: `test1:aci_asrcorr:0-aci`.

De to første `train`-eksemplene er reservert for one-shot / two-shot prompting
og er ekskludert fra evaluering:

```text
train:aci_asrcorr:0-aci
train:aci_asrcorr:1-aci
```

ACI-Bench beskrives i artikkelen
[ACI-BENCH: a Novel Ambient Clinical Intelligence Dataset for Benchmarking
Automatic Visit Note Generation](https://www.nature.com/articles/s41597-023-02487-3).
Datasettet er publisert under Creative Commons Attribution 4.0 International
License (CC BY 4.0). Se
[ACI-Bench-repoet](https://github.com/wyim/aci-bench) for original lisens og
siteringsinformasjon.

## Typisk bruk

Generer sammendrag på hovedsplittet:

```bash
python src/run_generate.py --strategies configs/strategies.json --splits test1
```

Evaluer de genererte sammendragene:

```bash
python src/run_judge.py --run-dir runs/<run_id>
```

Modell, API-endepunkt, reasoning-effort og maks utlengde konfigureres i:

```text
configs/endpoints.json
```

Standardoppsett:

```text
generator: responses, gpt-5.4, 1500 output tokens
judge:     responses, gpt-5.4, reasoning high, 16000 output tokens
```

For reasoning-modeller teller "thinking tokens" mot `max_output_tokens`, så
judge trenger større budsjett enn generatoren for å unngå tomme svar ved høyt
reasoning-nivå.

## Resume ved avbrudd

`run_generate.py` skriver hver produserte rad til `summaries.jsonl` umiddelbart
og kan gjenopptas hvis kjøringen avbrytes:

```bash
python src/run_generate.py --strategies configs/strategies.json --splits test1 --resume runs/<run_id>
```

Allerede produserte `(conversation_id, strategy_id)`-par hoppes over. Feilede
par logges til `summaries_errors.jsonl` og regenereres ved neste resume.

## Sammenligning mot manuell scoring

```bash
python src/compare_judge_manual.py --run-dir runs/<run_id> --manual path/to/manual_scoring.csv
```

De to Excel-filene i rotmappen er offentlige manuelle vedlegg:

```text
manual_evaluation_2samtaler_tilbakemeldinger.xlsx
manuell_dommeranalyse_5samtaler.xlsx
```

Øvrige private resultater, lokale kjøringer og manuelle arbeidsfiler holdes
utenfor repoet, for eksempel under `results/`.

## Forfattere

- **Rajvir Singh Aujla**
- **Joakim Otto Ruud**

## Lisens

Koden i dette repoet er lisensiert under [MIT License](LICENSE). MIT er valgt
fordi den er den vanligste lisensen for åpen forskningskode og legger få
restriksjoner på gjenbruk i videre arbeid.

ACI-Bench-datasettet under `data/aci_bench/` er ikke lisensiert under repoets
MIT-lisens. Det følger kildens egen CC BY 4.0-lisens; se
[ACI-Bench-repoet](https://github.com/wyim/aci-bench) for vilkår og
siteringsinformasjon.

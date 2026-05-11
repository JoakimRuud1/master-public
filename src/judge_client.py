from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Optional

from openai import OpenAI

try:
    from src.judge_schema import (
        GATE_SCORE_KEY,
        QUALITY_SCORE_KEYS,
        SCORE_KEYS,
        SYNTHESIS_SCORE_KEY,
        VOICE_SCORE_KEYS,
        calculate_primary_score,
        coerce_int_score,
        derive_voice_analysis,
    )
except Exception:
    from judge_schema import (  # type: ignore
        GATE_SCORE_KEY,
        QUALITY_SCORE_KEYS,
        SCORE_KEYS,
        SYNTHESIS_SCORE_KEY,
        VOICE_SCORE_KEYS,
        calculate_primary_score,
        coerce_int_score,
        derive_voice_analysis,
    )


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv()


_load_dotenv_if_available()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

RAW_RESPONSE_ALLOWED_KEYS = set(SCORE_KEYS) | {"rationale"}
RAW_RESPONSE_FORBIDDEN_KEYS = {
    "run_id",
    "conversation_id",
    "strategy_id",
    "split",
    "transcript_variant",
    "scores",
    "primary_score",
    "included_dimensions",
    "voice_analysis",
}


class JudgeResponseError(RuntimeError):
    def __init__(self, message: str, *, raw_response: Optional[str] = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


# fmt: off
RUBRIC_SET = """
<citation>
BESKRIVELSE: Er påstandene i CLINICAL_NOTE hensiktsmessig forankret i SOURCE_TRANSCRIPT?
MERK: Dette kriteriet krever IKKE synlige sitater i selve notatet. Det vurderer om hver faktiske påstand i notatet kan spores tilbake til kildetranskriptet uten tvetydighet.
MERK: En påstand er en faktisk utsagn som kan bestå av én eller flere setninger.
MERK: Ustøttede tillegg, tvetydig forankring eller utsagn som ikke kan lokaliseres i kildetranskriptet, teller negativt på dette kriteriet.

GRADERINGER:
1 = Flere påstander er ikke støttet av kildetranskriptet
2 = Én tydelig ustøttet påstand ELLER flere tvetydig støttede påstander
3 = De fleste påstander er støttet, men minst én påstand mangler tydelig forankring
4 = Alle klinisk relevante påstander er støttet, med bare mindre tvetydighet i forankringen
5 = Hver påstand i notatet kan tydelig spores til kildetranskriptet
<\\citation>

<accurate>
BESKRIVELSE: CLINICAL_NOTE er faktisk korrekt i forhold til SOURCE_TRANSCRIPT.
MERK: Ukorrekt informasjon inkluderer fabrikasjon, forfalskning, motsigelse, feilattribusjon, feil timing, feil negasjon og overdrivelse av sikkerhet.
MERK: Fabrikasjon betyr å legge til informasjon som ikke er til stede i kildetranskriptet. Forfalskning betyr å endre viktige detaljer slik at meningen ikke lenger er tro mot kilden.
MERK: Eksempler på problematiske feil inkluderer: symptomer oppgitt som fraværende når de var til stede, symptomer oppgitt som til stede når de ble benektet, feil taler som får tillagt et utsagn, feil timing eller varighet, og dokumentering av usikkerhet som om det var etablert fakta.

GRADERINGER:
1 = Flere store faktafeil, fabrikasjoner eller forfalskninger
2 = Én stor faktafeil, fabrikasjon eller forfalskning
3 = Minst én tydelig feilfremstilling av kontekst, timing, negasjon, attribusjon eller sikkerhet
4 = Mindre unøyaktigheter eller upresisjoner er til stede, men den overordnede kliniske meningen forblir tro mot kilden
5 = Alle påstander er faktisk korrekte og tro mot kildetranskriptet
<\\accurate>

<thorough>
BESKRIVELSE: CLINICAL_NOTE inkluderer den klinisk relevante informasjonen som trengs fra møtet.
MERK: Dette kriteriet vurderer fullstendighet for klinisk bruk, ikke om hver detalj fra transkriptet er inkludert.
MERK: Relevante utelatelser er klinisk viktige detaljer som burde ha vært inkludert. Potensielt relevante utelatelser er relevante, men mindre essensielle detaljer.

GRADERINGER:
1 = Mer enn én relevant utelatelse er til stede
2 = Én relevant utelatelse og flere potensielt relevante utelatelser er til stede
3 = Én relevant utelatelse er til stede
4 = Noen potensielt relevante utelatelser er til stede, men ingen tydelig relevant utelatelse
5 = Ingen relevant eller potensielt relevant utelatelse er til stede
<\\thorough>

<useful>
BESKRIVELSE: CLINICAL_NOTE er klinisk nyttig for det tiltenkte helsepersonellet.
MERK: Informasjon bør være relevant, hensiktsmessig prioritert og presentert på et passende detaljnivå for klinisk dokumentasjon.
MERK: Svært relevant og sikkerhetskritisk informasjon bør fremheves over mindre eller distraherende detaljer.

GRADERINGER:
1 = Notatet er i stor grad ikke nyttig for klinisk beslutningstaking eller oppfølging
2 = Noe nyttig informasjon er til stede, men viktig innhold er dårlig prioritert eller blandet med irrelevante detaljer
3 = Notatet er generelt nyttig, men prioritering eller detaljnivå er ikke passende
4 = Notatet er klinisk nyttig og for det meste godt prioritert, med bare mindre problemer i vektlegging eller relevans
5 = Notatet er klinisk nyttig, hensiktsmessig prioritert og på riktig detaljnivå hele veien
<\\useful>

<organized>
BESKRIVELSE: CLINICAL_NOTE er strukturert på en logisk måte som støtter rask forståelse av møtet.
MERK: Informasjon bør fremkomme i en rekkefølge og gruppering som passer klinisk dokumentasjon og gjør notatet lett å skanne.
MERK: God organisering kan være temporal, problembasert eller seksjonsbasert, så lenge den forbedrer lesbarhet og klinisk forståelse.

GRADERINGER:
1 = Notatet er gjennomgående uorganisert og vanskelig å følge
2 = Viktige deler er ute av rekkefølge eller dårlig gruppert
3 = Notatet er forståelig, men strukturen tilfører liten klarhet
4 = Notatet er logisk strukturert, med bare mindre svakheter i gruppering eller flyt
5 = Notatet er konsekvent godt organisert og lett å skanne og tolke
<\\organized>

<comprehensible>
BESKRIVELSE: Språket i CLINICAL_NOTE er klart, presist og lett for det tiltenkte helsepersonellet å forstå.
MERK: Dette inkluderer klarhet i ordlyd, entydig referanse til personer og hendelser, og tydelig skille mellom rapportert informasjon, observerte funn og usikkerhet.
MERK: Tvetydig formulering, vage referanser eller unødvendig kompleks ordlyd teller negativt på dette kriteriet.

GRADERINGER:
1 = Notatet er ofte uklart, tvetydig eller vanskelig å forstå
2 = Det er flere klarhetsproblemer eller minst én stor tvetydighet
3 = Notatet er forståelig totalt sett, men inneholder merkbar tvetydighet eller upresisjon
4 = Notatet er klart og for det meste presist, med bare mindre tvetydighet
5 = Notatet er konsekvent klart, presist og entydig
<\\comprehensible>

<succinct>
BESKRIVELSE: CLINICAL_NOTE er kortfattet og unngår unødvendig repetisjon eller detalj.
MERK: Korthet bør ikke skje på bekostning av å utelate klinisk relevant informasjon.
MERK: Redundans, fyll og unødvendig lang formulering teller negativt på dette kriteriet.

GRADERINGER:
1 = Notatet er overdrevent langt, repetitivt eller fylt med unødvendige detaljer gjennomgående
2 = Mer enn én del av notatet er unødvendig ordrik eller redundant
3 = Notatet er noe ordrikt eller repetitivt på minst ett viktig sted
4 = Notatet er samlet sett kortfattet, med bare mindre overskudd av ord
5 = Notatet er kortfattet, effektivt og fritt for unødvendig repetisjon
<\\succinct>

<abstraction>
BESKRIVELSE: Er syntese utover enkel ekstraksjon nødvendig for dette møtet?
MERK: Syntese betyr å kombinere eller reformulere informasjon fra SOURCE_TRANSCRIPT til et klinisk nyttig sammendrag, samtidig som det forblir tro mot kilden.
MERK: Dette bør markeres "Ja" bare når enkel ekstraksjon ikke er tilstrekkelig for å produsere et godt klinisk notat.

GRADERINGER:
0 = Nei
1 = Ja
<\\abstraction>

<synthesized>
BESKRIVELSE: Når syntese er nødvendig, integrerer og reformulerer CLINICAL_NOTE informasjon fra SOURCE_TRANSCRIPT på en hensiktsmessig måte uten å introdusere ustøttet resonnement.
MERK: God syntese betyr å kombinere relevant informasjon til et klinisk nyttig sammendrag. Det krever IKKE spekulativ inferens eller diagnostisk overtråkk.
MERK: Overtolkning, ustøttede konklusjoner eller syntese som går utover kilden teller negativt på dette kriteriet.

GRADERINGER:
NA = Det er ikke behov for syntese
1 = Notatet introduserer ukorrekt eller ustøttet resonnement
2 = Syntese utføres når det ikke er nødvendig, eller den kombinerer informasjon på en villedende måte
3 = Notatet forblir for det meste ekstraktivt når noe nyttig syntese var nødvendig
4 = Notatet viser hensiktsmessig syntese, men integreringen er begrenset eller ujevn
5 = Notatet integrerer og oppsummerer møtet på en hensiktsmessig, klar måte og uten ustøttet inferens
<\\synthesized>

<voice_summ>
BESKRIVELSE: Inneholder CLINICAL_NOTE stigmatiserende, dømmende eller unødvendig ladet språk?
MERK: Notatet bør bruke nøytralt og profesjonelt klinisk språk.

GRADERINGER:
0 = Nei
1 = Ja
<\\voice_summ>

<voice_note>
BESKRIVELSE: Inneholder SOURCE_TRANSCRIPT stigmatiserende, dømmende eller unødvendig ladet språk?
MERK: Dette kriteriet er inkludert for å skille problematisk språk som allerede er til stede i kilden fra språk introdusert av modellen.

GRADERINGER:
0 = Nei
1 = Ja
<\\voice_note>
"""
# fmt: on


def _is_null_equivalent(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "na", "n/a", "null", "none"}:
        return True
    return False


def _validate_raw_judge_response(obj: Dict[str, Any]) -> tuple[Dict[str, int | None], str]:
    if not isinstance(obj, dict):
        raise ValueError("Raw judge response must be a JSON object.")

    keys = set(obj.keys())
    forbidden = sorted(keys & RAW_RESPONSE_FORBIDDEN_KEYS)
    if forbidden:
        raise ValueError(f"Raw judge response included forbidden key(s): {forbidden}")

    unexpected = sorted(keys - RAW_RESPONSE_ALLOWED_KEYS)
    if unexpected:
        raise ValueError(f"Raw judge response included unexpected key(s): {unexpected}")

    fixed_scores: Dict[str, int | None] = {}
    for k in QUALITY_SCORE_KEYS:
        if k not in obj:
            raise ValueError(f"Missing raw judge score key: {k}")
        fixed_scores[k] = coerce_int_score(obj[k], k, (1, 2, 3, 4, 5))

    if GATE_SCORE_KEY not in obj:
        raise ValueError(f"Missing raw judge score key: {GATE_SCORE_KEY}")
    fixed_scores[GATE_SCORE_KEY] = coerce_int_score(obj[GATE_SCORE_KEY], GATE_SCORE_KEY, (0, 1))

    if SYNTHESIS_SCORE_KEY not in obj:
        raise ValueError(f"Missing raw judge score key: {SYNTHESIS_SCORE_KEY}")
    if fixed_scores[GATE_SCORE_KEY] == 1:
        fixed_scores[SYNTHESIS_SCORE_KEY] = coerce_int_score(
            obj[SYNTHESIS_SCORE_KEY],
            SYNTHESIS_SCORE_KEY,
            (1, 2, 3, 4, 5),
        )
    else:
        if not _is_null_equivalent(obj[SYNTHESIS_SCORE_KEY]):
            raise ValueError("synthesized must be null/NA when abstraction = 0")
        fixed_scores[SYNTHESIS_SCORE_KEY] = None

    for k in VOICE_SCORE_KEYS:
        if k not in obj:
            raise ValueError(f"Missing raw judge score key: {k}")
        fixed_scores[k] = coerce_int_score(obj[k], k, (0, 1))

    rationale = obj.get("rationale", "")
    if rationale is None:
        rationale = ""
    if not isinstance(rationale, str):
        rationale = str(rationale)

    return fixed_scores, rationale.strip()[:600]


def _build_stored_judgement(
    *,
    run_id: str,
    conversation_id: str,
    strategy_id: str,
    scores: Dict[str, int | None],
    rationale: str,
) -> Dict[str, Any]:
    primary_score, included_dimensions = calculate_primary_score(scores)
    return {
        "run_id": run_id,
        "conversation_id": conversation_id,
        "strategy_id": strategy_id,
        "scores": scores,
        "primary_score": primary_score,
        "included_dimensions": included_dimensions,
        "voice_analysis": derive_voice_analysis(scores),
        "rationale": rationale,
    }


def judge_summary(
    *,
    run_id: str,
    conversation_id: str,
    strategy_id: str,
    transcript_text: str,
    summary_text: str,
    api_endpoint: str = "responses",
    model: str = "gpt-5.4",
    temperature: Optional[float] = 0.0,
    max_output_tokens: int = 1500,
    reasoning_effort: Optional[str] = "high",
    retries: int = 2,
    retry_backoff_s: float = 1.0,
) -> Dict[str, Any]:
    """
    Parses a raw LLM judge response and returns a stored judge record:
    {
      run_id, conversation_id, strategy_id,
      scores:{...}, primary_score:float, included_dimensions:[...],
      voice_analysis:{...}, rationale:str
    }
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    if api_endpoint != "responses":
        raise ValueError(f"Unsupported api_endpoint: {api_endpoint}. Only 'responses' is implemented.")

    system_prompt = (
        "You are a strict evaluator of clinical note summaries. "
        "Return ONLY valid JSON. No markdown, no extra text."
    )

    raw_response_template = {
        "citation": 1,
        "accurate": 1,
        "thorough": 1,
        "useful": 1,
        "organized": 1,
        "comprehensible": 1,
        "succinct": 1,
        "abstraction": 0,
        "synthesized": None,
        "voice_summ": 0,
        "voice_note": 0,
        "rationale": "Short rationale (1-3 sentences).",
    }

    user_prompt = f"""
Du skal vurdere CLINICAL_NOTE opp mot SOURCE_TRANSCRIPT ved å bruke RUBRIC_SET nedenfor.
Rubrikkteksten i RUBRIC_SET er styrende for graderingen.
Returner KUN ett gyldig JSON-objekt som matcher schemaet nederst. Ikke bruk markdown eller ekstra tekst.

OUTPUT-REGLER:
- Returner et flatt JSON-objekt med kun feltene citation, accurate, thorough, useful, organized, comprehensible, succinct, abstraction, synthesized, voice_summ, voice_note og eventuelt rationale.
- Ikke returner `scores`, `run_id`, `conversation_id`, `strategy_id`, `split`, `transcript_variant`, `primary_score`, `included_dimensions` eller `voice_analysis`.
- Bruk heltall 1-5 for citation, accurate, thorough, useful, organized, comprehensible og succinct.
- Bruk heltall 0-1 for abstraction, voice_summ og voice_note.
- Når abstraction = 0, skal synthesized være null i JSON-output.
- Når abstraction = 1, skal synthesized være et heltall fra 1-5.
- SOURCE_TRANSCRIPT kan være på engelsk og CLINICAL_NOTE kan være på norsk.

RUBRIC_SET:
{RUBRIC_SET}

SOURCE_TRANSCRIPT:
{transcript_text}

CLINICAL_NOTE:
{summary_text}

RAW RESPONSE JSON SCHEMA (fill values, keep keys exactly):
{json.dumps(raw_response_template, ensure_ascii=False)}
""".strip()


    input_items = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_err: Exception | None = None
    last_raw_response: str | None = None
    for attempt in range(1, retries + 1):
        try:
            req: Dict[str, Any] = {
                "model": model,
                "input": input_items,
                "max_output_tokens": max_output_tokens,
            }
            if reasoning_effort is not None:
                req["reasoning"] = {"effort": reasoning_effort}
            elif temperature is not None:
                req["temperature"] = float(temperature)

            resp = _client.responses.create(**req)
            raw = (resp.output_text or "").strip()
            last_raw_response = raw
            if not raw:
                raise RuntimeError("Empty response from model.")

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
                raw = re.sub(r"\n?```\s*$", "", raw)

            # Parse JSON strictly
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError("Model output was not a JSON object.")

            scores, rationale = _validate_raw_judge_response(obj)
            return _build_stored_judgement(
                run_id=run_id,
                conversation_id=conversation_id,
                strategy_id=strategy_id,
                scores=scores,
                rationale=rationale,
            )

        except Exception as e:
            last_err = e
            # Retry once on invalid JSON/schema, then let run_judge mark the case as failed.
            if attempt < retries:
                time.sleep(retry_backoff_s * attempt)
            else:
                break

    raise JudgeResponseError(
        f"judge_summary failed after {retries} attempts: {last_err}",
        raw_response=last_raw_response,
    ) from last_err

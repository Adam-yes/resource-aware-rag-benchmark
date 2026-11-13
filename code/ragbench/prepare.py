"""Builds data/questions.csv from the archive and the sheets people must fill in."""
import pandas as pd

from .common import ARCHIVE, DATA, sha

MSLA = r"sl1|cw1"  # Resin (MSLA) printers and their curing stations: not FDM.
POINTER = (r"(?i)refer to|see (?:section|chapter|the)|section \d|chapter \d|can be found|is discussed|"
           r"is described in|mentioned in|covered in|the manual (?:provides|contains|explains)|listed (?:as|in)")


def build_questions():
    dev = pd.read_csv(ARCHIVE / "data/evaluation/evaluation_testset_50_short_query_length.csv")
    pool = pd.concat([pd.read_csv(f).assign(source_file=f.name)
                      for f in sorted((ARCHIVE / "data/processed/testset").glob("*.csv"))], ignore_index=True)
    test_a = pool[~pool.user_input.isin(dev.user_input)]
    assert len(dev) == 50 and len(test_a) == 162 and not set(dev.user_input) & set(test_a.user_input)
    df = pd.concat([dev.assign(split="dev"), test_a.assign(split="test_a")], ignore_index=True)
    human = DATA / "test_b_human.csv"
    if human.exists():
        b = pd.read_csv(human).dropna(subset=["user_input", "reference"]).assign(split="test_b")
        b["query_style"], b["query_length"] = "HUMAN", "HUMAN"
        df = pd.concat([df, b], ignore_index=True)
    df["qid"] = df.user_input.map(lambda q: sha(q, 10))
    assert df.qid.is_unique
    df["process"] = df.source_file.str.contains(MSLA, case=False).map({True: "MSLA", False: "FDM"})
    df["pointer_ref"] = df.reference.str.contains(POINTER)
    # The archived generator also wrote questions from tables of contents, which cannot support a real answer.
    df["toc_ref"] = df.reference_contexts.astype(str).str.contains(r"\.{12,}")
    df = df.rename(columns={"user_input": "question"})
    DATA.mkdir(exist_ok=True)
    cols = ["qid", "split", "question", "reference", "reference_contexts", "source_file", "process",
            "query_style", "query_length", "persona_name", "pointer_ref", "toc_ref"]
    df[cols].to_csv(DATA / "questions.csv", index=False)
    return df[cols]


def write_sheets(df):
    """Reference audit for dev: pointer-style references come first, because a reference such as
    'see section 7.3.3' cannot give credit to a correct answer."""
    dev = df[df.split == "dev"].copy()
    dev["flag_pointer"] = dev.reference.str.contains(POINTER)
    dev = dev.sort_values("flag_pointer", ascending=False)
    dev["repaired_reference"] = ""
    dev["comment"] = ""
    dev[["qid", "flag_pointer", "source_file", "question", "reference", "repaired_reference", "comment"]].to_csv(
        DATA / "reference_audit_SHEET.csv", index=False)
    template = DATA / "test_b_human_TEMPLATE.csv"
    if not template.exists():
        template.write_text("user_input,reference,reference_contexts,source_file,persona_name\n")
    return int(dev.flag_pointer.sum())


def main():
    df = build_questions()
    flagged = write_sheets(df)
    counts = df.groupby(["split", "process"]).size().to_dict()
    dev, ta = df[df.split == "dev"], df[df.split == "test_a"]
    print("test_a: same source passage as a dev question:", int(ta.reference_contexts.isin(dev.reference_contexts).sum()),
          "| short:", int((ta.query_length == "SHORT").sum()), "| pointer-style references:", int(ta.pointer_ref.sum()), "| from a table of contents:", int(ta.toc_ref.sum()),
          "| manuals without dev question:", sorted(set(ta.source_file) - set(dev.source_file)))
    print(f"questions.csv: {counts}; {flagged} pointer-style dev references flagged for review")

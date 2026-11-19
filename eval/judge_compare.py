"""Score the judge against a human-labeled file (needs a local Ollama with the judge model).

Run from the project root:

    python3 eval/judge_compare.py [labeled.jsonl] [--limit N]

Each row needs question, context, answer, human_faithful and human_relevant. For both metrics it
prints accuracy, how many human-labeled failures the judge caught (the number that matters for the
gate), how many good answers it wrongly failed, and every disagreement. The default file is the
20-case illustrative set, whose labels were written by an assistant, so treat the numbers as a
smoke test until a person has labeled real cases.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agents

HUMAN = {"faithfulness": "human_faithful", "relevance": "human_relevant"}
DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labeled_set.example.jsonl")


def load(path, limit=None):
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows[:limit] if limit else rows


def summarize(name, rows, key):
    graded = [r for r in rows if r["verdicts"][key] in ("pass", "fail")]
    errored = [r for r in rows if r["verdicts"][key] not in ("pass", "fail")]
    correct = sum((r["verdicts"][key] == "pass") == r[HUMAN[key]] for r in graded)
    human_fail = [r for r in rows if not r[HUMAN[key]]]
    caught = sum(r["verdicts"][key] == "fail" for r in human_fail)
    errored_on_fail = sum(r in errored for r in human_fail)
    human_pass = [r for r in rows if r[HUMAN[key]]]
    wrongly_failed = sum(r["verdicts"][key] == "fail" for r in human_pass)
    # An error is not a catch: the denominator is every human-labeled failure, errors included.
    print(f"{name}: agreement {correct}/{len(graded)} graded"
          f"  | caught {caught}/{len(human_fail)} human-labeled failures ({errored_on_fail} of them errored)"
          f"  | wrongly failed {wrongly_failed}/{len(human_pass)} good answers"
          f"  | errors {len(errored)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", default=DEFAULT)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = load(args.path, args.limit)
    print(f"judge model: {agents.JUDGE_MODEL}   cases: {len(rows)}   file: {os.path.basename(args.path)}")
    started = time.time()
    for index, row in enumerate(rows, start=1):
        try:
            row["verdicts"] = agents.judge(row["question"], row["answer"], row["context"])
        except Exception as error:  # a judge outage should not hide the other results
            row["verdicts"] = {"faithfulness": "error", "relevance": "error", "error": str(error)}
        print(f"  {index:>2}/{len(rows)} faithful {row['verdicts']['faithfulness']:<5} "
              f"relevant {row['verdicts']['relevance']:<5} | human {row['human_faithful']!s:<5} "
              f"{row['human_relevant']!s:<5} | {row['question'][:48]}", flush=True)
    print(f"\ntook {time.time() - started:.0f}s")
    summarize("faithfulness", rows, "faithfulness")
    summarize("relevance   ", rows, "relevance")
    print("\ndisagreements:")
    for row in rows:
        for key, human in ((k, row[HUMAN[k]]) for k in HUMAN):
            verdict = row["verdicts"][key]
            if verdict in ("pass", "fail") and (verdict == "pass") != human:
                print(f"  {key}: judge {verdict}, human {'pass' if human else 'fail'} | {row['question']} -> {row['answer'][:70]}")


if __name__ == "__main__":
    main()

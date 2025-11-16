import glob
import json
import os
import re
import collections


def main() -> None:
    arts = sorted(glob.glob("artifacts/splitter/*.jsonl"))
    rej = collections.Counter()
    adds = 0
    for path in arts[-500:]:
        if not re.fullmatch(r"[0-9a-f]{12}\.jsonl", os.path.basename(path)):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                telemetry = rec.get("telemetry", {})
                adds += telemetry.get("num_model_add", 0)
                for reasons in telemetry.get("rej_reasons", []):
                    for reason in reasons.get("reasons", []):
                        rej[reason] += 1

    print("model_add total:", adds)
    print("reject reasons:", dict(rej))


if __name__ == "__main__":
    main()

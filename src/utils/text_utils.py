import re


def extract_exact_answer(query, docs):
    """
    Extract exact answers directly from retrieved documents.
    Supports:
    - IDs (document numbers)
    - Dates / DateTime
    - Names (people)
    """

    query_lower = query.lower()

    if any(k in query_lower for k in ["number", "document", "id", "code"]):
        id_pattern = r"[A-Z0-9\-]{8,}"

        for d in docs:
            match = re.search(id_pattern, d)
            if match:
                return match.group()

    if any(k in query_lower for k in ["date", "completion", "deadline"]):

        datetime_pattern = r"\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s+[AP]M"

        date_pattern = r"\d{1,2}/\d{1,2}/\d{4}"

        for d in docs:
            match = re.search(datetime_pattern, d)
            if match:
                return match.group()

        for d in docs:
            match = re.search(date_pattern, d)
            if match:
                return match.group()

    if any(k in query_lower for k in ["who", "manager", "person", "assigned"]):

        name_pattern = r"[A-Z][a-z]+ [A-Z][a-z]+"

        candidates = []

        for d in docs:
            matches = re.findall(name_pattern, d)

            for m in matches:
                if m.lower() not in query_lower:
                    candidates.append(m)

        candidates = list(dict.fromkeys(candidates))

        if len(candidates) >= 2:
            return f"{candidates[0]} and {candidates[1]}"
        elif len(candidates) == 1:
            return candidates[0]

    return None
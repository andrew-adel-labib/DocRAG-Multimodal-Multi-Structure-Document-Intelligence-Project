import camelot


def extract_tables(pdf_path):
    table_texts = []

    try:
        tables = camelot.read_pdf(
            pdf_path,
            pages="all",
            flavor="stream"
        )

        for table in tables:
            df = table.df

            text = "\n".join([
                " | ".join(row)
                for row in df.values
            ])

            if text.strip():
                table_texts.append(text)

    except Exception:
        pass

    return table_texts
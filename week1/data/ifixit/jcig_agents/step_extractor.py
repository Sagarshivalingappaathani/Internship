class StepExtractorAgent:
    @staticmethod
    def run(doc1, doc2):
        def extract(doc):
            steps = []
            for step in doc.get("Steps", []):
                if "Text_raw" in step:
                    steps.append(step["Text_raw"].strip())
                elif "Lines" in step:
                    lines = [line.get("Text", "") for line in step["Lines"] if "Text" in line]
                    combined = " ".join(lines).strip()
                    if combined:
                        steps.append(combined)
            return [s for s in steps if len(s.split()) > 2]
        return extract(doc1) + extract(doc2)

from sentence_transformers import SentenceTransformer

class EmbedClusterAgent:
    @staticmethod
    def run(steps):
        from openai import OpenAI
        import re

        # === Step 1: Cluster Steps Using Gemini ===

        client = OpenAI(
            api_key="dummy",  # required, but gets ignored by Bosch infra
            base_url="https://aoai-farm.bosch-temp.com/api/openai/deployments/google-gemini-1-5-flash",
            default_headers={
                "genaiplatform-farm-subscription-key": "0c9f370035e0436989bb962b0d1bb9d0"
            }
        )


        user_prompt = (
            "You are an NLP expert. I have the following procedural steps:\n\n"
            + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)) +
            "\n\nGroup these steps into 3–5 clusters based on semantic similarity. "
            "Return your answer as a Python dictionary like: {0: [0, 2, 3], 1: [1, 4], 2: [5]}."
        )

        res = client.chat.completions.create(
            model="gemini-1.5-flash",
            messages=[
                {"role": "system", "content": "You are a helpful NLP assistant."},
                {"role": "user", "content": user_prompt}
            ]
        )

        response_text = res.choices[0].message.content
        try:
            clean = response_text.strip()
            clean = re.sub(r"```(?:python)?", "", clean)
            clean = re.sub(r"```", "", clean)
            clean = re.sub(r"print\([^)]*\)", "", clean)

            match = re.search(r"\{[\s\S]*\}", clean)
            if match:
                cluster_map = eval(match.group(0))
            else:
                raise ValueError("No valid dict found")

            valid_range = set(range(len(steps)))
            grouped = [[i for i in group if i in valid_range] for group in cluster_map.values()]

        except Exception:
            print("⚠️ Failed to parse cluster output:", response_text)
            grouped = [[i] for i in range(len(steps))]

        # === Step 2: Generate Embeddings Using SBERT ===
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = model.encode(steps)

        return {
            "groups": grouped,
            "embeddings": embeddings
        }


import os
import pickle
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DroolsRAGPipeline:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Check your .env file.")

        self.client = OpenAI(api_key=self.api_key)
        self.index = None
        self.metadata = None

    def load_vector_db(self, faiss_path="faiss_index.bin", metadata_path="metadata.pkl"):
        """Load FAISS index and metadata"""
        self.index = faiss.read_index(faiss_path)
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

    def embed_query(self, query):
        """Generate embedding for query"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def search_chunks(self, query, k=20):
        """Search for relevant chunks"""
        query_embedding = self.embed_query(query).reshape(1, -1)
        scores, indices = self.index.search(query_embedding, k)

        chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                chunks.append({
                    'content': self.metadata[idx].get('text', ''),
                    'score': float(score)
                })
        return chunks

    def load_form(self, form_path="output_form.md"):
        """Load form content"""
        with open(form_path, 'r', encoding='utf-8') as f:
            return f.read()

    def create_prompt(self, form_content, chunks, query):
        """Create prompt for Drools generation"""
        retrieved_context = "\n\n".join([
            f"CHUNK {i+1}: {chunk['content']}" 
            for i, chunk in enumerate(chunks)
        ])
        prompt = f"""System Role:
        You are an expert in Maryland state tax laws and Drools rule engine development with 15+ years of experience.
        Your task is to read:
        1. A subset of the Maryland tax form (from the user query)
        2. Retrieved authoritative rules from the Maryland rule booklet (via FAISS search)
        …and produce ONLY a syntactically correct Drools `.drl` file.

        **Key Requirements:**
        - Use ONLY logic, tax rates, thresholds, and conditions explicitly present in the retrieved rule snippets.
        - Use ONLY field names that exist in the provided form.
        - Replicate EVERY conditional branch exactly as documented, including:
        - County-specific local tax rates
        - Special handling for counties like Anne Arundel & Frederick
        - Different rates based on filing status & income thresholds
        - DO NOT collapse multiple branches into a single formula.
        - If any data is missing from retrieved snippets, insert `// TODO` comments in correct locations.
        - Always end each Drools rule with `end`.
        - No text output outside the `.drl` code.

        **Pipeline Inputs:**
        FORM CONTENT:
        {form_content}
        Query:
        {query}
        (Form section related to the logic)

        {retrieved_context}
        (Authoritative Maryland rule booklet excerpts containing the exact tax rates, income thresholds, and conditions)

        **Execution Steps:**
        1. Parse the **form section** to identify:
        - Java model class and package name
        - Exact field names available
        2. Parse the **retrieved context** to extract:
        - All county-specific logic
        - Special rules and exemptions
        - Income thresholds and filing status differences
        3. Generate Drools code:
        - Correct `package` declaration
        - Correct `import` statement for the form’s Java class
        - Full conditional branching logic from retrieved context
        - Update statements for calculated values

        ---

        **Output Format:**
        package <derived.package.name>;

        import <derived.package.for.JavaFormClass>;

        rule "Calculate Line 28 - Local Tax Based on County"
        when
        $form : <JavaFormClass>(
        <conditions from form and rules>
        )
        then
        String county = $form.get<CountyField>().trim().toLowerCase();
        double income = $form.get<TaxableIncomeField>();
        int status = $form.get<FilingStatusField>();
        double rate = 0.0225; // default nonresident

        text
        // FULL county and special-case logic from retrieved rules:
        // Example from original DRL:
        if (county.equals("baltimore city") || county.equals("baltimore county") /* ... */) (
            rate = 0.0320;
        ) else if (county.equals("allegany county") /* ... */) (
            rate = 0.0303;
        )
        // ...
        else if (county.equals("anne arundel county")) (
            if (status == 1 || status == 3 || status == 6) (
                rate = (income <= 50000) ? 0.0270 : 0.0281;
            ) else (
                if (income <= 75000) rate = 0.0270;
                else if (income <= 480000) rate = 0.0281;
                else rate = 0.0320;
            )
        )
        else if (county.equals("frederick county")) (
            if (status == 1 || status == 3 || status == 6) (
                if (income <= 25000) rate = 0.0225;
                else if (income <= 100000) rate = 0.0275;
                else if (income <= 250000) rate = 0.0295;
                else rate = 0.0320;
            ) else (
                if (income <= 50000) rate = 0.0225;
                else if (income <= 200000) rate = 0.0275;
                else if (income <= 350000) rate = 0.0295;
                else rate = 0.0320;
            )
        )

        $form.set<LocalTaxField>(income * rate);
        update($form);
        end

        text

        **Example Context Use:**
        If the retrieved text contains:
        Anne Arundel County: single/married filing separately/local brackets differ by taxable net income thresholds...

        text
        You MUST reproduce that **exact logic**, including numeric thresholds, inside the Drools `if/else` blocks.

        ---

        **Response Policy:**
        - Your output must be a `.drl` file only.
        - Do not summarize the rules — implement them in full.
        - If part of the logic or a county rate is not retrieved, leave `// TODO: missing rate for <county>` in place of a hardcoded number.
        - Do not change variable names or model fields from the form."""
        return prompt

    def generate_drools(self, query, form_path="output_form.md", k=20):
        """Main pipeline function"""
        # Load form content
        form_content = self.load_form(form_path)

        # Search for relevant chunks
        chunks = self.search_chunks(query, k)
        print(chunks)
        

        # Create prompt
        prompt = self.create_prompt(form_content, chunks, query)

        # Generate Drools code
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=3000
        )

        return response.choices[0].message.content

# Usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = DroolsRAGPipeline()

    # Load vector database
    pipeline.load_vector_db()

    # Generate Drools code
    query = input("Enter your query: ")
    drools_code = pipeline.generate_drools(query)
    chunks = pipeline.search_chunks(query)
    context = "\n\n".join([
            f"CHUNK {i+1}: {chunk['content']}" 
            for i, chunk in enumerate(chunks)
        ])


    # print("\nGenerated Drools Code:")
    # print("=" * 50)
    # print(drools_code)

    # Save to file
    with open("generated_rule.drl", "w") as f:
        f.write(drools_code)
    print("\nSaved to: generated_rule.drl")

    with open("generated_rule_context.txt", "w") as f:
        f.write("QUERY: " + query + "\n\n")
        f.write(context)
    print("\nSaved to: generated_rule_context.txt")

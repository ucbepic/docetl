# DocETL VLDB Talk Outline - 15 Minutes

## Slide 1: Title Slide
- **Title:** Agentic Query Rewriting and Evaluation for Complex Document Processing
- **Authors:** Shreya Shankar (with "on the faculty job market!" callout), Tristan Chambers, Tarak Shah, Aditya G. Parameswaran, Eugene Wu
- **Affiliations:** UC Berkeley EECS, BIDS Police Records Access Project, Columbia University
- **Links:** docetl.org • ⭐ 2.8k GitHub stars

## Slide 2: Opening Question (with animation)
- **Initial state:** "How should people use AI to process and analyze data?"
- **Animation sequence:**
  1. Add "unstructured text" before "data" (with underline) → "How should people use AI to process and analyze unstructured text data?"
  2. Add "in complex ways" after "data" (with underline) → "How should people use AI to process and analyze unstructured text data in complex ways?"

## Slide 3: Complex, AI-Powered Data Processing
- **Title:** Complex, AI-Powered Data Processing
- **Bullet 1:** One way: semantic operators[1] and applications[2]
  - [1] Patel et al. "Semantic Operators and Their Optimization" https://www.vldb.org/pvldb/vol18/p4171-patel.pdf
  - [2] Liu et al. "Palimpzest: Optimizing AI-Powered Analytics with Declarative Query Processing" https://mail.vldb.org/cidrdb/papers/2025/p12-liu.pdf
- **Bullet 2:** Can be effective but discovering the "right" logical pipeline is too hard

## Slide 4: Discovering the "Right" Logical Pipeline
- **Title:** Discovering the "right" logical pipeline
- **Story:** Police misconduct team wanted to build a database of police officers and instances of misconduct they exhibited
  - Given heterogeneous corpus: police reports, court transcripts, charging documents, RAP sheets, etc.
- **Key point:** Misconduct is well-specified in California Penal Code Section 13510.8(b)
  - Lists 9 types of serious misconduct:
    1. Dishonesty (false statements, tampering with evidence, perjury)
    2. Abuse of power (intimidating witnesses, false arrests)
    3. Physical abuse (excessive force)
    4. Sexual assault
    5. Demonstrating bias
    6. Acts violating the law
    7. Law enforcement gang participation
    8. Failure to cooperate with investigations
    9. Failure to intercede when observing misconduct
- **Problem:** 
  - If you specify this in semantic operations (PZ's `sem_add_columns` or LOTUS's `sem_map`), no guarantee of accuracy
  - GPT-4.1 and Gemini-2.5-pro don't even get 50% recall of what humans can identify
  - LOTUS: system only as good as the most accurate model you specify
  - PZ: finds cheaper plan with optimizer but still only as accurate as champion model
  - All systems assume user submits pipeline with perfectly scoped operators that work well with champion model
  - For complex tasks (10+ page documents, multiple extractions + reasoning), discovering such a pipeline is HARD

## Slide 5: How Can We Discover More Accurate Pipelines?
- **Title:** How can we discover more accurate pipelines?
- **Option 1:** Hire an "AI engineer" - several months of trial and error
- **Example scenario:** Extracting clauses from legal contracts
  - Simple clauses: parties, document name, agreement date
  - Complex clauses: IP restrictions, covenant not to sue (not consistently worded or positioned)
  - ~50 clause types (CUAD benchmark has 41 types)
- **Engineer's trial and error process:**
  1. Try extracting all clauses in one LLM call → bad recall
  2. Try one LLM call per clause type → still fails for hard types
  3. Chunk document into pieces (LLMs get distracted with lots of data)
  4. Query each chunk: "Does this chunk contain clause X?"
  5. Problem: Multiple chunks may appear to contain the same clause
  6. Add reconciliation step: LLM to pick best match or merge
  7. Continue iterating based on intuitions about LLM capabilities...
- **Point:** Humans decompose tasks in discombobulating ways based on their priors/intuitions about what LLMs can/can't do

## Slide 6: Humans Decompose Tasks in Predictable Ways
- **Title:** Humans decompose tasks in predictable ways
- **Examples of decomposition patterns:**
  - `Map("extract 3 things: A, B, C")` → `Map("extract A")` → `Map("extract B")` → `Map("extract C")`
  - `Map("list all potential instances of misconduct according to California penal code")` → Split doc into chunks → Map → `Reduce("given all the instances...aggregate into one list of unique...")`
- **Key insight:** Humans understand the intent of operations & rewrite in ways to logically preserve the intent
- **Our approach:** Our optimizer mimics this process with _agents_

## Slide 7: [Transition to DocETL]
- DocETL operators & rewrite directives
- Optimizer algorithm
- [To be filled in later]

## Remaining slides structure:
- DocETL DSL & Operators
- Agentic Query Rewriting
- System Design & Execution
- Evaluation & Findings
- Case Studies & Demos
- Takeaways
- Thank You

## Talk timing notes (15 minutes):
- Slides 1-2: 1.5 minutes (opening, set the stage)
- Slides 3-4: 3 minutes (problem statement with concrete example)
- Slides 5-6: 3 minutes (how humans approach this)
- Slides 7-X: 7.5 minutes (DocETL solution, evaluation, case studies)
- Final slide: 0.5 minutes (wrap up)
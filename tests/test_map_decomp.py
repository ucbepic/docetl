import re
from typing import List, Tuple, Any
from motion.dataset import Dataset
from motion.operators import LLMMapper, build_only
from dotenv import load_dotenv
import os
from litellm import completion

load_dotenv()

MODEL = "gpt-4o-mini"  # Adjust based on your available models

book_titles = [
    "The Curious Incident of the Dog in the Night-Time",
    "The Guernsey Literary and Potato Peel Pie Society",
    "The Hundred-Year-Old Man Who Climbed Out of the Window and Disappeared",
    "The Particular Sadness of Lemon Cake: A Novel of Magical Realism",
    "The Unbearable Lightness of Being in a World of Heavy Consequences",
    "The Hitchhiker's Guide to the Galaxy: The Complete Trilogy of Five",
    "The Absolutely True Diary of a Part-Time Indian: A Journey of Self-Discovery",
    "The Perks of Being a Wallflower: Coming of Age in the Digital Era",
    "The Brief Wondrous Life of Oscar Wao: A Tale of Love and the Dominican Diaspora",
    "The Elegance of the Hedgehog: Philosophical Musings in a Parisian Apartment",
    "The Curious Case of Benjamin Button and Other Jazz Age Stories",
    "The Time Traveler's Wife: A Love Story Across Dimensions and Decades",
    "The Immortal Life of Henrietta Lacks: Science, Ethics, and Human Legacy",
    "The Thousand Autumns of Jacob de Zoet: A Historical Epic of Old Japan",
    "The Mysterious Affair at Styles: Hercule Poirot's First Case",
    "The Strange Case of Dr. Jekyll and Mr. Hyde: A Study in Duality",
    "The Catcher in the Rye: Holden Caulfield's Journey Through New York City",
    "The Adventures of Huckleberry Finn: A Boy's Journey Down the Mississippi",
    "The Picture of Dorian Gray: A Portrait of Vanity and Corruption",
    "The Wind-Up Bird Chronicle: A Surreal Quest in Modern Tokyo",
    "The Unlikely Pilgrimage of Harold Fry: A Journey of a Thousand Miles",
    "The Girl with the Dragon Tattoo: A Swedish Thriller of Secrets and Revenge",
    "The Curious Incident of the Siamese Cat in the Afternoon",
    "The Remarkable Journey of Coyote Sunrise: A Cross-Country Adventure",
    "The Inconvenient Indian: A Curious Account of Native People in North America",
    "The Subtle Art of Not Giving a F*ck: A Counterintuitive Approach to Living a Good Life",
    "The Loneliness of the Long-Distance Runner: Stories of Endurance and Solitude",
    "The Hitchhiker's Guide to the Galaxy: Don't Panic Edition",
    "The Neverending Story: A Tale of Fantasy and Imagination",
    "The Absolutely True Diary of a Part-Time Superhero",
    "The Hundred-Year-Old Secret Society of the Whispering Willows",
    "The Peculiar Circumstances Surrounding the Disappearance of Miss Finch",
    "The Extraordinary Adventures of Alfred Kropp: Keeper of King Arthur's Sword",
    "The Unexpected Inheritance of Inspector Chopra: A Baby Elephant Mystery",
    "The Curious Case of the Clockwork Man: A Burton & Swinburne Adventure",
    "The Sweetness at the Bottom of the Pie: A Flavia de Luce Mystery",
    "The Wonderful Story of Henry Sugar and Six More Tales of the Unexpected",
    "The Incredible Journey of Edward Tulane: A Toy Rabbit's Tale of Love",
    "The Astonishing Color of After: A Story of Grief, Love, and Magic",
    "The Inexplicable Logic of My Life: A Coming-of-Age Tale",
    "The Mysterious Benedict Society and the Perilous Journey: Book Two",
    "The Undomestic Goddess: A Hilarious Tale of Corporate Burnout and Rural Bliss",
    "The Absolutely True Story of a Part-Time Indian Princess",
    "The Strange and Beautiful Sorrows of Ava Lavender: A Magical Realism Novel",
    "The Gigantic Beard That Was Evil: A Graphic Novel About Conformity",
    "The Hundred-Year-Old Man Who Climbed Out of the Window and Ran for President",
    "The Curious Incident of the Cat in the Daytime: A Feline Mystery",
    "The Guernsey Literary and Potato Peel Pie Society: The Lost Chapters",
    "The Particular Sadness of Chocolate Cake: A Culinary Mystery",
    "The Unbearable Lightness of Being a Wallflower in the Digital Age",
    "The Time Traveler's Daughter: Navigating a Legacy Across Centuries",
    "The Immortal Life of Henrietta Lacks: The Next Generation",
    "The Thousand Winters of Jacob de Zoet: A Tale of Perseverance in Feudal Japan",
    "The Mysterious Affair at Sunset: Miss Marple's Final Case",
    "The Strange Case of the Alchemist's Daughter: Monsters and Mysteries in Victorian London",
    "The Catcher in the Wifi: Holden Caulfield Navigates the Internet Age",
    "The Adventures of Huckleberry Finn in Space: A Galactic Journey",
    "The Picture of Dorian Gray's Selfie: Vanity in the Age of Social Media",
    "The Wind-Up Girl Chronicle: A Tale of Bioengineered Futures",
    "The Unlikely Pilgrimage of Harold Fry's Grandson: Retracing Steps",
    "The Girl with the Quantum Tattoo: A Sci-Fi Thriller of Alternate Realities",
    "The Remarkable Journey of Coyote Sunrise: The Road Trip Continues",
    "The Inconvenient Truth About Time Travel: A Scientific and Philosophical Exploration",
    "The Subtle Art of Giving Too Many F*cks: An Overachiever's Guide to Burnout",
    "The Loneliness of the Long-Distance Spaceship Captain: Tales from the Void",
    "The Neverending Story: The Lost Chapters of Fantastica",
    "The Extraordinary Adventures of Ordinary Alice: A Wonderland Reimagined",
    "The Unexpected Inheritance of Inspector Gadget: A Robotic Mystery",
    "The Curious Case of the Quantum Cat: Schrödinger's Detective Story",
    "The Sweetness at the Bottom of the Black Hole: An Astrophysicist's Memoir",
    "The Wonderful World of Whales: Secrets of the Cetacean Society",
    "The Incredible Journey of the Last Dodo: A Tale of Extinction and Hope",
    "The Astonishing Color of Invisible Light: Synesthesia in the Quantum Realm",
    "The Inexplicable Logic of Cats: A Feline Philosopher's Guide to Life",
    "The Mysterious Benedict Society and the Riddle of the Multiverse",
    "The Undomestic Android: A Comedy of Errors in the Smart Home Era",
    "The Absolutely True Story of a Part-Time Vampire Slayer's Apprentice",
    "The Strange and Beautiful Sorrows of the Last Unicorn on Earth",
    "The Gigantic Beard That Saved the World: An Eco-Fable",
    "The Hundred-Year-Old Algorithm That Escaped the Mainframe",
    "The Curious Incident of the Ghost in the Machine: A Digital Haunting",
    "The Literary and Logarithmic Society: Math Nerds Save the Library",
    "The Particular Joy of Homemade Ice Cream: A Culinary Love Story",
    "The Unbearable Lightness of Being a Cloud in a World of Big Data",
    "The Time Traveler's Pet: A Dog's Journey Through Human History",
    "The Immortal Life of Artificial Intelligence: From Turing to Transcendence",
    "The Thousand Moons of Europa: A Space Opera of Jovian Proportions",
    "The Mysterious Affair at the Quantum Cafe: Agatha Christie Meets Schrödinger",
    "The Strange Case of the Disappearing Bees: An Ecological Mystery",
    "The Catcher in the Virtual Rye: Holden Caulfield's Digital Descendants",
    "The Adventures of Sherlock Holmes in the 25th Century",
    "The Picture of Dorian Gray's Clone: Ethics in the Age of Genetic Engineering",
    "The Wind-Up Universe Chronicle: Cosmic Tales of Entropy and Renewal",
    "The Unlikely Pilgrimage of the Last Human on Earth",
    "The Girl with the Bioengineered Brain: A Neuro-Noir Thriller",
    "The Remarkable Journey to the Center of the Atom: A Subatomic Odyssey",
    "The Inconvenient Yeti: Cryptozoology in the Age of Climate Change",
    "The Subtle Art of Embracing Chaos: A Mathematician's Guide to Life",
    "The Loneliness of the Interplanetary Postal Worker: Cosmic Mail Tales",
    "The Neverending Story of the Expanding Universe: A Cosmological Epic",
]

# Create initial dataset
dataset = Dataset([(title, 1) for title in book_titles])


class BookTitleAnalyzer(LLMMapper):
    def generate_prompt(self, key: str, value: int) -> list:
        return [
            {
                "role": "system",
                "content": "You are a literary analyst with expertise in linguistics, creative writing, and market trends.",
            },
            {
                "role": "user",
                "content": f"""Perform an in-depth analysis of the following book title:

Title: {key}

Provide the following information:
1. Linguistic Analysis:
   a) Number of words
   b) Syllable count (estimate)
   c) Presence of alliteration or assonance
   d) Identify any archaic or unusual words

2. Thematic Elements:
   a) Primary theme(s) suggested by the title
   b) Emotional tone (e.g., mysterious, uplifting, melancholic)
   c) Any cultural or historical references implied

3. Genre and Market Appeal:
   a) Primary and secondary genre predictions
   b) Target age group
   c) Estimate potential market appeal (1-10 scale)

4. Creative Writing Aspects:
   a) Suggest a brief (2-3 sentence) plot synopsis based solely on the title
   b) Propose a potential opening line for the book

5. Comparative Analysis:
   a) Suggest two existing famous books with similar titles or themes
   b) Explain how this title might stand out or blend in with current market trends

Provide your analysis in a structured format, using the numbering system above.""",
            },
        ]

    def process_response(self, response: Any, **prompt_kwargs) -> Tuple[str, dict]:
        content = response.choices[0].message.content.strip()
        # We'll keep the raw response as is, given its complex structure
        return (prompt_kwargs["key"], content)

    @build_only
    def validate(
        self, input_key: str, input_value: int, output_key: str, output_value: str
    ) -> None:
        # Use LLM for content quality check
        validation_prompt = f"""
        Analyze the following book title analysis for quality and completeness:

        Title: {input_key}
        Analysis:
        {output_value}

        Please verify:
        1. Is the linguistic analysis accurate and complete?
        2. Are the thematic elements well-identified and relevant?
        3. Is the genre prediction reasonable, and is the market appeal justified?
        4. Is the plot synopsis creative and relevant to the title?
        5. Are the comparative analysis and market trend assessment insightful?

        Respond with YES if all criteria are met satisfactorily, or NO if any aspect is lacking or could be significantly improved, with a brief explanation.
        """

        response = completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": validation_prompt}],
        )

        validation_result = response.choices[0].message.content.strip()
        if "NO" in validation_result:
            raise ValueError(f"LLM Validation Failed: {validation_result}")


# Apply operation
if __name__ == "__main__":
    pipeline = dataset.map(BookTitleAnalyzer(model=MODEL))

    pipeline = pipeline.build(sample_size=15)
    # result = pipeline.execute()

    # # Print results
    # for title, analysis in result:
    #     print(f"\nTitle: {title}")
    #     print("Analysis:")
    #     print(analysis)
    #     print("-" * 70)

    # # Print the number of titles analyzed
    # print(f"Number of titles analyzed: {len(result)}")

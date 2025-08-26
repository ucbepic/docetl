import json
import random
from typing import List, Dict

def fetch_amazon_reviews_sample() -> List[Dict]:
    """
    Fetch a sample of Amazon product reviews from a public dataset.
    We'll use the Amazon Customer Reviews dataset samples.
    """
    
    # Sample review data - in a real scenario, you might scrape or use an API
    # For this demo, I'll create realistic sample reviews for various product categories
    
    categories = [
        "Electronics", "Books", "Home & Kitchen", "Toys & Games", 
        "Health & Personal Care", "Sports & Outdoors", "Clothing",
        "Beauty", "Automotive", "Pet Supplies"
    ]
    
    # Generate sample reviews with realistic patterns
    sample_reviews = []
    
    review_templates = [
        {
            "positive": [
                "This {product} exceeded my expectations! The quality is outstanding and it arrived quickly.",
                "I'm extremely satisfied with this {product}. It works exactly as described and the price was great.",
                "Fantastic {product}! I've been using it for weeks now and couldn't be happier with my purchase.",
                "The {product} is amazing. Great build quality and excellent customer service from the seller.",
                "Highly recommend this {product}. It's exactly what I was looking for and the value is unbeatable."
            ],
            "negative": [
                "Very disappointed with this {product}. It broke after just a few days of use.",
                "The {product} didn't match the description at all. Poor quality and overpriced.",
                "Waste of money. The {product} stopped working within a week and the return process was difficult.",
                "Don't buy this {product}. Cheap materials and terrible performance.",
                "The {product} arrived damaged and doesn't work properly. Very frustrating experience."
            ],
            "neutral": [
                "The {product} is okay. It does what it's supposed to but nothing special.",
                "Average {product}. Works fine but there are probably better options out there.",
                "The {product} is decent for the price. Some features work well, others not so much.",
                "It's a basic {product} that gets the job done. No major complaints but not impressed either.",
                "The {product} meets expectations but doesn't exceed them. Fair value for money."
            ]
        }
    ]
    
    products = {
        "Electronics": ["wireless headphones", "smartphone", "laptop", "tablet", "smart watch", "bluetooth speaker"],
        "Books": ["novel", "cookbook", "self-help book", "textbook", "children's book", "biography"],
        "Home & Kitchen": ["coffee maker", "blender", "vacuum cleaner", "air fryer", "instant pot", "knife set"],
        "Toys & Games": ["board game", "LEGO set", "action figure", "puzzle", "remote control car", "doll"],
        "Health & Personal Care": ["vitamin supplement", "electric toothbrush", "fitness tracker", "face cream", "shampoo", "protein powder"],
        "Sports & Outdoors": ["yoga mat", "dumbbells", "camping tent", "hiking backpack", "running shoes", "bike helmet"],
        "Clothing": ["t-shirt", "jeans", "dress", "jacket", "sneakers", "sweater"],
        "Beauty": ["lipstick", "foundation", "eye shadow palette", "perfume", "hair straightener", "nail polish"],
        "Automotive": ["car phone mount", "tire inflator", "dash cam", "car vacuum", "jump starter", "seat covers"],
        "Pet Supplies": ["dog food", "cat toy", "pet bed", "leash", "aquarium filter", "bird cage"]
    }
    
    # Generate 200 reviews
    for i in range(200):
        category = random.choice(categories)
        product = random.choice(products[category])
        
        # Weighted sentiment distribution: 60% positive, 25% negative, 15% neutral
        sentiment_choice = random.choices(
            ['positive', 'negative', 'neutral'],
            weights=[0.6, 0.25, 0.15],
            k=1
        )[0]
        
        review_text = random.choice(review_templates[0][sentiment_choice]).format(product=product)
        
        # Add some variation to reviews
        if random.random() > 0.7:
            additional_comments = [
                " Shipping was fast.",
                " Great packaging.",
                " Would buy again!",
                " Customer service was helpful.",
                " Price has gone up since I bought it.",
                " Comes with clear instructions.",
                " Battery life could be better.",
                " Perfect gift idea!",
                " Size runs small/large.",
                " Easy to set up and use."
            ]
            review_text += " " + random.choice(additional_comments)
        
        # Generate realistic metadata
        review = {
            "review_id": f"R{i+1:04d}",
            "product_id": f"B{random.randint(1000, 9999)}{random.choice(['XYZ', 'ABC', 'DEF', 'GHI'])}",
            "product_name": f"{product.title()} - {random.choice(['Premium', 'Standard', 'Deluxe', 'Basic', 'Pro'])} Edition",
            "category": category,
            "rating": {
                'positive': random.choice([4, 5, 5, 5]),
                'negative': random.choice([1, 1, 2]),
                'neutral': random.choice([3, 3, 4])
            }[sentiment_choice],
            "review_title": review_text.split('.')[0][:50] + "...",
            "review_text": review_text,
            "reviewer_name": f"Customer{random.randint(100, 999)}",
            "review_date": f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            "verified_purchase": random.choice([True, True, True, False]),  # 75% verified
            "helpful_votes": random.randint(0, 150),
            "total_votes": random.randint(0, 200)
        }
        
        sample_reviews.append(review)
    
    return sample_reviews

def save_reviews_to_file(reviews: List[Dict], output_path: str):
    """Save reviews to a JSON file for DocETL processing."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(reviews)} reviews to {output_path}")

def main():
    print("Collecting Amazon product reviews...")
    
    # Fetch reviews
    reviews = fetch_amazon_reviews_sample()
    
    # Save to data directory
    output_path = "/workspace/amazon-reviews-pipeline/data/amazon_reviews.json"
    save_reviews_to_file(reviews, output_path)
    
    # Print some statistics
    print(f"\nDataset Statistics:")
    print(f"Total reviews: {len(reviews)}")
    
    # Calculate rating distribution
    ratings = {}
    for review in reviews:
        rating = review['rating']
        ratings[rating] = ratings.get(rating, 0) + 1
    
    print("\nRating Distribution:")
    for rating in sorted(ratings.keys()):
        print(f"  {rating} stars: {ratings[rating]} reviews ({ratings[rating]/len(reviews)*100:.1f}%)")
    
    # Category distribution
    categories = {}
    for review in reviews:
        cat = review['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nCategory Distribution:")
    for cat in sorted(categories.keys()):
        print(f"  {cat}: {categories[cat]} reviews")

if __name__ == "__main__":
    main()
import requests
from bs4 import BeautifulSoup
import re

def extract_overview_figure(arxiv_id):
    url = f"https://ar5iv.org/html/{arxiv_id}"
    print(f"Fetching URL: {url}")
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Failed to fetch content. Status code: {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # ar5iv typically uses <figure> tags
        figures = soup.find_all('figure')
        print(f"Found {len(figures)} figures")
        
        score_map = {}
        
        for fig in figures:
            # Check caption
            caption = fig.find('figcaption')
            caption_text = caption.get_text().lower() if caption else ""
            
            # Find image source
            img = fig.find('img')
            if not img:
                continue
                
            img_src = img.get('src')
            if not img_src:
                continue
                
            # Construct full URL (ar5iv images are relative)
            if img_src.startswith('/'):
                 full_img_url = f"https://ar5iv.org{img_src}"
            else:
                 # Usually they are like "S1.F1.large.jpg" relative to the html path
                 # https://ar5iv.org/html/1706.03762 -> https://ar5iv.org/html/1706.03762/assets/x.png
                 # Actually ar5iv usually rewrites them to absolute or relative. 
                 # Let's check what we get.
                 full_img_url = f"https://ar5iv.org/html/{arxiv_id}/{img_src}"

            # Calculate score based on keywords
            score = 0
            keywords = ['overview', 'architecture', 'model', 'framework', 'pipeline', 'schematic', 'figure 1', 'fig. 1']
            
            for word in keywords:
                if word in caption_text:
                    score += 1
            
            # Prioritize "Figure 1" or "Fig 1" often contains the overview
            if "figure 1" in caption_text or "fig. 1" in caption_text:
                score += 2
                
            if score > 0:
                score_map[full_img_url] = {
                    'score': score,
                    'caption': caption_text,
                    'src': full_img_url
                }
                print(f"Candidate found (Score {score}): {caption_text[:50]}... -> {full_img_url}")

        if not score_map:
            print("No suitable overview figure found.")
            return None
            
        # Return best match
        best_img = max(score_map.values(), key=lambda x: x['score'])
        return best_img

    except Exception as e:
        print(f"Error during extraction: {e}")
        return None

# Test with "Attention Is All You Need"
arxiv_id = "1706.03762"
print(f"Testing extraction for {arxiv_id}...")
result = extract_overview_figure(arxiv_id)

if result:
    print("\n✅ Best Match:")
    print(f"Image URL: {result['src']}")
    print(f"Caption: {result['caption']}")
else:
    print("\n❌ Failed to find a figure.")

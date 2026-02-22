import urllib.request
import json
import urllib.parse
import time

queries = [
    "physics-informed neural networks sequence to sequence",
    "physics-informed neural networks long time integration"
]

for q in queries:
    print(f"\n--- Searching: {q} ---")
    url = f'https://api.semanticscholar.org/graph/v1/paper/search?query={urllib.parse.quote(q)}&limit=3&fields=title,authors,year,abstract,citationCount'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        data = json.loads(response.read())
        for paper in data.get('data', []):
            authors = [a['name'] for a in paper.get('authors', [])]
            print(f"Title: {paper.get('title')}")
            print(f"Authors: {', '.join(authors)}")
            print(f"Year: {paper.get('year')} | Citations: {paper.get('citationCount')}")
            abstract = paper.get('abstract') or ''
            print(f"Abstract: {abstract[:300]}...\n")
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(2)


import urllib.request
import xml.etree.ElementTree as ET

url = 'http://export.arxiv.org/api/query?search_query=all:physics-informed+AND+all:Regge-Wheeler&start=0&max_results=5'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    response = urllib.request.urlopen(req)
    xml_data = response.read()
    root = ET.fromstring(xml_data)
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text.replace('\n', ' ')
        authors = [a.find('{http://www.w3.org/2005/Atom}name').text for a in entry.findall('{http://www.w3.org/2005/Atom}author')]
        print(f"Title: {title}")
        print(f"Authors: {', '.join(authors)}\n")
except Exception as e:
    print(f"Error: {e}")

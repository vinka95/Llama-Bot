import urllib.request
import xml.etree.ElementTree as ET
import faiss
import numpy as np
import json


def fetch_papers():
    url = 'http://export.arxiv.org/api/query?search_query=ti:llama&start=0&max_results=70'
    response = urllib.request.urlopen(url)
    data = response.read().decode('utf-8')
    root = ET.fromstring(data)

    papers_list = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
        papers_list.append({"title": title, "summary": summary})

    return papers_list


def save_papers_to_file(papers, filename="papers.txt"):
    with open(filename, 'w', encoding='utf-8') as file:
        for paper in papers:
            file.write(f"Title: {paper['title']}\n")
            file.write(f"Summary: {paper['summary']}\n")
            file.write("\n")  # Add an empty line for better readability
            
            
def load_papers_from_file(filename="papers.txt"):
    
    papers_file = open(filename, "r")
    papers_data = papers_file.read()
            
    return papers_data
            
            
def generate_embeddings(papers, openai_embeddings):
    
    embeddings = []
    
    for paper in papers:
        embedding = openai_embeddings.embed_query(paper['title'] + paper['summary'])
        embeddings.append(embedding)
    return embeddings


def create_faiss_index(embeddings):
    dimension = len(embeddings[0])  # Get the dimension of the embeddings
    index = faiss.IndexFlatL2(dimension)  # Use IndexFlatL2 for L2 distance (Euclidean)
    
    # FAISS expects numpy arrays in float32
    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)  # Add embeddings to the index

    return index




# Fetch and print a sample of the papers
papers = fetch_papers()


# Print a confirmation message
print("Papers saved to 'papers_data.json'")



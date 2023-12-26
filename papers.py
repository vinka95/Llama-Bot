import urllib.request
import xml.etree.ElementTree as ET


def fetch_papers():
    """Method to fetch SOTA papers on Llama from Arxiv
    """
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
    """Saves paper text data to text file for further processing
    """
    with open(filename, 'w', encoding='utf-8') as file:
        for paper in papers:
            file.write(f"Title: {paper['title']}\n")
            file.write(f"Summary: {paper['summary']}\n")
            file.write("\n")  # Add an empty line for better readability
            
            
def load_papers_from_file(filename="papers.txt"):
    """ Loads text data """
    papers_file = open(filename, "r")
    papers_data = papers_file.read()
            
    return papers_data


# Fetch and print a sample of the papers
papers = fetch_papers()

# Print a confirmation message
print("Papers saved to 'papers_data.json'")
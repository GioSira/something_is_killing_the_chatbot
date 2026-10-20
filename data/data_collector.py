from typing import List

import os
import re
import time
import requests
import random
from tqdm import tqdm
from bs4 import BeautifulSoup



class RetrieveError(Exception):
    def __init__(self, message, status_code=None):
        super.__init__(*args, **kwargs)
        self.message = message


url = "https://something-is-killing-the-children.fandom.com/api.php"
# Un User-Agent più "simile a un browser"
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
        " like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

visited_pages = set()


def _request_exp_backoff(params,
                         max_retries=10, 
                         initial_delay=1, 
                         backoff_factor=2):

    delay = initial_delay

    for attempt in range(max_retries):
        
        try:
            
            res = requests.get(url, params=params, headers=headers, timeout=10)
            
            # Se il server risponde con status di blocco/errore temporaneo
            if res.status_code in {400, 403, 429, 500, 502, 503, 504}:
                print(
                    f"⚠️ Server ha restituito status {res.status_code} per la"
                    f" pagina. Tentativo {attempt + 1}/{max_retries}..."
                )

            if res.status_code == 200:
                return res

        except Exception:
            if attempt == max_retries -1:
                raise

            sleep_time = delay + random.uniform(0, 0.1)
            time.sleep(sleep_time)
            delay *= backoff_factor

    return None


def get_category_member(category_title: str) -> List[str]:

    members = []
    cmcontinue = None

    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": f"Category:{category_title}",
            "cmlimit": "max",
        }

        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        res = _request_exp_backoff(params)
        if not res:
            print(f"Skipping Category: {category_title}")
            continue

        res = res.json()
        
        members.extend(res.get("query", {}).get("categorymembers", []))

        if "continue" in res:
            cmcontinue = res["continue"]["cmcontinue"]
        else:
            break

    return members


def get_all_categories():

    categories = []
    accontinue = None

    print("🔄 Recupero di tutte le categorie dalla Wiki...")

    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "allcategories",
            "aclimit": "max",
        }

        if accontinue:
            params["accontinue"] = accontinue

        res = _request_exp_backoff(params)
        if not res:
            raise RetrieveError("Failed to retrieve all categories.")
        
        res = res.json()
        cats = res.get("query", {}).get("allcategories", [])

        for cat in cats:
            # Il nome della categoria si trova all'interno del campo chiave '*'
            categories.append(cat["*"]) 

        # Gestione della paginazione se ci sono più di 500 categorie
        if "continue" in res:
            accontinue = res["continue"]["accontinue"]
        else:
            break

    return categories


def get_page_json(page_title: str):

    params = {"action": "parse", "format": "json", "prop": "text", "page": page_title}

    response = _request_exp_backoff(params)
    if not response:
        raise RetrieveError(f"Failed to retrieve page {page_title}")
    
    json_response = response.json()
    return json_response.get("parse", {}).get("text", {}).get("*", "")


def _clean_soup_text(soup: BeautifulSoup) -> str:
  
    # 1. Rimuove elementi inutili (tabelle, box laterali, script, note a piè di pagina)
    for tag in soup.find_all(
        [
            "script",
            "style",
            "table",
            "div",
            "aside",
            "sup",
            "span",
        ],
        class_=[
            "navbox",
            "toc",
            "portal",
            "reference",
            "mw-editsection",
            "infobox",
        ],
    ):
        tag.decompose()

    # 2. UNWRAP dei link: toglie i tag <a> ma conserva il testo all'interno,
    # evitando che spezzino le frasi.
    for a_tag in soup.find_all("a"):
        a_tag.unwrap()

    # 3. Estrae il testo usando lo spazio come separatore invece del newline
    text = soup.get_text(separator=" ")

    # 4. Pulisce gli spazi multipli e le linee vuote eccessive con le Regular Expressions
    text = re.sub(
        r"[ \t]+", " ", text
    )  # Sostituisce spazi multipli con uno singolo
    text = re.sub(r"\n\s*\n", "\n\n", text)  # Pulisce i ritorni a capo doppi

    return text.strip()

def download_page(folder: str, page_title: str):

    if page_title in visited_pages:
        return

    visited_pages.add(page_title)

    html_page = get_page_json(page_title)
    soup = BeautifulSoup(html_page, "html.parser")

    text = _clean_soup_text(soup)

    # Salva il file pulendo il nome del titolo
    safe_filename = (
        "".join(c for c in page_title if c.isalnum() or c in (" ", "_", "-"))
        .replace(" ", "_")
        .rstrip()
        .lower()
    )

    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"{safe_filename}.md")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {page_title}\n\n{text}")


def crawl_all(base_dir: str = "./collection/siktc"):

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    all_categories = get_all_categories()
    for category in tqdm(all_categories, desc="Categories", total=len(all_categories)):
        crawl_category(base_dir, category)


def crawl_category(folder: str, category_name: str):

    members = get_category_member(category_name)

    for member in tqdm(members, desc=category_name, total=len(members)):
        title = member["title"]
        ns = member["ns"]

        if ns == 0:
            download_page(folder, title)
        else:
            sub_cat = os.path.join(folder, title.replace("Category:", "").replace(" ", "_"))
            if not os.path.exists(sub_cat):
                os.makedirs(sub_cat)
            crawl_category(sub_cat, title)
        
        time.sleep(0.15)


if __name__ == "__main__":
    crawl_all()

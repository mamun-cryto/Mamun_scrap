import requests
from bs4 import BeautifulSoup
import pandas as pd
from huggingface_hub import HfApi, login

# Hugging Face Credentials
HF_USERNAME = "almamunkhan"  # তোমার Hugging Face ইউজারনেম
HF_REPO_NAME = "Mamun_Scrap"  # তোমার ডেটাসেট রিপোজিটরি নাম

# Hugging Face API Token দিয়ে লগইন করা
login(token="your_hugging_face_token_here")

# ওয়েবসাইট স্ক্র্যাপ ফাংশন
def scrape_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # পেজের শিরোনাম
        title = soup.title.string if soup.title else "No Title"
        
        # পেজের মূল লেখা (প্যারাগ্রাফ)
        content = ' '.join([p.text for p in soup.find_all('p')])
        
        # নিউজ হেডলাইনগুলি
        headlines = [h.text for h in soup.find_all(['h1', 'h2', 'h3'])]

        # সমস্ত লিঙ্ক সংগ্রহ
        links = [a['href'] for a in soup.find_all('a', href=True)]
        
        # ইমেজ লিঙ্ক সংগ্রহ
        images = [img['src'] for img in soup.find_all('img', src=True)]

        return {
            "URL": url,
            "Title": title,
            "Content": content,
            "Headlines": headlines,
            "Links": links,
            "Images": images
        }
    else:
        return None

# Jamuna.tv স্ক্র্যাপ করা
url = "https://jamuna.tv/"
data = [scrape_website(url)]  # একাধিক পেজ স্ক্র্যাপ করতে চাইলে লুপ ব্যবহার করা যাবে

# DataFrame তৈরি করা
df = pd.DataFrame(data)

# CSV ফাইল হিসেবে সংরক্ষণ
csv_filename = "jamuna_tv_full_data.csv"
df.to_csv(csv_filename, index=False)
print(f"✅ Data saved as {csv_filename}")

# Hugging Face-এ ফাইল আপলোড করা
api = HfApi()
api.upload_file(
    path_or_fileobj=csv_filename,
    path_in_repo=f"{csv_filename}",
    repo_id=f"{HF_USERNAME}/{HF_REPO_NAME}",
    repo_type="dataset",
)

print(f"🚀 File uploaded to Hugging Face: https://huggingface.co/datasets/{HF_USERNAME}/{HF_REPO_NAME}")

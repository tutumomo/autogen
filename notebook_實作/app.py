# for 08.function call.ipynb 建立的 function call 範例
"""
import requests 和 from bs4 import BeautifulSoup：這些是 Python 庫，用於發送 HTTP 請求和解析 HTML 內容。
URL = "https://catalog.data.gov/dataset"：這是要訪問的網頁的 URL。
def list_datasets(url):：定義了一個函數 list_datasets，它接受一個參數 url，用於指定要提取資料集的網頁 URL。
response = requests.get(url)：發送一個 HTTP GET 請求來獲取指定 URL 的內容，並將響應存儲在變數 response 中。
soup = BeautifulSoup(response.text, 'html.parser')：使用 BeautifulSoup 將獲得的 HTML 內容解析為可操作的 Python 物件，存儲在 soup 中。
result = ""：初始化一個空字串，用於存儲提取的資料集標題和連結。
for dataset in soup.find_all('h3', class_='dataset-heading'):：遍歷所有標籤名為 'h3'，且 class 為 'dataset-heading' 的元素。這些元素通常包含資料集的標題。
a_tag = dataset.find('a')：在每個資料集標題元素中尋找第一個 'a' 標籤，這是資料集標題的超鏈接。
title = a_tag.text.strip()：從 'a' 標籤中獲取文本，並使用 strip() 方法去除前後的空白字符。
link = a_tag['href']：從 'a' 標籤中獲取 'href' 屬性的值，即超鏈接的 URL。
result = result + f"\nTitle: {title}\nLink: {link}\n"：將每個資料集的標題和連結格式化為一個字串，並添加到結果字串中。
return result：返回包含所有資料集標題和連結的字串。
總的來說，這段程式碼通過爬取指定網頁上的 HTML 內容，解析出資料集的標題和連結，最後返回這些資訊的格式化字串。

能否再詳細說明"soup = BeautifulSoup(response.text, 'html.parser')"
當我們使用 requests.get(url) 從網頁中獲取 HTML 內容後，我們需要將這個 HTML 內容轉換為一個 Python 可以操作的結構。這就是 BeautifulSoup 出現的地方。
這行程式碼 soup = BeautifulSoup(response.text, 'html.parser') 實際上創建了一個 BeautifulSoup 物件，用於解析 HTML 內容。讓我們進一步解釋它的作用：
response.text：response 物件的 .text 屬性包含了從網頁獲取的原始 HTML 內容。這裡我們將它傳遞給 BeautifulSoup，以便解析它。
'html.parser'：這是 BeautifulSoup 的一個解析器，用於解析 HTML 內容並構建對應的 Python 物件樹。在這裡，我們使用了預設的 HTML 解析器，即 'html.parser'。這個解析器通常被用於解析 HTML 內容，並且在大多數情況下表現良好。
當這行程式碼執行完畢後，soup 變數就會成為一個 BeautifulSoup 物件，我們可以使用它來方便地搜索、遍歷和操作 HTML 內容。在這段程式碼的後續部分中，我們使用 soup.find_all() 方法來查找所有標籤，以及使用 soup.find() 方法來查找單個標籤，從而獲取我們需要的資料集標題和連結。
"""
import requests
from bs4 import BeautifulSoup

URL = "https://catalog.data.gov/dataset"

def list_datasets(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    result = ""

    for dataset in soup.find_all('h3', class_='dataset-heading'):
        a_tag = dataset.find('a')
        title = a_tag.text.strip()
        link = a_tag['href']
        result = result + f"\nTitle: {title}\nLink: {link}\n"
    return result
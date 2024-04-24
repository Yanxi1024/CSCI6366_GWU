import requests
from lxml import etree as et
from loguru import logger

import PicDownloader

START_URL = "https://www.bing.com/images/search?view=detailV2&ccid=asnc%2F8Y%2B&id=95A022C759C7B2ACE541ED329308AB9B3B87315E&thid=OIP.asnc!_8Y-ZNgLVM0thkfLAgHaE9&mediaurl=https%3A%2F%2Fpassnownow.com%2Fwp-content%2Fuploads%2F2015%2F06%2Fduck.jpg&exph=1339&expw=2000&q=duck+photos&simid=607996305554550567&form=IRPRST&ck=B896F010181C8D56B19A9C634446BE0E&selectedindex=10&itb=0&ajaxhist=0&ajaxserp=0&vt=0&sim=11&cdnurl=https%3A%2F%2Fth.bing.com%2Fth%2Fid%2FR.6ac9dcffc63e64d80b54cd2d8647cb02%3Frik%3DXjGHO5urCJMy7Q%26pid%3DImgRaw%26r%3D0"
PIC_NUM = 100
START_NAME = 1

#Config
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36"}

def collectPic():
    url = START_URL
    counter = START_NAME

    try:
        response = requests.request(url = url, method = "GET", headers = headers, timeout = 10)
    except:
        logger.error(f"request faild >> url: {url}")
    else:
        # print(response.text)
        root = et.HTML(response.text)
        # print(response.text)
        img_path = root.xpath("./body//img[@src]/@src")
        
        PicDownloader.downloadImage(img_path[0], counter)
        logger.debug(f"url: {img_path[0]}")



if __name__ == "__main__":
    collectPic()
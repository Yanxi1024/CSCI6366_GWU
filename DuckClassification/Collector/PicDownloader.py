import requests

def downloadImage(image_url, counter):
    image_file_path = f"./RawData/Duck{counter}.jpg"
    # 发送GET请求
    response = requests.get(image_url)
    
    # 检查请求是否成功
    if response.status_code == 200:
        # 打开文件以写入二进制模式
        with open(image_file_path, 'wb') as file:
            # 将响应内容（图片数据）写入文件
            file.write(response.content)
        print(f'Image successfully downloaded: {image_file_path}')
    else:
        print(f'Failed to download image: {response.status_code}')

if __name__ == "__main__":
    image_url = 'https://hawaiibirdingtrails.hawaii.gov/wp-content/uploads/Muscovy-Duck-female_Michelle-Moore.jpg'
    image_file_path = 'downloaded_image.jpg'
    downloadImage(image_url, 1)

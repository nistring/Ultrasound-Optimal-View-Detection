"""
This program automatically downloads all the files of a directory from surgstory
"""
from unicodedata import name
from selenium import webdriver
import time
import os
import base64
import numpy as np
from webdriver_manager.microsoft import EdgeChromiumDriverManager


channel = None
category = None #'test set'

# Fill out the ID & PW to login
print('Login')
ID = input('ID : ')
PW = input('Password : ')

# Open surgstory
browser = webdriver.Edge(EdgeChromiumDriverManager().install())
browser.get("https://www.surgstory.com/v2/login")

# Send ID & PW then click the login button
browser.find_element("xpath", "/html/body/div[1]/div[1]/div[1]/div/div[2]/form/div[1]/div/input").send_keys(ID)
browser.find_element("xpath", "/html/body/div[1]/div[1]/div[1]/div/div[2]/form/div[2]/div/input").send_keys(PW)
browser.find_element("xpath", "/html/body/div[1]/div[1]/div[1]/div/div[2]/form/div[4]/div/span").click()
browser.implicitly_wait(3)

# Select the right channel
browser.find_element("xpath", f'//*[@title="{channel}"]').click()
browser.find_element("xpath", "//html/body/div[1]/div[1]/div[1]/div/div[1]/div[2]/div/span").click()
browser.implicitly_wait(3)
time.sleep(1)
# Move to the directory of interest
browser.get('https://www.surgstory.com/v2/drive?folder=3622')
browser.implicitly_wait(3)
time_list = np.zeros(600)

for i in range(600):
    n = browser.find_elements("xpath", f'//*[@class="ellipsis max_100p hover_point pr_5"]')[i]
    n.click()
    browser.implicitly_wait(3)
    browser.find_element("xpath", '//*[@class="row time"]').click()
    browser.implicitly_wait(3)
    time_text = browser.find_elements_by_class_name("time_text")[-1].text[-8:]
    time_list[599-i] = int(time_text.replace(':',''))
    canvas = browser.find_elements_by_class_name("annotation_canvas")#_hit")
    #canvas = browser.find_element_by_css_selector("#canvas")

    for canva, category in zip(canvas[:3], ['artery', 'rib', 'nerve']):
        # get the canvas as a PNG base64 string
        canvas_base64 = browser.execute_script("return arguments[0].toDataURL('image/png').substring(21);", canva)

        # decode
        canvas_png = base64.b64decode(canvas_base64)

        # save to a file
        with open(f"{category}/{600-i}.png", 'wb') as f:
            f.write(canvas_png)

    if len(canvas) > 3:
        for m, canva in enumerate(canvas[3:]):
            canvas_base64 = browser.execute_script("return arguments[0].toDataURL('image/png').substring(21);", canva)

            # decode
            canvas_png = base64.b64decode(canvas_base64)

            # save to a file
            with open(f"etc/{600-i}_{m}.png", 'wb') as f:
                f.write(canvas_png)
    
    browser.get('https://www.surgstory.com/v2/drive?folder=3622')
    browser.implicitly_wait(3)
    time.sleep(1)
# cur_dir = os.path.dirname(os.path.abspath(__file__))
# save_dir = os.path.join(cur_dir, 'videos')

# Get the list of files to be downloaded

# No = browser.find_elements("xpath", '//*[@class="download_btn"]')
# for n in No:
#     n.click()
#     time.sleep(1)

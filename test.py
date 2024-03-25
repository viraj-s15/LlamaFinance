from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

def test_endpoint():
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get("http://localhost:8000/test")
    success_message = driver.find_element(By.XPATH, '//pre').text
    assert success_message == '{"message":"Initialization successful"}'
    driver.quit()

if __name__ == "__main__":
    test_endpoint()
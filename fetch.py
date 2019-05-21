import urllib
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import StaleElementReferenceException



def main():

    driver = webdriver.Remote(
        desired_capabilities=DesiredCapabilities.CHROME
    )

    # page = requests.get('https://www.avito.ru/respublika_krym/zemelnye_uchastki/prodam/izhs?q=%D0%91%D0%B5%D1%80%D0%B5%D0%B3%D0%BE%D0%B2%D0%BE%D0%B5%20%D0%A4%D0%B5%D0%BE%D0%B4%D0%BE%D1%81%D0%B8%D1%8F')
    # soup = BeautifulSoup(page.content, 'html.parser')
    # # content_block = soup.find_all("div", class_="catalog-list js-catalog-list clearfix")
    # content_block = soup.find_all("h3", class_="title item-description-title")
    pagelinks = []

    driver.get('https://www.avito.ru/respublika_krym/zemelnye_uchastki/prodam/izhs?q=%D0%91%D0%B5%D1%80%D0%B5%D0%B3%D0%BE%D0%B2%D0%BE%D0%B5%20%D0%A4%D0%B5%D0%BE%D0%B4%D0%BE%D1%81%D0%B8%D1%8F')
    content_block = driver.find_elements_by_css_selector('.title.item-description-title')
    i = 0
    phones = driver.find_elements_by_css_selector(
        '.js-item-extended-contacts.item-extended-contacts.button.button-origin.button-origin_small')
    print(phones)
    for phone in phones:

        phone.click()
        wait = WebDriverWait(driver, 1)
        element = wait.until(ec.presence_of_element_located((By.CSS_SELECTOR, '.item-extended-phone')))
        # found only the first tag w/ attr so need to save all
        # need to fixate it on each elem thru parent or inherited struc
        img_src = element.get_attribute("src")
        print(img_src)
        fname = 'phone{0}.png'.format(i)
        urllib.request.urlretrieve(img_src, fname)
        i += 1
        # try:
        #     element = WebDriverWait(driver, 1).until(
        #         ec.presence_of_element_located((By.CSS_SELECTOR, '.item-extended-phone'))
        #     )
        # except StaleElementReferenceException:
        #     pass
        #     # driver.quit()



        # # img = driver.find_element_by_css_selector('.item-extended-phone')

    # print(pagelinks)

    # for ref in pagelinks:
    #     driver.get(ref)
    #     # soup = BeautifulSoup(page.text, 'html.parser')
    #     # phone = soup.find("div", class_="item-phone-number js-item-phone-number")
    #     phone = driver.find_element_by_css_selector('.button.item-phone-button.js-item-phone-button.button-origin.button-origin-blue.button-origin_full-width.button-origin_large-extra.item-phone-button_hide-phone.item-phone-button_card.js-item-phone-button_card')
    #     phone.click()
    #     # image = driver.find_elements_by_xpath("/html/body/div[3]/div[1]/div[3]/div[3]/div[2]/div[1]/div[1]/div/div[1]/div/div/a/img")
    #     image = driver.find_elements_by_tag_name("img")
    #     for img in image:
    #         img_src = img.get_attribute("src")
    #         print(img_src)


main()

from playwright.sync_api import sync_playwright

URL = "https://coffee-assist.streamlit.app/"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, timeout=60_000)
    page.wait_for_timeout(5_000)

    button = page.get_by_role("button", name="Yes, get this app back up!")
    if button.count() > 0:
        print("App was asleep, waking it up...")
        button.click()
        page.wait_for_timeout(30_000)
    else:
        print("App is awake.")

    browser.close()
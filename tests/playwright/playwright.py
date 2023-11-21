"""
Run pre-recorded smoke tests using playwright.

Notes:

- the db must be reset to the baseline with demo stories
- site must be running (currently hardcoded to port 5000)

More notes:

This is _just a smoke test_, it doesn't do any assertions.
The actions were _recorded_ using playwright's supertastic
code generation.  https://playwright.dev/docs/codegen

Then I added some tweaks:

Menu sub-items are only visible after hovering over the menu, e.g.:
  page.locator("#menu_books").hover()
  page.locator("#book_new").click()
"""

from playwright.sync_api import Playwright, sync_playwright  # , expect


def run(p: Playwright) -> None:  # pylint: disable=too-many-statements
    "Run the smoke test."

    # Run headless, can add an env var later for headless or not.
    headless = True

    def _print(s):
        if headless:
            print(s)

    _print("Opening browser.")
    browser = p.chromium.launch(headless=headless)
    context = browser.new_context()
    context.set_default_timeout(3000)
    page = context.new_page()

    _print("Reset db.")
    page.goto("http://localhost:5000/dev_api/load_demo")

    # Hardcoded port will cause problems ...
    page.goto("http://localhost:5000/")

    # Open Tutorial
    _print("Tutorial check.")
    page.get_by_role("link", name="Tutorial", exact=True).click()
    page.locator("#ID-178-1").click()
    page.frame_locator('iframe[name="wordframe"]').get_by_placeholder(
        "Translation"
    ).click()
    page.frame_locator('iframe[name="wordframe"]').get_by_placeholder(
        "Translation"
    ).fill("big grey thing")
    page.frame_locator('iframe[name="wordframe"]').get_by_role(
        "button", name="Save"
    ).click()
    page.get_by_title("Mark rest as known, mark page as read", exact=True).click()
    page.get_by_title(
        "Mark rest as known, mark page as read, then go to next page"
    ).click()
    page.get_by_title("Mark page as read, then go to next page", exact=True).click()
    page.get_by_role("link", name="Home").click()

    # Open and archive book.
    _print("Archive.")
    page.get_by_role("link", name="Büyük ağaç").click()
    page.get_by_role("link", name="Archive book").click()

    # Make a new book
    _print("New book.")
    page.locator("#menu_books").hover()
    page.locator("#book_new").click()
    page.locator("#language_id").select_option("4")
    page.get_by_label("Title").click()
    page.get_by_label("Title").fill("Hello")
    page.get_by_label("Text", exact=True).click()
    page.get_by_label("Text", exact=True).fill("Hello there.")
    page.get_by_role("button", name="Save").click()

    # Edit a term.
    _print("Edit term.")
    page.locator("#ID-1-1").click()
    page.frame_locator('iframe[name="wordframe"]').get_by_placeholder(
        "Translation"
    ).click()
    page.frame_locator('iframe[name="wordframe"]').get_by_placeholder(
        "Translation"
    ).fill("Hi.")
    page.frame_locator('iframe[name="wordframe"]').get_by_role(
        "button", name="Save"
    ).click()

    # Archive book, check archive.
    _print("Archive.")
    page.get_by_role("link", name="Archive book").click()
    page.locator("#menu_books").hover()
    page.get_by_role("link", name="Book archive").click()

    # Open term listing.
    _print("Term listing.")
    page.locator("#menu_terms").hover()
    page.get_by_role("link", name="Terms", exact=True).click()
    page.get_by_role("link", name="Hello").click()
    page.get_by_role("link", name="Sentences").click()
    page.get_by_role("link", name="back to list").click()

    # Edit language.
    _print("Edit language.")
    page.locator("#menu_settings").hover()
    page.get_by_role("link", name="Languages").click()
    page.get_by_role("link", name="English").click()
    page.get_by_role("button", name="Save").click()

    # Wipe the db.
    _print("Reset db.")
    page.get_by_role("link", name="click here").click()

    # Create a new language.
    _print("New language.")
    page.get_by_role("link", name="create your language.").click()
    page.locator("#predefined").select_option("Spanish")
    page.get_by_role("button", name="go").click()
    page.get_by_role("button", name="Save").click()

    # Create a new book for the new lang.
    _print("New book.")
    page.get_by_role("link", name="Create one?").click()
    page.get_by_label("Title").click()
    page.get_by_label("Title").fill("Hola.")
    page.get_by_label("Text", exact=True).fill("Tengo un perro.")
    page.get_by_role("button", name="Save").click()

    # Interact with text.
    page.get_by_text("perro").click()
    page.frame_locator('iframe[name="wordframe"]').get_by_placeholder(
        "Translation"
    ).click()
    page.frame_locator('iframe[name="wordframe"]').get_by_placeholder(
        "Translation"
    ).fill("dog.")
    page.frame_locator('iframe[name="wordframe"]').get_by_role(
        "button", name="Save"
    ).click()

    # Go home, backup is kicked off.
    _print("Verify backup started.")
    page.locator("#reading-footer").get_by_role("link", name="Home").click()
    page.get_by_role("link", name="Back to home.").click()

    # Check version.
    _print("Version.")
    page.locator("#menu_about").hover()
    page.get_by_role("link", name="Version and software info").click()

    # Custom style.
    _print("Custom style.")
    page.locator("#menu_settings").hover()
    page.get_by_role("link", name="Settings").click()
    page.get_by_label("Custom styles").click()
    page.get_by_label("Custom styles").fill(
        "span.status0 { background-color: yellow; }"
    )
    page.get_by_role("button", name="Save").click()

    # ---------------------
    context.close()
    browser.close()


with sync_playwright() as sp:
    run(sp)
print("ok.")

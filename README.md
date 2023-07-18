**Project 1: Simple Calculator**

This is a simple Python calculator that performs basic arithmetic operations: addition, subtraction, multiplication, and division. 
The user provides two numbers, and the calculator will perform the four operations on these numbers.
**Functions**
The calculator contains the following functions:

1. **add(a, b):** This function takes in two numbers and returns their sum.
2. **subtract(a, b):** This function takes in two numbers and returns the result of the first number subtracted by the second number.
3. **multiply(a, b):** This function takes in two numbers and returns their product.
4. **divide(a, b):** This function takes in two numbers and returns the result of the first number divided by the second number.

Please note that this script does not handle errors. If you attempt to divide by zero, for example, it will raise an error.

**Project 2: Head Scraper**

This Python script, called Head Scraper, allows you to scrape and analyze headings (h1, h2, h3, h4) from a list of websites. The program takes a CSV file containing website URLs as input. It checks the accessibility of each URL, determines if it can be scraped, and provides a status report.

The script utilizes the pandas library to handle CSV file operations, the requests library to make HTTP requests to the websites, and the BeautifulSoup library to parse and extract the desired headings from the HTML content.

Here are the **key functionalities** of the Head Scraper:

1. **Upload CSV:** The script prompts the user to provide the file path for the list of websites in CSV format.
2. **Preliminary Details:** It displays the number of websites in the sheet and the column names.
3. **URL Checking:** The script checks each URL's scrapability by making an HTTP request and examining the response status code. It records the scrapability status (Yes or No) for each URL.
4. **Scrape Headings:** The script demonstrates how to scrape headings from a single URL and displays the obtained h1, h2, h3, and h4 tags.
5. **Bulk Heading Scraping:** If the user chooses to proceed, the script scrapes headings from all the URLs in the list, storing the results in separate columns in the dataframe.
6. **Output:** The script displays the head of the modified dataframe, which includes the Scrape_Status and heading columns. It also exports the modified dataframe to a new CSV file.

Please note that this script provides a basic implementation and does not handle exceptions or errors that may occur during the scraping process.

#importing all necessary libaries and modules - mostly, pandas, requests and beautifulsoup
import pandas as pd
from sys import argv 
import requests
from bs4 import BeautifulSoup

#defining a function to turn the csv into a pandas dataframe
def upload_csv(file_path):
    data = pd.read_csv(file_path)
    return data

#defining a function to display preliminary details of the sheet
def display_deets(data):
    n_rows = len(data)
    print("The number of websites in this sheet are:", n_rows)
    print("The columns in this sheet are:")
    for col in data.columns: 
        print(col)
    col_list = list(data.columns)
    return n_rows, col_list
  
#defining a function to check if the url is scrapable or not
def check_possible(url_text):
    timeout = 15
    try:
        #adding a timeout feature to the requests.get function after 15 seconds
        response = requests.get(url_text, timeout = timeout)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException:
        return False
    except requests.exceptions.Timeout:
        return False
      
#defining a function to scrape relevant tags from each URL using BeautifulSoup
def scrape_mainheads(url_text): 
     response = requests.get(url_text)
     soup = BeautifulSoup(response.text, 'html.parser')
     h1_tags = soup.find_all('h1')
     h2_tags = soup.find_all('h2')
     h3_tags = soup.find_all('h3')
     h4_tags = soup.find_all('h4')
     h1_contents = [tag.text.strip() for tag in h1_tags]
     h2_contents = [tag.text.strip() for tag in h2_tags]
     h3_contents = [tag.text.strip() for tag in h3_tags]
     h4_contents = [tag.text.strip() for tag in h4_tags]
     return h1_contents, h2_contents, h3_contents, h4_contents

#collecting the filepath to access from the user as input
filepath = input("Enter the file path for list of websites:" + ' ')

#running the turning the csv into a df function
userfile = upload_csv(filepath)

#running the preliminary details of the df function
display_deets(userfile)

#checking to confirm the name of the URL column - similar to matching column on SaaS tools
url_col = input("Enter the name of the url columns:" + ' ')

#defining two counters for scrapable and non-scrapable links
not_scrapable = 0
scrapable = 0

#declaring a list to hold the scrape_status corresponding to each URL
scrape_status = []

#status update on URL checking started to keep user informed
print("Url Checking Started")

#setting up a for loop that applies the url_test function to each row in URL_col and returns either a Yes or No to the list of scrape status
for row in userfile[url_col]: 
    #printing Checking as a status update for the user once each URL is processed
    print("Checking...")
    url_test = check_possible(row)
    if url_test == True: 
        scrapable += 1
        scrape_status.append("Yes")
    else: 
        not_scrapable += 1
        scrape_status.append("No")
    
#print URL scraping status data
print("URL checking complete.")
print(f"Here is a status report on these urls:\n Scrapable = {scrapable} \n Not Scrapable = {not_scrapable}")

#moving the scrape_status list to a column in the main dataframe
userfile['Scrape_Status'] = scrape_status
print("Let's check scrape output!")

#running the scrape_mainheads function to one random entry
h1_list, h2_list, h3_list, h4_list = scrape_mainheads(userfile.loc[2, url_col])

#printing the output from the above for the user to review
print(h1_list, h2_list, h3_list, h4_list)

#asking the user if they want to do the same for all websites
check = input("Are you ready to scrape all Headings? Enter Y for Yes.")

#checking for user input and running the scrape_mainheads function if the response is Yes
if check == 'Y': 
    #declaring various lists to hold the scrape output
    h1_main = []
    h2_main = []
    h3_main = []
    h4_main = []
    for row in userfile[url_col]:
            print("Scraping")
            h1_list, h2_list, h3_list, h4_list = scrape_mainheads(row)
            h1_string = ', '.join(h1_list)
            h2_string = ', '.join(h2_list)
            h3_string = ', '.join(h3_list)
            h4_string = ', '.join(h4_list)
            h1_main.append(h1_string)
            h2_main.append(h2_string)
            h3_main.append(h3_string)
            h4_main.append(h4_string)
else: 
    print("Thank you!")

#moving various lists for the different headings as columns to the main file
userfile['H1_Text'] = h1_main
userfile['H2_Text'] = h2_main
userfile['H3_Text'] = h3_main
userfile['H4_Text'] = h4_main

#printing the entire file's head for the user to review
print(userfile.head)

#creating a new csv file with the new columns and not including the index column
userfile.to_csv('output11.csv', index=False)

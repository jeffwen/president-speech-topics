from bs4 import BeautifulSoup
from datetime import datetime, date, time
import requests
import pymongo
from progressbar import ProgressBar
import re
import pickle


link = 'http://www.presidency.ucsb.edu/sou.php'


# setting up the BeautifulSoup requester 
def requester(url):
    '''
    input: url that will be scraped
    output: a BeautifulSoup object that can be parsed
    '''
    response = requests.get(url)

    if response.ok:
        return BeautifulSoup(response.text)


##########################
### State of the Union ###    
##########################


# extract the state of the union address urls
def get_union_address_url(url):
    '''
    input: state of the union page (http://www.presidency.ucsb.edu/sou.php)
    output: list of urls for each of the state of the union addresses
    '''
    data = requester(url)

    # only extract particular links on the page that pertain to the state of the union addresses
    a = data.find('table').find_all('a', href = re.compile('.*/ws/index'))
    speech_urls = []
    for link in a:
        speech_urls.append(link['href'])
    return speech_urls


# extract the text from a single SOU address as a dictionary
def parse_union_address(url):
    '''
    input: single SOU url 
    output: dictionary with key details of the SOU
    '''
    data = requester(url)

    # the HTML of the page typically has formatted president name and speech title info
    try:
        president, title = data.find('title').text.split(':',1)
    except ValueError:
        president = ''
        title = ''

    # extract date and turn to date time object 
    date = datetime.strptime(data.find('span',{'class':'docdate'}).text, '%B %d, %Y')

    # extract each paragraph an put into a list 
    text = data.find('span',{'class':'displaytext'})
    text_list = text.find_all(text=True)
    text_list = [i.strip() for i in text_list]

    # create the dictionary
    aDict = {'president':president.strip(),'title':title.strip(),'date':date,'text':text_list,'url':url,'speech':'State of the Union'}
    return aDict


# create a list of dictionaries of each speech
def generate_union_list(urls):
    '''
    input: list of urls of all SOU addresses
    output: list of dictionaries (each representing a SOU)
    '''
    union_dict_list = []
    error_url = []
    pbar = ProgressBar() # keep track of progress

    # parse through each url and return a dictionary to create a list of dictionaries (one for each SOU)
    for url in pbar(urls):
        try:
            union_dict_list.append(parse_union_address(url))
        except:
            error_url.append(url)
    return union_dict_list


## processing the page and extracting the information
# all_links = get_union_address_url(link)
# speeches_list = generate_union_list(all_links)

# with open('union.pkl', 'w') as picklefile:
#     pickle.dump(speeches_list, picklefile)

# put state of the union in mongodb
# try:
#     conn=pymongo.MongoClient()
#     print "Connected successfully!"
# except pymongo.errors.ConnectionFailure, e:
#    print "Could not connect to MongoDB: %s" % e 

# db = conn['project_fletcher']
# collection = db.speeches
# collection.insert(speeches_list)


##########################
### Inaugural Speeches ###
##########################

link = 'http://www.presidency.ucsb.edu/inaugurals.php'


# get the url of each inaugural address
def get_inaug_urls(url):
    '''
    input: inaugural address page (http://www.presidency.ucsb.edu/inaugurals.php)
    output: list of urls for each of the inaugural addresses
    '''
    inaug_urls = []
    data = requester(url)
    a = data.find('table').find_all('a', href = re.compile('.*/ws/index'))
    for link in a:
        inaug_urls.append(link['href'])
    return inaug_urls


# parse the page to extract text and details
def parse_inaug_address(url):
    '''
    input: single url for the inaugural address
    output: dictionary with speech details
    '''
    data = requester(url)
    try:
        president, title = data.find('title').text.split(':',1)
    except ValueError:
        president = ''
        title = ''
    date = datetime.strptime(data.find('span',{'class':'docdate'}).text, '%B %d, %Y')
    text = data.find('span',{'class':'displaytext'})
    text_list = text.find_all(text=True)
    text_list = [i.strip() for i in text_list]
    aDict = {'president':president.strip(),'title':title.strip(),'date':date,'text':text_list,'url':url,'speech':'Inaugural Address'}
    return aDict


# create a list of dictionaries of each speech
def generate_inaug_list(urls):
    '''
    input: list of urls for the inaugural addresses
    output: list of dictionary with speech details
    '''
    inaug_dict_list = []
    error_url = []
    pbar = ProgressBar()
    for url in pbar(urls):
        inaug_dict_list.append(parse_inaug_address(url))
    return inaug_dict_list

## get the inagural address information
# all_links = get_inaug_urls(link)
# speeches_list = generate_inaug_list(all_links)

# with open('inaug.pkl', 'w') as picklefile:
#     pickle.dump(speeches_list, picklefile)

# put inaugural address in mongodb
# try:
#     conn = pymongo.MongoClient()
#     print "Connected successfully!"
# except pymongo.errors.ConnectionFailure, e:
#    print "Could not connect to MongoDB: %s" % e 

# db = conn['project_fletcher']
# collection = db.speeches
# collection.insert(speeches_list)


##########################
#### Press Conference ####    
##########################

link_base = 'http://www.presidency.ucsb.edu/news_conferences.php?year='


# get the urls of each press conference for a particular year
def get_press_urls(url):
    '''
    input: press release page url (http://www.presidency.ucsb.edu/news_conferences.php?year=)
    output: list of urls for each of the press release pages
    '''
    press_urls = []
    data = requester(url)
    a = data.find('table').find_all('a', href = re.compile('.*/ws/index'))
    for link in a:
        press_urls.append(link['href'][2:])
    return press_urls


# get all the urls from 1929-2016 (full range of the press releases)
def all_press_url(years=range(1929,2017),link_base = 'http://www.presidency.ucsb.edu/news_conferences.php?year='):
    '''
    input: range of years wanted, base link to be scraped
    output: list of urls for each of the press release pages
    '''
    all_urls = []
    pbar = ProgressBar()

    # appending the year to the base url to search through the different years
    for year in pbar(years):
        all_urls += get_press_urls(link_base + str(year))
    return all_urls

# extract the press release information and return a dictionary 
def parse_press_conf(url, base = 'http://www.presidency.ucsb.edu'):
    '''
    input: url of the press release page
    output: dictionary for each of the press releases
    '''
    data = requester(base+url)
    try:
        president, title = data.find('title').text.split(':',1)
    except ValueError:
        president = ''
        title = ''
    date = datetime.strptime(data.find('span',{'class':'docdate'}).text, '%B %d, %Y')
    text = data.find('span',{'class':'displaytext'})
    text_list = text.find_all(text=True)
    text_list = [i.strip() for i in text_list]
    aDict = {'president':president.strip(),'title':title.strip(),'date':date,'text':text_list,'url':base+url,'speech':'Press Conference'}
    return aDict

# create a list of dictionaries of each speech
def generate_press_list(urls):
    press_dict_list = []
    error_url = []
    pbar = ProgressBar()
    for url in pbar(urls):
        press_dict_list.append(parse_press_conf(url))
    return press_dict_list


# all_links = get_press_urls(link)
# speeches_list = generate_press_list(all_links)

# with open('press.pkl', 'w') as picklefile:
#     pickle.dump(speeches_list, picklefile)

# put press conferences in mongodb
# try:
#     conn = pymongo.MongoClient()
#     print "Connected successfully!"
# except pymongo.errors.ConnectionFailure, e:
#    print "Could not connect to MongoDB: %s" % e 

# db = conn['project_fletcher']
# collection = db.speeches
# collection.insert(speeches_list)

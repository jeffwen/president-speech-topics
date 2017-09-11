# Mr. President, what did you say?

Look at the final [result](http://jeffwen.com/html/president.html) of this project! If you want to read the blog post for this project take a look [here](http://jeffwen.com/2016/03/18/mr_president_what_did_you_say).

There is more information about the motivation for this project and the process on the blog so I'll try to keep this short.

## Scripts

`project_fletcher_cleaning.py` - this file contains the scripts that I used to scrape the information from [The American Presidency Project](http://www.presidency.ucsb.edu/sou.php).

* The process of scraping was more or less the same for each of the different types of speeches (state of the union, inaugural address, etc.)
	1. scrape the links for each of the speeches of a certain type (state of the union, inaugural address, etc.)
	2. use those links to scrape the individual speech page for the speech text, president name, date, and title
	3. create a dictionary for a particular speech
	4. loop through all the links to create a list of dictionaries (one for each speech)
	5. put everything into MongoDB

`project_fletcher_processing.py` - this file contains the scripts and functions used to generate topics via latent dirichlet allocation (LDA) and similarity scores using the document matrix after latent semantic indexing (LSI)

* I commented out the code that I wasn't using but I wanted to keep the code in the files because I wanted to have a record of what I did

## Final Product

`president.html` - The eventual product was a webpage with a couple visualizations created using D3.js.

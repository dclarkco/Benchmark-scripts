##################################
# Benchmark Mineral Intelligence #
# ---------DCLARK 2023---------- #
#         finnhub_tools.py       #
##################################

# Various tools and modules to take company name data, unclean, unorganized, and possibly redundant,
# and extract corporate meta-data from each identified company in an object-oriented and organized fashion.
# Returns a dataframe of cleaned company names, with meta-data identified from project stakeholders as being valuable fields.

# Cleaned of private keys and references, this code was useful for a single project, and is now redundant following the finalization of my contract's work.
# The fruits of this code are reflected in the Benchmark Pro company database, and it's utility is no longer required.
# It has been cleared as an example of the work I did during my time at BMI.

import regex as re, pandas as pd, isocodes as isostd, numpy as np
import dataframer, traceback, regex, finnhub, os, sys
from datetime import datetime, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from thefuzz import fuzz, process

exchanges = [
'AD', 'AQ', 'AS', 'AT', 'AX', 'BA', 'BC', 'BD', 'BE', 'BH', 'BK', 'BO', 'BR', 'CA', 'CN', 'CO',
'CR', 'CS', 'DB', 'DE', 'DS', 'DU', 'F', 'HE', 'HK', 'HM', 'IC', 'IR', 'IS', 'JK', 'JO', 'KL', 
'KQ', 'KS', 'KW', 'L', 'LN', 'LS', 'MC', 'ME', 'MI', 'MU', 'MX', 'NE', 'NL', 'NS', 'NZ', 'OL', 
'PA', 'PM', 'PR', 'QA', 'RG', 'SA', 'SG', 'SI', 'SN', 'SR', 'SS', 'ST', 'SW', 'SZ', 'T', 'TA', 
'TL', 'TO', 'TW', 'TWO', 'TU', 'US', 'V', 'VI', 'VN', 'VS', 'WA', 'XA', 'HA', 'SX', 'TG', 'SC'
]

companies = []

api_key = (os.environ.get('FINNHUB_KEY'))
finnhub_client = finnhub.Client(api_key = api_key)

class Company:
	def __init__(self, ticker = None, original_name = None, price = None, sheet = None, confidence = None):
		# can be passed in as parameters or ignored and set to None
		self.original_ticker = ticker
		self.original_name = original_name
		self.price = price
		self.sheet = sheet
		self.confidence = confidence
		self.finnhub_name = None
		self.link = None
		self.exchange = None
		self.country = None
		self.ownership = None
		self.ipo = None

		if ticker == np.nan or ticker == None:
			print("No Ticker passed to company object. Searching yahoo finance for tickers for company:", self.original_name)
			self.ticker = scrape_yahoo_finance(self.original_name)
		else:
			self.ticker = ticker

		if self.ticker:
			# Make an API call to retrieve company information
			print(f"Making API call to retrieve company: {self.original_name} information...")
			company_info = finnhub_client.company_profile(symbol=ticker)

			if company_info:
				self.build_profile(company_info)

			if not company_info:
				print(f"__ERROR: No company information found for {self.original_name} with ticker {self.ticker}.")
				self.ticker = scrape_yahoo_finance(self.original_name)
				print(self.original_ticker, self.ticker, self.original_name)

				self.build_profile(finnhub_client.company_profile(symbol=self.ticker))

	def build_profile(self, company_info):
		#self.ticker = company_info.get('ticker', ticker)
		self.link = company_info.get('weburl', None)
		self.exchange = company_info.get('exchange', None)
		self.country = dataframer.get_country_code(country_name = company_info.get('country', None), input_mode='alpha_2')
		self.finnhub_name = company_info.get('name', None)
		self.ipo = company_info.get('ipo', None)
		self.confidence = fuzz.ratio(self.original_name, self.finnhub_name)
		#self.ownership = finnhub_client.ownership(self.ticker, limit = 3).get('ownership', None)
		self.price = finnhub_client.quote(self.ticker).get('c', None)
	

	def __str__(self):
		return f"Original Name: {self.original_name} | Finnhub Name: {self.finnhub_name} | Ticker: {self.ticker} | Exchange: {self.exchange} | Country: {self.country} | Price: {self.price} | Link: {self.link} | IPO: {self.ipo} | Ownership: {self.ownership} | Confidence: {self.confidence} | Sheet: {self.sheet}"

def get_finnhub_company_profile(ticker):

	finnhub_client = finnhub.Client(api_key = api_key)
	return finnhub_client.company_profile(symbol = ticker)

def get__all_finnhub_tickers():

	finnhub_client = finnhub.Client(api_key = api_key)

	exchangeDF = pd.read_csv('reference/exchanges.csv')

	print(exchangeDF.columns, exchangeDF.head())

	limit = 100

	ticker_list = []
	
	for index, ex in exchangeDF.iterrows():

		try:
			print(ex['code'], ex['name'], ex['country'])
			tickers = finnhub_client.stock_symbols(ex['code'])
			print(len(tickers))

			code = ex['code']
			name = ex['name']
			alpha3 = isostd.get_country_code(ex['country'], 'alpha_2')
			country_name = ex['country_name']

			for t in tickers:

				t['exchange'] = code
				t['exchange_name'] = name
				t['ex_country_alpha3'] = alpha3
				t['ex_country_name'] = country_name

				#print(t)
				ticker_list.append(t)

			if index>limit:
				break

		except Exception as e:
			print(str(e))
			traceback.print_exc()


	df = pd.DataFrame(ticker_list)

	df = df.loc[:,['description','displaySymbol', 'symbol', 'exchange', 'exchange_name', 'ex_country_alpha3', 'ex_country_name', 'currency', 'type', 'isin', 'mic', 'shareClassFIGI', ]]
	df = df.set_index('symbol')

	print(df.columns, df.head())

	df.to_csv("reference/global_tickers.csv")

	return df

def get_tickers(path, sheet = None, limit = 20, search_mode = "TFIDF"):

	custom_stop_words = ['ltd', 'co', 'inc', 'incorporated', 'corp', 'plc', 'company', 'group', 'mining', 'machinery', 'energy', 'petrochemical']

	tfidf_params = {
		'max_df': 0.10,        # Example: Exclude terms in > 85% of documents
		'min_df': 1,           # Example: Include terms that appear in at least 1 documents
		'max_features': 5,      # Example: No limit on the number of features
		#'stop_words': custom_stop_words, 
		'ngram_range': (1, 3), # Example: Use unigrams (single words)
		'sublinear_tf': True,  # Example: Apply sublinear TF scaling
		'use_idf': True,       # Example: Apply IDF scaling
		'smooth_idf': True,    # Example: Smooth IDF weights
		#'idf_min': 1e-5,      # Example: Minimum IDF value
		'norm': 'l2'          # Example: Normalize vectors with 'l2' norm
	}

	# load and clean asset dataframe
	asset_df = dataframer.data_framer(path, sheetname = sheet, clean = False)
	#asset_df = asset_df[-10:]
	# add in new columns: Stock Ticker, Exchange, Country, Price, Link, Confidence
	asset_df.insert(len(asset_df.columns), "Confidence", None)
	asset_df.insert(len(asset_df.columns), "Exchange Country", None)
	asset_df.insert(len(asset_df.columns), "Stock Price", None)
	asset_df.insert(len(asset_df.columns), "Link", None)
	asset_df.insert(len(asset_df.columns), "Finnhub Name", None)

	asset_df['Company'] = asset_df['Company'].astype(str)
	asset_df['Company'] = asset_df['Company'].apply(dataframer.extract_first_substring, delimiters = r'[/\(\)]')

	print("ASSETDF INFO:", asset_df.info, asset_df.head())

	last_value = asset_df.iloc[-1,0]
	print("_____LAST VALUE:", type(last_value))

	if search_mode == "TFIDF":
		# load and clean ticker dataframe
		ticker_df = pd.read_csv('reference/global_tickers.csv')
		ticker_df.dropna(subset = ['description'], inplace = True)
		ticker_df['description'] = ticker_df['description'].astype(str)
		ticker_df['description'] = ticker_df['description'].str.lower()

		if len(custom_stop_words) > 0:
		# 	ticker_df['description'] = ticker_df['description'].apply(dataframer.clean_text, custom_stop_words)
			asset_df['Company'] = asset_df['Company'].apply(dataframer.clean_text, custom_stop_words)

		print("TICKER_DF INFO:", ticker_df.info, ticker_df.head())
		# Initialize the TF-IDF vectorizer
		tfidf_vectorizer = TfidfVectorizer(**tfidf_params)
		# Fit and transform the description column to TF-IDF vectors
		tfidf_matrix = tfidf_vectorizer.fit_transform(ticker_df['description'])		

	last_corp = ""
	last_company = None

	try:
		for index, row in asset_df.iterrows(): # for row in dataframe
			corp = row['Company']	# get company name
			if corp == None or corp == np.nan: # if company name is empty, skip
				break
			if corp == last_corp: # if company name is a the same as last (for consecutive company), skip and use last company object
				company = last_company
				print(f"DUPLICATE_MATCHED: {company.original_name} | WITH description: {company.finnhub_name} | TICKER: {company.ticker}")

			else: # if company name is not empty, search for ticker
				#NEEDS TO BE REFACTORED

				if search_mode == "TFIDF":
					# Convert the query to a TF-IDF vector
					query_vector = tfidf_vectorizer.transform([corp])
					
					cosine_similarities = (linear_kernel(query_vector, tfidf_matrix).flatten()) 				# Calculate cosine similarities between the query vector and all rows
					most_similar_index = cosine_similarities.argsort()[-1] 				# Find the index of the most similar row
					most_similar_row = ticker_df.iloc[most_similar_index]
					most_similar_score = round((cosine_similarities[most_similar_index]), 2) # cosine similarity
					
					if most_similar_score >= 0.67: # if company is public, high confidence match and/or ticker
						ticker = most_similar_row['symbol']
						print(f"MATCHED: {corp} | WITH: {most_similar_row['description']} | TICKER: {ticker} | COSINE_SIM: {most_similar_score}")

					else: # if company is private, low confidence match and/or no ticker
						ticker = None
						print(f"NOT MATCHED:{corp} | WITH NEAREST: {most_similar_row['description']} | COSINE_SIM: {most_similar_score}")

				elif search_mode == "FUZZY":

					filtered_df = ticker_df[ticker_df['description'].str.contains(corp)]

					if len(filtered_df) > 0:
						# Get a list of strings that contain the search term
						strings_to_search = filtered_df['description'].tolist()
						print(strings_to_search)
						# Find the closest match using FuzzyWuzzy
						closest_match, most_similar_score = process.extractOne(corp, strings_to_search)
						print(f"Closest match: {closest_match} with similarity score: {most_similar_score}")

						# Find the index of the closest match:
						closest_match_index = ticker_df[ticker_df['description'] == closest_match].index[0]

						# Retrieve the most similar row from the DataFrame
						most_similar_row = ticker_df.iloc[closest_match_index]

						# get ticker from most similar row
						ticker = most_similar_row['symbol']
					else:
						print("No matches found for the search term within the DataFrame.")
						ticker = None
						most_similar_score = 0
					
				#fuzz_similarity_score = round(fuzz.partial_ratio(corp.lower(), most_similar_row['description'].lower()), 2) # levenshtein distance
				#ensemble_similarity_score = round(((most_similar_score + fuzz_similarity_score) / 2), 2) # average of levenshtein and cosine similarity
				
				elif search_mode == "YFINANCE":
					ticker = yfinance_ticker_lookup(corp)
					most_similar_score = 0

				# if ticker is not found, search finnhub data
				company = Company(ticker = ticker, original_name = corp, sheet = sheet) # create company object, automatically get finnhub data from ticker
				companies.append(company)

				print(f"COMPANY OBJECT CREATED #{len(companies)}: {corp} | FINNHUB: {company.finnhub_name} | {company.country} | {company.ticker} | CONFIDENCE = {company.confidence}")

			# add company data to asset dataframe
			asset_df.at[index, "Stock Ticker"] =   company.ticker 
			asset_df.at[index, "Exchange Country"] = company.country
			asset_df.at[index, "Exchange"] = company.exchange
			asset_df.at[index, "Stock Price"] = company.price
			asset_df.at[index, "Link"] = company.link
			asset_df.at[index, "Confidence"] = company.confidence
			asset_df.at[index, "Finnhub Name"] = company.finnhub_name

			last_corp = corp
			last_company = company

			if index > limit:
				break

		for company in companies:
			print(company.__str__())

		companies_data = {
		"Original Name": [company.original_name for company in companies],
		"Finnhub Name": [company.finnhub_name for company in companies],
		"Ticker": [company.ticker for company in companies],
		"Exchange": [company.exchange for company in companies],
		"Country": [company.country for company in companies],
		"Price": [company.price for company in companies],
		"Link": [company.link for company in companies],
		"Confidence": [company.confidence for company in companies]
		}

		companyDF = pd.DataFrame(companies_data)
		
		companyDF.to_csv(f"reference/{sheet}_finnhub_companies.csv")
		asset_df.to_csv(f"reference/{sheet}_tickers.csv")

		return asset_df

	except Exception as e:
		print(f"get_tickers Error: {str(e)}")
		traceback.print_exc()
		return -1

def yfinance_ticker_lookup_DEPRECIATED(query):
	import yfinance as yf
	import yahooquery as yq

	try:
		result = yq.search(query)
		if 'shortname' in result:
			print("Single quote found for:", query," -- ",result['shortname'])

		if 'symbol' in result:
			return result['symbol']

		elif 'quotes' in result and len(result['quotes']) > 0:
			print("Multiple quotes found: ", query, "\n")
			# Filter quotes based on quoteType
			filtered_quotes = [quote for quote in result['quotes'] if quote.get('quoteType') in ['EQUITY', 'STOCK']]

			if filtered_quotes:
				# Perform fuzzy matching on shortnames
				#print(filtered_quotes)
				tickers = {}

				for quote in filtered_quotes:
					print(quote['shortname'])
					tickers[quote['shortname']] = quote['symbol']

				match, score = process.extractOne(query, tickers.keys())

				if score > 60:  # Adjust the threshold based on your needs
					matching_symbol = tickers[match]
					print(f"Fuzzy match found: {match} (Score: {score}) -- {matching_symbol}")
					return matching_symbol
			else:
				print("No matches found for the search term")
				return None
		else:
			return None
	except Exception as e:
		print(f"yfinance_ticker_lookup Error: {str(e)}")
		

	# if ticker:
	# 	stock_data = yf.download(ticker, start="2023-01-01", end="2023-12-31")
	# 	print(f"Stock data for {company_name} ({ticker}):\n", stock_data.head())
	# else:
	# 	print(f"No ticker found for {company_name}")

def remove_stopwords(text, stop_words):
	words = text.split()
	words = [word for word in words if word not in stop_words]
	return ' '.join(words)

def search_finnhub_data(query):

	# Load your DataFrame from "global_tickers.csv"
	df = pd.read_csv('reference/global_tickers.csv')
	df.dropna(subset = ['description'], inplace = True)
	df['description'] = df['description'].astype(str)
	df['description'] = df['description'].str.lower()
	stop_words = []

	if len(stop_words) > 0:
		df['description'] = df['description'].apply(remove_stopwords, stop_words)

	print(df.head(), df.info)
	# Initialize the TF-IDF vectorizer
	tfidf_vectorizer = TfidfVectorizer()
	# Fit and transform the description column to TF-IDF vectors
	tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])
	# Convert the query to a TF-IDF vector
	query_vector = tfidf_vectorizer.transform([query])
	# Calculate cosine similarities between the query vector and all rows
	cosine_similarities = linear_kernel(query_vector, tfidf_matrix).flatten()
	# Find the index of the most similar row
	most_similar_index = cosine_similarities.argsort()[-1]
	# Retrieve the most similar row from the DataFrame
	most_similar_row = df.iloc[most_similar_index]
	# Print or return the most similar row
	print(f"MATCHED: {query} | WITH description: {most_similar_row['description']} | TICKER: {most_similar_row['symbol']}")

	return most_similar_row['symbol']

def scrape_yahoo_finance(query):
	# URL for Yahoo Finance lookup
	from thefuzz import fuzz, process
	import requests
	from bs4 import BeautifulSoup

	url = f"https://finance.yahoo.com/lookup?s={query}"
	custom_headers = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36", "Accept":"text/html,application/xhtml+xml,application/xml; q=0.9,image/webp,image/apng,*/*;q=0.8"}

	try:
		# Send an HTTP GET request to the URL
		response = requests.get(url, headers = custom_headers)

		# Check if the request was successful
		if response.status_code == 200:
			# Parse the HTML content using Beautiful Soup
			soup = BeautifulSoup(response.text, "html.parser")
			# Find the elements containing stock tickers
			table_elements = soup.find("tbody")

			#ticker_elements = soup.find_all("a", class_="Fw(b)")
			#name_elements = soup.find_all("td", class_ = "data-col1 Ta(start) Pstart(10px) Miw(80px)")

			if table_elements == None:
				return None

			# Extract the tickers from the elements

			best_match = {}
			best_score = 0

			for element in table_elements:
				#tickers.append(element.find("a").get("data-symbol"))
				ticker = element.find("a").get("data-symbol")
				name = element.find("a", class_ = "Fw(b)").get("title")
				suffix = element.find("a", class_ = "Fw(b)").get("href")
				link = f"finance.yahoo.com{suffix}"
				price = element.find("td", class_ ="data-col2 Ta(end) Pstart(20px) Pend(15px)").string
				exchange = element.find("td", class_ ="data-col5 Ta(start) Pstart(20px) Pend(6px) W(30px)").string
				similarity = fuzz.ratio(query.lower(), name.lower())
				
				if similarity > best_score:
					best_match = ticker
					print(ticker, name, price, link, similarity, exchange)

			print(best_match)
			return best_match
		else:
			print(f"Failed to retrieve data. Status code: {response.status_code}")
			return -1

	except Exception as e:
		print(f"scrape_yahoo_finance Error: {str(e)}")
		traceback.print_exc()
		return -1

def get_forex(currency = 'USD', start_date = '2015-01-01', date_list = []):

	import finnhub
	from datetime import datetime, timedelta

	api_key = (os.environ.get('FINNHUB_KEY')) # get the API key from the environment (set in .bash_profile, or hard code it here at your own risk)
	finnhub_client = finnhub.Client(api_key=api_key)
	forex_list = []

	if len(date_list) == 0:
		start_date = datetime.strptime(start_date, '%Y-%m-%d')
		end_date = datetime.now()
		# Generate list of dates
		date_list = [(start_date + timedelta(days=x)).strftime('%Y-%m-%d') for x in range((end_date - start_date).days + 1)]
		
	i = 0
	for date in date_list: # iterate through each day, and query finnhub for the currency exchange rates
		print(f"Retrieving {currency} exchange rates for date: {date}")
		ex = finnhub_client.forex_rates(base = currency, date = date)
		# modify the dictionary to include the date
		ex_on_date = {
			'date' : date,
			'base' : ex['base'],
			**ex['quote']
		}
		i += 1

		forex_list.append(ex_on_date)

	print(f"{len(forex_list)} days of data retrieved.")
	df = pd.DataFrame(forex_list) 	# create dataframe from the list of dictionaries
	df.set_index('date', inplace=True) 	# set the 'date' column as index and remove it from columns

	print(df)
	df.to_csv(f'reference/{currency}_forex.csv')
	return df

def get_forex_candles(currency = 'USD', exchange = "oanda", start_date = '2015-01-01'):

	import finnhub
	from datetime import datetime, timedelta

	api_key = (os.environ.get('FINNHUB_KEY')) # get the API key from the environment (set in .bash_profile, or hard code it here at your own risk)
	finnhub_client = finnhub.Client(api_key=api_key)
	forex_list = []

	start_epoch = int(datetime.datetime.strptime(start_date, '%Y-%m-%d').timestamp())
	end_epoch = int(time.time())

	print(start_epoch, end_epoch)

	# Get the forex candles
	res = finnhub_client.forex_candles('OANDA:EUR_USD', 'D', start_epoch, end_epoch)
	i = 0
	for date in date_list: # iterate through each day, and query finnhub for the currency exchange rates
		print(f"Retrieving {currency} exchange rates for date: {date}")
		ex = finnhub_client.forex_rates(base = currency, date = date)
		# modify the dictionary to include the date
		ex_on_date = {
			'date' : date,
			'base' : ex['base'],
			**ex['quote']
		}
		i += 1

		forex_list.append(ex_on_date)

	print(f"{len(forex_list)} days of data retrieved.")
	df = pd.DataFrame(forex_list) 	# create dataframe from the list of dictionaries
	df.set_index('date', inplace=True) 	# set the 'date' column as index and remove it from columns

	print(df)
	df.to_csv(f'reference/{currency}_forex.csv')
	return df


def collate_company_corpus(path): 
	import openpyxl

	# Load the Excel file
	file_path = path
	workbook = openpyxl.load_workbook(file_path)

	# Create an empty list to store DataFrames
	dfs = []	

	# Iterate through all sheets in the workbook
	for sheet_name in workbook.sheetnames:
		sheet = workbook[sheet_name]

		# Extract data from the sheet and create a DataFrame
		data = sheet.values
		columns = next(data)  # Assumes the first row contains column headers
		df = pd.DataFrame(data, columns=columns)

		# Add the "Sheet" column to the DataFrame
		df['Sheet'] = sheet_name
		df['Notes'] = None

		# Select only the "Company" column and the new "Sheet" & "notes" column
		df = df[['Sheet', 'Company']]

		# Append the DataFrame to the list
		dfs.append(df)

	# Concatenate all DataFrames into a primary DataFrame
	primary_df = pd.concat(dfs, ignore_index=True)

	# Close the workbook
	workbook.close()

	# Clean data, remove duplicates, split combined entries, generate slugs, and sort
	primary_df = primary_df.dropna(how="any", axis=0)
	primary_df['Company'] = primary_df['Company'].fillna('')
	primary_df['Company'] = primary_df['Company'].str.strip()
	primary_df['Company'] = primary_df['Company'].replace(r'\bNone\b', '', regex=True)
	primary_df['Company'] = primary_df['Company'].str.replace('&', 'and')
	primary_df['Company'] = primary_df['Company'].str.replace('[%,,.\-¬†’\']', '', regex=True)	# primary_df['Company'] = primary_df['Company'].str.replace('.', '')
	# primary_df['Company'] = primary_df['Company'].str.replace('-', '')
	# primary_df['Company'] = primary_df['Company'].str.replace('¬†', '')
	primary_df['Company'] = primary_df['Company'].replace('\d+', '', regex=True)
	primary_df['Company'] = primary_df['Company'].astype(str)

	# primary_df['Company'] = primary_df['Company'].str.replace('from acquisition of', '')
	# primary_df['Company'] = primary_df['Company'].str.replace('partnered with', '')
	# primary_df['Company'] = primary_df['Company'].str.replace('formerly', '')
	# primary_df['Company'] = primary_df['Company'].str.replace('others', '')
	# primary_df['Company'] = primary_df['Company'].str.replace('incl', '')
	# primary_df['Company'] = primary_df['Company'].str.replace('including', '')
	# primary_df['Company'] = primary_df['Company'].str.replace('in dispute', '')
	# primary_df['Company'] = primary_df['Company'].str.replace('incl', '')
	# primary_df['Company'] = primary_df['Company'].str.replace('acquired', '')
	# primary_df['Company'] = primary_df['Company'].str.replace('acquisition', '')
	# primary_df['Company'] = primary_df['Company'].str.replace('previously', '')
	# primary_df['Company'] = primary_df['Company'].str.replace('Other', '')
	# primary_df['Company'] = primary_df['Company'].str.replace('Global', '')
	# primary_df['Company'] = primary_df['Company'].str.replace('pre', '')
	# primary_df['Company'] = primary_df['Company'].str.replace('owned by', '')
	# primary_df['Company'] = primary_df['Company'].str.replace('Various', '')
	# primary_df['Company'] = primary_df['Company'].str.replace('proposed', '')
	# primary_df['Company'] = primary_df['Company'].str.replace('uding earnin right', '')
	# primary_df['Company'] = primary_df['Company'].str.replace('Tianqi proposed  in Q', '')
	# primary_df['Company'] = primary_df['Company'].str.replace('Co Ltd', '')
	# primary_df['Company'] = primary_df['Company'].str.replace('JV', '')
	print(primary_df.columns)
	primary_df[['Company', 'Notes']] = primary_df['Company'].apply(dataframer.split_and_expand).apply(pd.Series)
	#primary_df = primary_df.explode('Company', ignore_index=True) <- if we want to keep the notes column as separate entries.
	primary_df['Company'] = primary_df['Company'].str.strip()

	primary_df = primary_df.drop_duplicates(subset=['Company'], keep='first')
	primary_df = primary_df.dropna(how="any", axis=0)

	primary_df['Slug'] = primary_df['Company'].apply(dataframer.generate_slug)
	primary_df = primary_df.sort_values(by=['Sheet', 'Company'])

	print(primary_df.head())
	primary_df.to_csv('reference/company_corpus.csv', index=False)
	return primary_df

def get_finnhub_data(file, limit = 1000):

	# Load and clean asset dataframe

	asset_df = dataframer.data_framer(file, clean = False)
	asset_df = asset_df.dropna(subset=['New Company Name'])

	# Add in new columns: Confidence, Finnhub Name, Link, 
	asset_df.insert(4, "FH_Ticker", None)
	asset_df.insert(5, "Confidence", None)
	asset_df.insert(6, "Finnhub Name", None)
	asset_df.insert(7, "Link", None)
	asset_df.insert(8, "FH HQ Country", None)

	for index, row in asset_df.iterrows():
		if index > limit:
			break
		corp_name = str(row['New Company Name'])
		ticker = str(row['Stock Ticker']) # I shouldn't have to fucking convert to str, @pandas...

		# print( f"Retrieving data for company: \'{corp_name}\' with ticker: \'{ticker}\' | dtype of name: {type(corp_name)}, dtype of ticker: {type(ticker)}")
		if corp_name == None or corp_name == np.nan or corp_name == "" or corp_name == " " or corp_name == "nan":
			break
		company = Company(ticker = ticker, original_name = corp_name)
		print(company.original_name, company.finnhub_name, company.ticker, company.exchange, company.country, company.price, company.link, company.ipo, company.confidence)

		asset_df.loc[index,'FH_Ticker'] = company.ticker
		asset_df.loc[index,'Confidence'] = company.confidence
		asset_df.loc[index,'Finnhub Name'] = company.finnhub_name
		asset_df.loc[index,'Link'] = company.link
		asset_df.loc[index,'Share Price'] = company.price
		asset_df.loc[index,'FH HQ Country'] = company.country

	asset_df.to_csv(f"reference/Nickel_company_database_FH.csv")
	return asset_df

get_finnhub_data('/Users/dclark/Downloads/Benchmark Company Database_MASTER(2).xlsx')

#get_tickers(path = '/Users/dclark/Documents/benchmark/code/misc/Lithium-ion-Battery-Database-Q3-2023.xlsx', sheet = 'Cell Supply', limit = 5000, search_mode="YFINANCE")
#get_forex(currency = "USD", start_date="2023-01-01")
#collate_company_corpus(path = 'reference/company_corpus_cleaned.xlsx')
#get_tickers("reference/company_corpus.csv", limit = 5000, search_mode="YFINANCE")


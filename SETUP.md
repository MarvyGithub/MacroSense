\# MacroSense Setup Instructions



\## Getting Your FRED API Key



1\. Go to https://fred.stlouisfed.org/docs/api/api\_key.html

2\. Click Request API Key

3\. Create a free account

4\. Copy your API key



\## Setting Up Your Environment



1\. Clone this repository

2\. Create a file called .env in the root folder

3\. Add this line to your .env file:



FRED\_API\_KEY=your\_api\_key\_here



4\. Install required libraries:



pip install -r requirements.txt



5\. Run notebooks in order:

&#x20;  - 01\_data\_collection.ipynb

&#x20;  - 02\_EDA.ipynb

&#x20;  - 03\_Feature\_Engineering.ipynb

&#x20;  - 04\_modelling.ipynb

&#x20;  - 05\_evaluation.ipynb

&#x20;  - 06\_walk\_forward.ipynb



6\. Launch dashboard:



streamlit run app.py


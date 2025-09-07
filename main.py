import streamlit as st
#from st_copy_to_clipboard import st_copy_to_clipboard # pip install st-copy-to-clipboard
from email_gen_llama import email_generator

st.title("📧 Cold Email :blue[Generator]",help="ℹ️ Currently using llama 3.1")
st.info("""Currently using llama 3.1""", icon="ℹ️")

url_input = st.text_input("Enter a URL...", value="https://internshala.com/job/detail/search-engine-optimization-seo-executive-job-in-pune-at-up-market-research1755528054", help="job listing listing page url")

col1, col2 = st.columns(2) # to get two option in same line

with col1:
    # template choice
    template_choice = st.selectbox(label="Choose a Template..." , options=('template-1','template-2','template-3'),width=350)
with col2:
    # max token to use choice
    tokens = st.selectbox("Token Size", options=(250,564,1048,1248,1520,2048),width=340, index=1, help="max llm token usage limit")

# temperature parameter adjustment 
temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, width=200,value=1.0, help="control creativity in the email writing")

# final submit button
submit_button = st.button("Generate", type='primary')


if submit_button:
    try:
        mail_gen = email_generator(url_input)
        
        scrapping = mail_gen.data_scrapper()
        making_db = mail_gen.db(path='my_portfolio.csv')
        
        first_llm = mail_gen.side_llm(scrapping)
        parsed_info = mail_gen.json_parser(first_llm)
        
        meta_links = mail_gen.links_from_db(making_db,parsed_info)
        mail_writer = mail_gen.cold_email_llm(parsed_info,meta_links, template_no=str(template_choice), temperature=temperature, token_size=tokens)
        
        st.code(mail_writer, language='markdown') # the output of the EMAIL in markdown down format. Avails a copy option on the top right corner
        
        #st_copy_to_clipboard(mail_writer) # practical need for now
        
    except Exception as e: # error handling or exception handling
        print("Error :- ", e)


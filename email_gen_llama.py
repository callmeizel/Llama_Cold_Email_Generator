import pandas as pd
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

import chromadb
import uuid


_=load_dotenv()

class email_generator:
    def __init__(self,url):
        self.url = url
    
    def data_scrapper(self):
        
        self.inp_url = self.url
        
        self.loader = WebBaseLoader(str(self.inp_url))
        self.scrapped = self.loader.load().pop().page_content
        
        return self.scrapped
        
    
    def db(self,path='my_portfolio.csv'):
        
        self.path = path # explicit for now the data file
        
        # BUILDING THE DATABSE 
        self.client = chromadb.PersistentClient(path='vectorstore')
        
        self.collection = self.client.get_or_create_collection(name='portfolio')
        
        # PUSHING THE DATA INTO THE DATABASE
        self.df = pd.read_csv(self.path)
        
        if not self.collection.count():
            for _,rows in self.df.iterrows():
                self.collection.add(documents=rows['Techstack'],
                                    metadatas={'Links':rows['Links']}, # metadata are in form of a dictonary
                                    ids=[str(uuid.uuid4())])
        
        return self.collection
    
    
    def side_llm(self,scrapped_data):
        
        self.scrapped_data=scrapped_data
        self.data_from_site_temp = PromptTemplate.from_template("""
                                                     ### SCCRAPE TEXT FROM WEBSITE:
                                                     {page_data}
                                                     ### INSTRUCTIONS:
                                                     you scraped text from the career's page of a website.
                                                     your job is to extract the job postings and return them in JSON format cotaining
                                                     following keys: 'role','experience;','skills','description'.
                                                     Olny return the valid JSON
                                                     ### VALID JSON (NO PREAMBLE)""")
        
        self.llm  = ChatGroq(model = 'llama-3.3-70b-versatile',
               temperature=0.3,
               api_key=f'{os.getenv('GROQ_LLAMA_APIKEY')}')
        
        self.chain = self.data_from_site_temp | self.llm
        
        self.response = self.chain.invoke(input={'page_data':self.scrapped_data})
        
        return self.response.content
        
    def json_parser(self,data):
        
        self.json_parser = JsonOutputParser()
        self.data = data
        
        self.json_formatted = self.json_parser.parse(self.data)
        
        return self.json_formatted
    
    def links_from_db(self,db,url_info):
        
        self.links = db.query(query_texts=url_info['skills'],
                                           n_results=2).get('metadatas',[])
        
        links_no = self.links
        
        return links_no
    
    def cold_email_llm(self, description, metadata,temperature=1,token_size=564,template_no="template-1"):
        
        self.description = description
        self.metadata = metadata
        
        self.temperature = temperature
        self.token_size = token_size
        self.template_no = template_no
    
        self.prompt_tempt_1 = PromptTemplate.from_template(template="""
                                                 ### JOB DESCRIPTION:
                                                 {job_description}
                                                 
                                                 ### INSTRUCTION:
                                                 You are Mohan, a business development executive at a Big Tech company.
                                                 Your job is to write a cold email to the client regarding the job mentioned above
                                                 describing the capability in fulfilling their needs.
                                                 Also add the most relevant ones from the following links to showcase your Companies PortFolio Links: {link_list}
                                                 Remember you are Mohan, BDE at a Big Tech Company.
                                                 DO NOT PROVIDE A PREAMBLE.
                                                 ### EMAIL (NO PREAMBLE):""")
        
        self.prompt_tempt_2 = PromptTemplate.from_template(template="""
                                                          ### JOB DESCRIPTION (TECHNICAL SPECIFICATIONS):
                                                        {job_description}

                                                        ### RELEVANT PROJECTS & DOCUMENTATION:
                                                        {link_list}

                                                        ### INSTRUCTION:
                                                        You are Arjun Singh, a Technical Solutions Consultant at a major tech corporation. Your goal is to initiate a conversation with a technical lead or engineering manager.

                                                        1.  Thoroughly analyze the technical specifications in the job description to identify the core technical challenges (e.g., scalability, integration, specific tech stack).
                                                        2.  Write a cold email that highlights 2-3 specific technical capabilities of your company that directly solve these challenges.
                                                        3.  Avoid generic sales language. Focus on value, outcomes, and technical expertise.
                                                        4.  From the list of projects and documentation, Showcarse the links below that best demonstrates the most critical technical capability needed by the client.
                                                        5.  Remember, your persona is Arjun Singh, a technical expert.
                                                        6.  DO NOT write a preamble. The output must begin directly with the email subject line.
                                                        
                                                        ### TECHNICAL OUTREACH EMAIL (NO PREAMBLE):""")

        self.prompt_tempt_3 = PromptTemplate.from_template(template="""
                                                                    ### CLIENT REQUIREMENTS:
                                                                    {job_description}

                                                                    ### COMPANY PORTFOLIO:
                                                                    {link_list}

                                                                    ### INSTRUCTION:
                                                                    You are Priya Sharma, a Senior Account Executive at a leading technology firm. Your task is to write a concise and compelling cold email to a potential client based on their requirements.

                                                                    1.  Analyze the client's key needs from the requirements provided.
                                                                    2.  Draft an email that directly addresses their primary challenge.
                                                                    3.  Position our company as the expert solution.
                                                                    4.  From the Company Portfolio, select and include the TWO most relevant links that serve as powerful case studies for this client.
                                                                    5.  Maintain a highly professional and confident tone.
                                                                    6.  You are Priya Sharma. Do not add any preamble or introduction before the email.

                                                                    ### EMAIL DRAFT (NO PREAMBLE):""")

        self.llm2 = ChatGroq(model = 'llama-3.3-70b-versatile',
                    temperature=self.temperature,
                    max_tokens=float(self.token_size),
                    api_key=f'{os.getenv('GROQ_LLAMA_APIKEY')}')
        
        if self.template_no == "template-1":
            self.mail_chain = self.prompt_tempt_1 | self.llm2
            
        elif self.template_no == "template-2":
            self.mail_chain = self.prompt_tempt_2 | self.llm2
            
        elif self.template_no == "template-3":
            self.mail_chain = self.prompt_tempt_3 | self.llm2
        
        self.mail_output = self.mail_chain.invoke(input={'job_description':str(self.description['description']),
                                                         'link_list':self.metadata})
        
        self.final_email = self.mail_output.content
        
        return self.final_email
from dotenv import dotenv_values
from sqlalchemy import create_engine, Column, String, Boolean, DateTime
from sqlalchemy.orm import sessionmaker

def connect_to_db():
	config = dotenv_values(".env")
	engine = create_engine(config["LIVING_ROOM_MSSQL"])
	Session = sessionmaker(bind=engine)
	return Session

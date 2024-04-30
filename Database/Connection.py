from mongoengine import *
from pymongo import monitoring
from dotenv import dotenv_values
import logging

log = logging.getLogger(__name__)


class CommandLogger(monitoring.CommandListener):

	def started(self, event):
		log.debug("Command {0.command_name} with request id {0.request_id} started on server {0.connection_id}".format(event))
		pass

	def succeeded(self, event):
		log.debug("Command {0.command_name} with request id {0.request_id} on server {0.connection_id} succeeded in {0.duration_micros} microseconds".format(event))
		pass

	def failed(self, event):
		log.critical("Command {0.command_name} with request id {0.request_id} on server {0.connection_id} failed in {0.duration_micros} microseconds".format(event))
		pass


def connect_to_db():
	config = dotenv_values(".env")
	monitoring.register(CommandLogger())
	connect(host=config["MONGO_CONN_LIVING_ROOM"], uuidRepresentation='standard')

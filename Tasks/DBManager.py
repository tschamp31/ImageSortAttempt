import shutil
import time

from Database import *
import ThirdParty.deepface as deepface
from PIL import Image
import logging.config
import numpy as np
import os
import io

from Utils import FacialLikeness

log = logging.getLogger(__name__)


class DBManager:
	def __init__(self, work_queue, test):
		log.info("DBManager: {} - {}".format(os.getpid(),test))
		self.compare_faces_follow_up(work_queue)

	def compare_faces_follow_up(self, face_relation_queue):
		log.info("DBManager: {} - Starting Compare Face Part 2 - Waiting {}s".format(os.getpid(), os.getpid()/300))
		#time.sleep(os.getpid() / 300)
		while True:
			try:
				follow_job = face_relation_queue.get(block=True)
				if follow_job is None:
					log.info("DBManager:{} - Completed Their Queue.".format(os.getpid()))
					return
				like_face_list = follow_job[2]
				unlike_face_list = follow_job[1]
				base_face_job = follow_job[0]
				if base_face_job is None:
					face_relation_queue.task_done()
					continue
				log.info("DBManager: {} - Starting Face {}".format(os.getpid(), base_face_job.FaceID))
				start = time.time()
				no_face_count = 0
			except Exception as e:
				log.error("DBManager:{} - Queue Error - {}".format(os.getpid(), e))
				break
			try:
				if like_face_list is not None and len(like_face_list) > 0 and base_face_job is not None:
					try:
						like_update_count = FaceToFaceRelation.objects(ReferenceFace__in=like_face_list).update(upsert=True, multi=True, add_to_set__LikeFaces=base_face_job, full_result=True)
						log.info("DBManager: {} - Face {} - LikeFace Lists: {} ".format(os.getpid(), base_face_job.FaceID, like_update_count))
					except Exception as e:
						log.error("DBManager: Like Exception - {}".format(e))
				if unlike_face_list is not None and len(unlike_face_list) > 0 and base_face_job is not None:
					try:
						unlike_update_count = FaceToFaceRelation.objects(ReferenceFace__in=unlike_face_list).update(upsert=True, multi=True, add_to_set__UnLikeFaces=base_face_job, full_result=True)
						log.info("DBManager: {} - Face {} - UnLikeFace Lists: {} ".format(os.getpid(), base_face_job.FaceID, unlike_update_count))
					except Exception as e:
						log.error("DBManager: Unlike Exception - {}".format(e))
				else:
					face_relation_queue.task_done()
			except Exception as e:
				log.error("DBManager: Face Comparison Part 2 Exception - {}".format(e))
			log.info("DBManager: {} - Completed Face - {} - Took {}s".format(os.getpid(), base_face_job.FaceID, time.time()-start))
			face_relation_queue.task_done()

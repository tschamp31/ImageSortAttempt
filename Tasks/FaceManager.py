import shutil
from Database import *
from deepface import DeepFace
from PIL import Image
import logging.config
import numpy as np
import os
import io

from Utils import FacialLikeness

log = logging.getLogger(__name__)


class FaceManager:
	def __init__(self, work_queue, instruction_queue, input_path, face_path, no_face_path):
		log.info("PID: {}".format(os.getpid()))
		self.input_path = input_path
		self.face_path = face_path
		self.no_face_path = no_face_path
		self.instruction_queue = instruction_queue
		job_to_do = instruction_queue.get(block=True)
		log.info("PID: {} - Job Set To - {}".format(os.getpid(),job_to_do))
		match job_to_do:
			case "ProcessFiles":
				self.process_file(work_queue)
			case "CompareFaces":
				self.compare_faces(work_queue)
			case "ExtractEmbeddings":
				self.extract_embeddings(work_queue)
			case "DimensionReduction":
				self.dimension_reduction_attempt(work_queue)

	def is_faces_alike(self, input_stat):
		return input_stat > 0.80

	def extract_embeddings(self, face_queue):
		log.info("PID: {} - Starting Extract Embeddings.".format(os.getpid()))
		faces_processed = 0
		log.info("Current Total Faces: {}".format(face_queue.qsize()))
		while True:
			try:
				face_obj = face_queue.get(block=True)
				if faces_processed % 1000 == 0:
					log.info("PID: {} - Estimated Faces To Go - {}".format(os.getpid(), face_queue.qsize()))
				if face_obj is None:
					log.info("PID:{} - Completed Their Queue.".format(os.getpid()))
					return
			except Exception as e:
				log.error("PID:{} - Queue Error - {}".format(os.getpid(), e))
				break

			log.debug("PID: {} - Processing - {}".format(os.getpid(), face_obj.FaceID))
			try:
				face_embed_obj = DeepFace.represent(np.array(Image.open(face_obj.Image)), detector_backend="skip", model_name="Facenet512")
				face_embed_as_list = list(face_embed_obj[0]["embedding"])
				did_insert = FacialEmbeddings.objects(FaceEntry=face_obj).modify(upsert=True, new=True, Facenet512=face_embed_as_list)
				log.debug("Added New FacialEmbeddings - {}".format(did_insert))
			except ValueError as e:
				log.error("FaceRepresentError-ValueError: {}".format(e))
				face_queue.task_done()
				faces_processed += 1
				continue
			except Exception as e:
				log.error("FaceRepresentError-Exception: {}".format(e))
				face_queue.task_done()
				faces_processed += 1
				continue
			face_queue.task_done()
			faces_processed += 1

	def process_file(self, file_queue):
		log.info("PID: {} - Starting Process Files.".format(os.getpid()))
		while True:
			try:
				file_path = file_queue.get(block=True)
				log.info("PID:{} - Processing File - {}".format(os.getpid(), file_path))
				if file_path is None:
					log.info("PID:{} - Completed Their Queue.".format(os.getpid()))
					break
			except Exception as e:
				log.error("PID:{} - Queue Error - {}".format(os.getpid(), e))
				break
			try:
				file = os.path.basename(file_path)
				try:
					file_entry = Files.objects(FileName=file).modify(upsert=True, new=True, set_on_insert__FileID=uuid.uuid4(), set_on_insert__FilePath='PathPlaceHolder-'+str(uuid.uuid4()))
					if file_entry.Processed is None or file_entry.Processed is False:
						try:
							face_objs = DeepFace.extract_faces(file_path, detector_backend='opencv')

							for face_obj in face_objs:
								image = Image.fromarray((face_obj['face'] * 255).astype(np.uint8))
								image_buffer = io.BytesIO()
								image.save(image_buffer, format="JPEG")

								current_face = Faces.objects.create(FaceID=uuid.uuid4(), Image=image_buffer)
								del image
								del image_buffer

								FileFaceRelations.objects(File=file_entry).modify(upsert=True, new=True, set_on_insert__FileRelationID=uuid.uuid4(), add_to_set__Faces=current_face)
								try:
									face_embed_obj = DeepFace.represent(face_obj['face'], detector_backend='skip', model_name='VGG-Face', enforce_detection=False)
									FacialEmbeddings.objects(FaceEntry=current_face).modify(upset=True, new=True, set_on_insert__FaceEmbedID=uuid.uuid4(), set_on_insert__FaceEntry=current_face, VGGFace=face_embed_obj[0]["embedding"])
								except Exception as e:
									log.error("FaceEmbeddingError: {}".format(e))
							new_location = shutil.move(file_path, self.face_path)
						except ValueError as e:
							new_location = shutil.move(file_path, self.no_face_path)
							log.debug("FaceExtract: {}".format(e))
						except Exception as e:
							new_location = shutil.move(file_path, self.no_face_path)
							log.error("OtherError: {}".format(e))
						file_entry.DateProcessed = datetime.datetime.now()
						file_entry.Processed = True
						file_entry.FilePath = new_location
						file_entry.save()
				except Exception as e:
					log.error("FileLookup: {}".format(e))
			except Exception as e:
				log.error("FileException: {}".format(e))
			file_queue.task_done()
			# progress_bar.update(n=1)

	def compare_faces(self, face_queue):
		log.info("PID: {} - Starting Compare Faces".format(os.getpid()))
		faces_processed = 0
		current_total_faces = Faces.objects.count()
		log.info("Current Total Faces: {}".format(current_total_faces))
		while True:
			try:
				base_face_job = face_queue.get(block=True)
				if faces_processed % 100 == 0:
					log.info("PID: {} - Processed {} Faces - Estimated Total Faces - {}".format(os.getpid(), faces_processed, face_queue.qsize()))
				no_face_count = 0
				if base_face_job is None:
					log.info("PID:{} - Completed Their Queue.".format(os.getpid()))
					return
			except Exception as e:
				log.error("PID:{} - Queue Error - {}".format(os.getpid(), e))
				break
			try:
				base_face_entry = FaceToFaceRelation.objects(ReferenceFace=base_face_job).first()
				if base_face_entry is None:
					base_face_entry = FaceToFaceRelation(FaceToFaceID=uuid.uuid4(), ReferenceFace=base_face_job)
					base_face_entry.save()

				base_face_embedding = list(FacialEmbeddings.objects(FaceEntry=base_face_job).no_dereference().scalar('GhostFaceNet')[0])
				like_faces_count = len(base_face_entry.LikeFaces)
				unlike_faces_count = len(base_face_entry.UnLikeFaces)
				if (like_faces_count + unlike_faces_count) >= (current_total_faces - 100):
					log.info("PID: {} - Face {} - Is Already Completed - Moving On.".format(os.getpid(), base_face_job.FaceID))
					face_queue.task_done()
					continue
				for reference_face in Faces.objects.all():
					in_likeness_cnt = FaceToFaceRelation.objects(ReferenceFace=base_face_job, LikeFaces__in=[reference_face]).count()
					in_un_likeness_cnt = FaceToFaceRelation.objects(ReferenceFace=base_face_job, UnLikeFaces__in=[reference_face]).count()
					if in_likeness_cnt == 0 and in_un_likeness_cnt == 0:
						if reference_face.FaceID != base_face_job.FaceID:
							try:
								reference_face_embedding = list(FacialEmbeddings.objects(FaceEntry=reference_face).no_dereference().scalar('GhostFaceNet')[0])
								results = FacialLikeness.verify(base_face_embedding, reference_face_embedding, distance_metric="euclidean_v2")
								if results["verified"]:
									base_face_entry.update(add_to_set__LikeFaces=reference_face)
									if not FaceToFaceRelation.objects(ReferenceFace=reference_face):
										FaceToFaceRelation.objects(ReferenceFace=reference_face).modify(
											upsert=True, new=True,
											set_on_insert__FaceToFaceID=uuid.uuid4(),
											set_on_insert__ReferenceFace=reference_face,
											add_to_set__LikeFaces=base_face_job)
									else:
										FaceToFaceRelation.objects(ReferenceFace=reference_face).modify(
											upsert=True, new=True, add_to_set__LikeFaces=base_face_job)
								else:
									base_face_entry.update(add_to_set__UnLikeFaces=reference_face)
									if not FaceToFaceRelation.objects(ReferenceFace=reference_face):
										FaceToFaceRelation.objects(ReferenceFace=reference_face).modify(
											upsert=True, new=True,
											set_on_insert__FaceToFaceID=uuid.uuid4(),
											set_on_insert__ReferenceFace=reference_face,
											add_to_set__UnLikeFaces=base_face_job)
									else:
										FaceToFaceRelation.objects(ReferenceFace=reference_face).modify(
											upsert=True, new=True, add_to_set__UnLikeFaces=base_face_job)
							except ValueError as e:
								log.error("No Face Exception - {}".format(e))
								no_face_count += 1
								if no_face_count > 4:
									log.error("No Face Error Count At 4 - Skipping")
									face_queue.task_done()
									break
							except Exception as e:
								log.error("During Reference Iterations - Different Exception - {}".format(e))
						else:
							log.info("Same Face Skipping.")
			except Exception as e:
				log.error("Face Comparison Exception - {}".format(e))
			log.info("PID: {} - Completed Face - {}".format(os.getpid(),base_face_job.FaceID))
			faces_processed += 1
			face_queue.task_done()

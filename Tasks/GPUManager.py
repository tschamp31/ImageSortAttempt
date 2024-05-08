import shutil
import time

from Database import *
from deepface import DeepFace
from PIL import Image
import logging.config
import numpy as np
import os
import io

from Utils import FacialLikeness

log = logging.getLogger(__name__)


class GPUManager:
	def __init__(self, work_queue, instruction_queue, db_queue, input_path, face_path, no_face_path):
		self.input_path = input_path
		self.face_path = face_path
		self.no_face_path = no_face_path
		self.instruction_queue = instruction_queue
		self.db_queue = db_queue
		job_to_do = instruction_queue.get(block=True)
		log.info("GPUManager: {} - Job Set To - {}".format(os.getpid(),job_to_do))
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
		log.info("GPUManager: {} - Starting Extract Embeddings.".format(os.getpid()))
		faces_processed = 0
		log.info("Current Total Faces: {}".format(face_queue.qsize()))
		while True:
			try:
				face_obj = face_queue.get(block=True)
				if faces_processed % 1000 == 0:
					log.info("GPUManager: {} - Estimated Faces To Go - {}".format(os.getpid(), face_queue.qsize()))
				if face_obj is None:
					log.info("GPUManager:{} - Completed Their Queue.".format(os.getpid()))
					return
			except Exception as e:
				log.error("GPUManager:{} - Queue Error - {}".format(os.getpid(), e))
				break

			log.debug("GPUManager: {} - Processing - {}".format(os.getpid(), face_obj.FaceID))
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
		log.info("GPUManager: {} - Starting Process Files.".format(os.getpid()))
		while True:
			try:
				file_path = file_queue.get(block=True)
				log.info("GPUManager:{} - Processing File - {}".format(os.getpid(), file_path))
				if file_path is None:
					log.info("GPUManager:{} - Completed Their Queue.".format(os.getpid()))
					break
			except Exception as e:
				log.error("GPUManager:{} - Queue Error - {}".format(os.getpid(), e))
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
		log.info("GPUManager: {} - Starting Compare Face Part 1 - Waiting {}s".format(os.getpid(), os.getpid()/300))
		#time.sleep(os.getpid() / 300)
		faces_processed = 0
		current_total_faces = Faces.objects.count()
		log.debug("Current Total Faces: {}".format(current_total_faces))
		while True:
			try:
				base_face_job = face_queue.get(block=True)
				start = time.time()
				if faces_processed % 100 == 0:
					log.debug("GPUManager: {} - Processed {} Faces - Estimated Total Faces - {}".format(os.getpid(), faces_processed, face_queue.qsize()))
				no_face_count = 0
				if base_face_job is None:
					log.info("GPUManager:{} - Completed Their Queue.".format(os.getpid()))
					return
				log.info("GPUManager: {} - Starting Face {}".format(os.getpid(), base_face_job.FaceID))
				try:
					base_face_entry = FaceToFaceRelation.objects(ReferenceFace=base_face_job).first()

					base_face_embedding = list(FacialEmbeddings.objects(FaceEntry=base_face_job).scalar('GhostFaceNet')[0])
					like_faces_count = len(base_face_entry.LikeFaces)
					unlike_faces_count = len(base_face_entry.UnLikeFaces)
					if (like_faces_count + unlike_faces_count) >= (current_total_faces - 10):
						log.info("GPUManager: {} - Face {} - Is Already Completed - Moving On.".format(os.getpid(), base_face_job.FaceID))
						faces_processed += 1
						face_queue.task_done()
						continue
					like_face_list = []
					unlike_face_list = []
					# Slower but shouldn't be... sigh.
					# for reference_face in FaceToFaceRelation.objects(Q(LikeFaces__nin=[base_face_job]) & Q(UnLikeFaces__nin=[base_face_job])).scalar("ReferenceFace").all().allow_disk_use(True):
					# TODO: REvert back to original reference face logic.
					# Likely need to make jobs to track.
					# Step 1 -> this compare -> Step 2 DB propagation -> step 3 db job calculation
					for reference_face in Faces.objects().all():
						try:
							if reference_face is not None and reference_face.FaceID != base_face_job.FaceID:
								try:
									reference_face_embedding = list(FacialEmbeddings.objects(FaceEntry=reference_face).scalar('GhostFaceNet')[0])
									results = FacialLikeness.verify(base_face_embedding, reference_face_embedding, distance_metric="euclidean_v2")
									if results["verified"]:
										like_face_list.append(reference_face)
									else:
										unlike_face_list.append(reference_face)
								except ValueError as e:
									log.error("No Face Exception - {}".format(e))
									if no_face_count > 4:
										log.error("No Face Error Count At 4 - Skipping")
										no_face_count += 1
										face_queue.task_done()
										break
							else:
								log.debug("Same Face Skipping.")
						except Exception as e:
							log.error("During Reference Iterations - Different Exception - {}".format(e))

					base_face_entry.update(push_all__LikeFaces=like_face_list, push_all__UnLikeFaces=unlike_face_list)
					# TODO ADD DEDUPLICATION?
					# TODO Fix logic if entry doesn't exist.
					if base_face_job is not None and (len(like_face_list) > 0 or len(unlike_face_list)):
						self.db_queue.put([base_face_job, like_face_list, unlike_face_list])
				except Exception as e:
					log.error("GPUManager: Face Comparison Part 1 Exception - {}".format(e))
			except Exception as e:
				log.error("GPUManager:{} - Queue Error - {}".format(os.getpid(), e))
				break
			log.info("GPUManager: {} - Completed Faces - {} - Took {}s".format(os.getpid(), base_face_job.FaceID, time.time()-start))

			faces_processed += 5
			face_queue.task_done()

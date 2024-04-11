import time
from pymongo import monitoring

from deepface import DeepFace
import os
import filetype
from PIL import Image
import uuid
import numpy as np
import datetime
from mongoengine import *
import io
from multiprocessing import Pool, Queue
from dotenv import dotenv_values
import logging

log = logging.getLogger()
log.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

config = dotenv_values(".env")

working_path = u"C:\\ImageSortAttempt\\Unsorted\\Sorted"
input_directory = u""
output_directory = u"Sorted\\"
input_path = os.path.join(working_path, input_directory)
output_path = os.path.join(working_path, output_directory)

# Using to track stopping points
facesProcessed = 0
facesJob = None

connect(host=config["MONGO_CONN"], uuidRepresentation='standard')


class Faces(DynamicDocument):
	FaceID = UUIDField(required=True, primary_key=True, default=uuid.uuid4())
	Image = ImageField(required=True, unique=True, size=(224, 224, False))
	FriendlyName = StringField(required=False)
	Tags = ListField(StringField())
	DateCreated = DateTimeField(default=datetime.datetime.utcnow(), required=True)
	meta = {
		'indexes': [
			'FriendlyName'
		],
		'collection': 'Faces'
	}


class FacialEmbeddings(DynamicDocument):
	FaceEmbedID: UUIDField(required=True, primary_key=True, default=uuid.uuid4())
	Face: ReferenceField(Faces, PULL)
	VGGFace: ListField(FloatField(), required=False)
	Facenet: ListField(FloatField(), required=False)
	Facenet512: ListField(FloatField(), required=False)
	OpenFace: ListField(FloatField(), required=False)
	DeepFace: ListField(FloatField(), required=False)
	DeepID: ListField(FloatField(), required=False)
	Dlib: ListField(FloatField(), required=False)
	ArcFace: ListField(FloatField(), required=False)
	SFace: ListField(FloatField(), required=False)
	GhostFaceNet: ListField(FloatField(), required=False)
	meta = {
		'indexes': [
			'Face'
		],
		'collection': 'FacialEmbeddings'
	}


class FaceGroups(DynamicDocument):
	FaceGroupID = UUIDField(required=True, primary_key=True, default=uuid.uuid4())
	IncludedFaces = ListField(ReferenceField(Faces, PULL), required=True)
	ExcludedFaces = ListField(ReferenceField(Faces, PULL), required=False)
	FriendlyName = StringField(required=False)
	LikenessThreshold = FloatField(required=True, default=70.0)
	Tags = ListField(StringField())
	DateCreated = DateTimeField(default=datetime.datetime.utcnow(), required=True)
	DateUpdated = DateTimeField(required=False)
	meta = {
		'collection': 'FaceGroups'
	}


class Files(DynamicDocument):
	FileID = UUIDField(required=True, primary_key=True, default=uuid.uuid4())
	FileName = StringField(required=True, unique=False)
	FilePath = StringField(required=True, unique=True)
	Processed = BooleanField(required=True, default=False)
	DateProcessed = DateTimeField(default=datetime.datetime.utcnow(), required=True)
	meta = {
		'indexes': [
			'FileName'
		],
		'collection': 'Files'
	}


class FileFaceRelations(DynamicDocument):
	FileRelationID = UUIDField(required=True, primary_key=True, default=uuid.uuid4())
	File = ReferenceField(Files, PULL, required=True, unique=True)
	Faces = ListField(ReferenceField(Faces, PULL), required=True)
	meta = {
		'collection': 'FileFaceRelations'
	}


class FaceToFaceRelation(DynamicDocument):
	FaceToFaceID = UUIDField(required=True, primary_key=True, default=uuid.uuid4())
	ReferenceFace = ReferenceField(Faces, PULL, required=True, unique=True)
	LikeFaces = ListField(ReferenceField(Faces, PULL), required=False)
	UnLikeFaces = ListField(ReferenceField(Faces, PULL), required=False)
	LikenessThreshold = FloatField(required=True, default=70.0)
	meta = {
		'collection': 'FaceToFaceRelation'
	}


class JobFaces(DynamicDocument):
	FacesProcessedID = UUIDField(required=True, primary_key=True, default=uuid.uuid4())
	FacesDone = ListField(ReferenceField(Faces, PULL), required=False)
	FacesToDo = ListField(ReferenceField(Faces, PULL), required=False)
	DoneCount = IntField(min_value=0, default=0)
	ToDoCount = IntField(min_value=0)
	StartDate = DateTimeField(default=datetime.datetime.utcnow(), required=False)
	FinishDate = DateTimeField(required=False)
	JobCompleted = BooleanField(required=False, default=False)
	meta = {
		'collection': 'Job_Faces'
	}


class JobFiles(DynamicDocument):
	FilesProcessedID = UUIDField(required=True, primary_key=True, default=uuid.uuid4())
	Done = ListField(ReferenceField(Files, PULL), required=True)
	ToDo = ListField(ReferenceField(Files, PULL), required=True)
	DoneCount = IntField(min_value=0, default=0)
	ToDoCount = IntField(min_value=0)
	StartDate = DateTimeField(default=datetime.datetime.utcnow(), required=False)
	FinishDate = DateTimeField(required=False)
	meta = {
		'collection': 'Job_Files'
	}


backends = [
	'opencv',
	'ssd',
	'dlib',
	'mtcnn',
	'retinaface',
	'mediapipe',
	'yolov8',
	'yunet',
	'fastmtcnn',
	'ghosefacenet'
]

acceptableFiles = [
	'jpg',
	'jpeg',
	'png',
	'apng',
	'jpx',
	'tif',
	'heic',
	'webp'
]


class CommandLogger(monitoring.CommandListener):

	def started(self, event):
		log.debug("Command {0.command_name} with request id "
		          "{0.request_id} started on server "
		          "{0.connection_id}".format(event))

	def succeeded(self, event):
		log.debug("Command {0.command_name} with request id "
		          "{0.request_id} on server {0.connection_id} "
		          "succeeded in {0.duration_micros} "
		          "microseconds".format(event))

	def failed(self, event):
		log.debug("Command {0.command_name} with request id "
		          "{0.request_id} on server {0.connection_id} "
		          "failed in {0.duration_micros} "
		          "microseconds".format(event))


monitoring.register(CommandLogger())


def process_file(file_queue):
	log.info(f"PID:{os.getpid()} - Working")
	while True:
		file_path = file_queue.get(block=True)
		log.info(f"PID:{os.getpid()} - Processing - {file_path}")
		if file_path is None:
			break
		file = os.path.basename(file_path)
		try:
			Files.objects.get(FileName=file, FilePath=file_path, Processed=True)
		except DoesNotExist:
			try:

				face_objs = DeepFace.extract_faces(file_path)
				os.rename(file_path, os.path.join(output_path, file))

				currentFile = Files(FileID=uuid.uuid4(), FileName=file, FilePath=file_path, Processed=True)
				currentFile.save()
				for face_obj in face_objs:

					image = Image.fromarray((face_obj['face'] * 255).astype(np.uint8))
					imageBuffer = io.BytesIO()
					image.save(imageBuffer, format="JPEG")

					currentFace = Faces(FaceID=uuid.uuid4(), Image=imageBuffer)
					currentFace.save()
					try:
						face_embed_obj = DeepFace.represent(face_obj['face'])
						FacialEmbeddings.objects(Face=Faces).modify(upset=True, new=True, set_on_insert__FaceEmbedID=uuid.uuid4(), VGGFace=face_embed_obj[0]["embedding"])
					except Exception as e:
						log.error(f"FaceEmbeddingError: {e}")
						pass
					FileFaceRelations.objects(File=currentFile).modify(upsert=True, new=True, set_on_insert__FileRelationID=uuid.uuid4(), add_to_set__Faces=currentFace)
			except ValueError as e:
				log.error(f"FaceExtract: {e}")
				pass
			except Exception as e:
				log.error(f"OtherException: {e}")
				pass
			pass
		except Exception as e:
			log.error(f"FileException: {e}")
			pass


def main():
	max_processes = 4
	file_queue = Queue()
	worker_pool = Pool(max_processes, process_file, (file_queue,))
	if not os.path.isdir(input_path):
		raise IOError("Input Path not found: " + input_path)
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	for root, dirs, files in os.walk(input_path):
		for file in files:
			filePath = os.path.join(root, file)
			kind = filetype.guess(filePath)
			if kind is not None and kind.extension.lower() in acceptableFiles:
				file_queue.put(filePath)

	file_queue.close()
	file_queue.join_thread()

	worker_pool.close()
	worker_pool.join()


def is_faces_alike(input_stat):
	return input_stat > 0.80


def compare_faces(job_id):
	# df = DeepFace(model_name=ModelName.VGGFace, detector_backend=DetectorBackend.OpenCV, align=False)
	faces_job = JobFaces.objects(FacesProcessedID=uuid.UUID(job_id)).first()
	if faces_job is None:
		faces_job = JobFaces(FacesProcessedID=uuid.uuid4(), StartDate=datetime.datetime.utcnow())
		faces_job.save()

	if not faces_job.JobCompleted:
		JobFaces(FacesProcessedID=faces_job.FacesProcessedID)
		faces_job.ToDoCount = Faces.objects.count()
		facesProcessed = faces_job.DoneCount

		if faces_job.DoneCount == 0:
			print(
				f"Starting Job - {faces_job.FacesProcessedID} - {(faces_job.DoneCount / faces_job.ToDoCount) * 100}% Done.")
		else:
			print(
				f"Resuming Job - {faces_job.FacesProcessedID} - - {(faces_job.DoneCount / faces_job.ToDoCount) * 100}% Done.")

		for baseFace in Faces.objects:
			noFaceCount = 0
			baseFaceEntry = FaceToFaceRelation.objects(ReferenceFace=baseFace).first()
			if baseFaceEntry is None:
				baseFaceEntry = FaceToFaceRelation(FaceToFaceID=uuid.uuid4(),
				                                   ReferenceFace=baseFace)
				baseFaceEntry.save()

			if (facesProcessed % 100 == 0):
				faces_job.DoneCount = facesProcessed
				faces_job.save()
				print(
					f"***Still Running***\nFacesProcessed:{facesProcessed} - {(faces_job.DoneCount / faces_job.ToDoCount) * 100}% Done.")
			for referenceFace in Faces.objects:
				in_likeness_cnt = FaceToFaceRelation.objects(ReferenceFace=baseFace,
				                                             LikeFaces__in=[referenceFace]).count()
				in_un_likeness_cnt = FaceToFaceRelation.objects(ReferenceFace=baseFace,
				                                                UnLikeFaces__in=[
					                                                referenceFace]).count()
				# print("Likeness: " + str(in_likeness_cnt) + " Unlikeness: " + str(inUnLikenessCnt))
				if in_likeness_cnt == 0 and in_un_likeness_cnt == 0:
					if referenceFace.FaceID != baseFace.FaceID:
						try:
							results = DeepFace.verify(np.array(Image.open(baseFace.Image)),
							                          np.array(Image.open(referenceFace.Image)))

							if is_faces_alike(results['distance']):
								baseFaceEntry.update(add_to_set__LikeFaces=referenceFace)
								if not FaceToFaceRelation.objects(ReferenceFace=referenceFace):
									FaceToFaceRelation.objects(ReferenceFace=referenceFace).modify(
										upsert=True, new=True,
										set_on_insert__FaceToFaceID=uuid.uuid4(),
										set_on_insert__ReferenceFace=referenceFace,
										add_to_set__LikeFaces=baseFace)
								else:
									FaceToFaceRelation.objects(ReferenceFace=referenceFace).modify(
										upsert=True, new=True, add_to_set__LikeFaces=baseFace)
							else:
								baseFaceEntry.update(add_to_set__UnLikeFaces=referenceFace)
								if not FaceToFaceRelation.objects(ReferenceFace=referenceFace):
									FaceToFaceRelation.objects(ReferenceFace=referenceFace).modify(
										upsert=True, new=True,
										set_on_insert__FaceToFaceID=uuid.uuid4(),
										set_on_insert__ReferenceFace=referenceFace,
										add_to_set__UnLikeFaces=baseFace)
								else:
									FaceToFaceRelation.objects(ReferenceFace=referenceFace).modify(
										upsert=True, new=True, add_to_set__UnLikeFaces=baseFace)

						# print("Done Comparing BaseFace: " + str(baseFace.FaceID) + " and ReferenceFace: " + str(referenceFace.FaceID) + " Likeness: " + str(results['distance']))
						except ValueError:
							# print("No Face Detected")
							noFaceCount += 1
							if noFaceCount > 4:
								# print("Going To Next Base Face.")
								break
						except Exception as e:
							print("Different Exception? " + e)
				else:
					facesProcessed += 1
			# print("Same Face Skipping.")
			facesProcessed += 1
		faces_job.JobCompleted = True
		faces_job.FinishDate = datetime.datetime.utcnow()
		faces_job.save()
	else:
		print(f"Job: {faces_job.FacesProcessedID} is already completed.")


def on_exit(sig, func=None):
	facesJob.DoneCount = facesProcessed
	facesJob.save()
	print("Saving Faces....")
	time.sleep(10)  # so you can see the message before program exits


if __name__ == '__main__':
	print("Starting....")
	main()

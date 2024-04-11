from deepface import *
from deepface.commons.errors import *
from deepface.commons.options import *
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

config = dotenv_values(".env")

working_path = u"D:\\PhonePhotos\\Facebook"
input_directory = u""
output_directory = u"Sorted\\"
input_path = os.path.join(working_path, input_directory)
output_path = os.path.join(working_path, output_directory)

# Using to track stopping points
facesProcessed = 0
facesJob = None

connect(host=config["mongoConnStr"], uuidRepresentation='standard')


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


class Job_Faces(DynamicDocument):
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


class Job_Files(DynamicDocument):
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


def processFile(file_queue):
	print(os.getpid(), "Working")
	df = DeepFace(model_name=ModelName.VGGFace, detector_backend=DetectorBackend.OpenCV, align=False, enforce_detection=True, target_size=(224, 224))
	while True:
		# print(os.getpid(), "processing", file)
		try:
			filePath = file_queue.get(block=True)
			if filePath is None:
				break
			file = os.path.basename(filePath)
			Files.objects.get(FileName=file, FilePath=filePath, Processed=True)
		except DoesNotExist:
			try:
				df.set_variable("img_path", filePath)
				face_objs = df.extract_faces()
				os.rename(filePath, os.path.join(output_path, file))
				currentFile = Files(FileID=uuid.uuid4(), FileName=file, FilePath=filePath, Processed=True)
				currentFile.save()
				for face_obj in face_objs:
					image = Image.fromarray((face_obj['face'] * 255).astype(np.uint8))
					imageBuffer = io.BytesIO()
					image.save(imageBuffer, format="JPEG")
					currentFace = Faces(FaceID=uuid.uuid4(), Image=imageBuffer)
					currentFace.save()
					FileFaceRelations.objects(File=currentFile).modify(upsert=True, new=True, set_on_insert__FileRelationID=uuid.uuid4(), add_to_set__Faces=currentFace)
			except NoFaceDetectedException as e:
				pass
				# print(e)
			except Exception as e:
				print("Other Exception: " + str(e))
		except Exception as e:
			print("File Exception? - " + str(e))


def main():
	max_processes = 4
	file_queue = Queue()
	worker_pool = Pool(max_processes, processFile, (file_queue,))
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


def isFacesAlike(inputStat):
	return inputStat > 0.80


def compareFaces(JobID):
	df = DeepFace(model_name=ModelName.VGGFace, detector_backend=DetectorBackend.OpenCV, align=False)
	facesJob = Job_Faces.objects(FacesProcessedID=uuid.UUID(JobID)).first()
	if facesJob is None:
		facesJob = Job_Faces(FacesProcessedID=uuid.uuid4(), StartDate=datetime.datetime.utcnow())
		facesJob.save()

	if not facesJob.JobCompleted:
		Job_Faces(FacesProcessedID=facesJob.FacesProcessedID)
		facesJob.ToDoCount = Faces.objects.count()
		facesProcessed = facesJob.DoneCount

		if facesJob.DoneCount == 0:
			print(f"Starting Job - {facesJob.FacesProcessedID} - {(facesJob.DoneCount/facesJob.ToDoCount) * 100}% Done.")
		else:
			print(f"Resuming Job - {facesJob.FacesProcessedID} - - {(facesJob.DoneCount/facesJob.ToDoCount) * 100}% Done.")

		for baseFace in Faces.objects:
			noFaceCount = 0
			baseFaceEntry = FaceToFaceRelation.objects(ReferenceFace=baseFace).first()
			if baseFaceEntry is None:
				baseFaceEntry = FaceToFaceRelation(FaceToFaceID=uuid.uuid4(), ReferenceFace=baseFace)
				baseFaceEntry.save()
			df.set_variable("img_path", np.array(Image.open(baseFace.Image)))
			if (facesProcessed % 100 == 0):
				facesJob.DoneCount = facesProcessed
				facesJob.save()
				print(f"***Still Running***\nFacesProcessed:{facesProcessed} - {(facesJob.DoneCount/facesJob.ToDoCount) * 100}% Done.")
			for referenceFace in Faces.objects:
				inLikenessCnt = FaceToFaceRelation.objects(ReferenceFace=baseFace, LikeFaces__in=[referenceFace]).count()
				inUnLikenessCnt = FaceToFaceRelation.objects(ReferenceFace=baseFace, UnLikeFaces__in=[referenceFace]).count()
				# print("Likeness: " + str(inLikenessCnt) + " Unlikeness: " + str(inUnLikenessCnt))
				if (inLikenessCnt == 0 and inUnLikenessCnt == 0):
					if (referenceFace.FaceID != baseFace.FaceID):
						try:
							df.set_variable("img2_path", np.array(Image.open(referenceFace.Image)))
							results = df.verify()

							if isFacesAlike(results['distance']):
								baseFaceEntry.update(add_to_set__LikeFaces=referenceFace)
								if not FaceToFaceRelation.objects(ReferenceFace=referenceFace):
									FaceToFaceRelation.objects(ReferenceFace=referenceFace).modify(upsert=True, new=True, set_on_insert__FaceToFaceID=uuid.uuid4(), set_on_insert__ReferenceFace=referenceFace, add_to_set__LikeFaces=baseFace)
								else:
									FaceToFaceRelation.objects(ReferenceFace=referenceFace).modify(upsert=True, new=True, add_to_set__LikeFaces=baseFace)
							else:
								baseFaceEntry.update(add_to_set__UnLikeFaces=referenceFace)
								if not FaceToFaceRelation.objects(ReferenceFace=referenceFace):
									FaceToFaceRelation.objects(ReferenceFace=referenceFace).modify(upsert=True, new=True, set_on_insert__FaceToFaceID=uuid.uuid4(), set_on_insert__ReferenceFace=referenceFace, add_to_set__UnLikeFaces=baseFace)
								else:
									FaceToFaceRelation.objects(ReferenceFace=referenceFace).modify(upsert=True, new=True, add_to_set__UnLikeFaces=baseFace)

							# print("Done Comparing BaseFace: " + str(baseFace.FaceID) + " and ReferenceFace: " + str(referenceFace.FaceID) + " Likeness: " + str(results['distance']))
						except NoFaceDetectedException:
							# print("No Face Deteced")
							noFaceCount += 1
							if (noFaceCount > 4):
								# print("Going To Next Base Face.")
								break
						except Exception as e:
							print("Different Exception? " + e)
				else:
					facesProcessed += 1
					# print("Same Face Skipping.")
			facesProcessed += 1
		facesJob.JobCompleted = True
		facesJob.FinishDate = datetime.datetime.utcnow()
		facesJob.save()
	else:
		print(f"Job: {facesJob.FacesProcessedID} is already completed.")


def on_exit(sig, func=None):
	facesJob.DoneCount = facesProcessed
	facesJob.save()
	print("Saving Faces....")
	time.sleep(10)  # so you can see the message before program exits


if __name__ == '__main__':
	print("Starting....")
	main()

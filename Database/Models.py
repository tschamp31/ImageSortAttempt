from enum import auto, IntEnum
from mongoengine import *
import datetime
import uuid


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
	FaceEmbedID = UUIDField(required=True, primary_key=True, default=uuid.uuid4())
	FaceEntry = ReferenceField(Faces, DO_NOTHING, required=True, unique=True)
	VGGFace = ListField(FloatField())
	Facenet = ListField(FloatField(), required=False)
	Facenet512 = ListField(FloatField(), required=False)
	OpenFace = ListField(FloatField(), required=False)
	DeepFace = ListField(FloatField(), required=False)
	DeepID = ListField(FloatField(), required=False)
	Dlib = ListField(FloatField(), required=False)
	ArcFace = ListField(FloatField(), required=False)
	SFace = ListField(FloatField(), required=False)
	GhostFaceNet = ListField(FloatField(), required=False)
	meta = {
		'indexes': [
			'FaceEntry'
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
		'indexes': [
			'File',
			'Faces'
		],
		'collection': 'FileFaceRelations'
	}


class FaceToFaceRelation(DynamicDocument):
	FaceToFaceID = UUIDField(required=True, primary_key=True, default=uuid.uuid4())
	ReferenceFace = ReferenceField(Faces, PULL, required=True, unique=True)
	LikeFaces = ListField(ReferenceField(Faces, PULL), required=False)
	UnLikeFaces = ListField(ReferenceField(Faces, PULL), required=False)
	LikenessThreshold = FloatField(required=True, default=70.0)
	meta = {
		'indexes': [
			'ReferenceFace'
		],
		'collection': 'FaceToFaceRelation'
	}


class Status(IntEnum):
	NotStarted = auto()
	Started = auto()
	Processing = auto()
	Complete = auto()

	def use_name(self):
		return self.name

	def describe(self):
		return self.name, self.value

	@classmethod
	def get_options(cls):
		return [name for name, member in cls.__members__.items()]


class JobStatus(DynamicDocument):
	JobStatusID = UUIDField(required=True, primary_key=True, default=uuid.uuid4())
	FriendlyName = StringField(required=False)
	JobStatus = EnumField(Status, default=Status.NotStarted)
	DoneCount = IntField(min_value=0, default=0)
	ToDoCount = IntField(min_value=0)
	StartDate = DateTimeField(default=datetime.datetime.utcnow(), required=False)
	FinishDate = DateTimeField(required=False)
	meta = {
		'indexes': [
			'FriendlyName'
		],
		'collection': 'JobStatus'
	}


class JobFaces(DynamicDocument):
	FacesProcessedID = UUIDField(required=True, primary_key=True, default=uuid.uuid4())
	FriendlyName = StringField(required=False)
	FacesDone = ListField(ReferenceField(Faces, PULL), required=False)
	FacesToDo = ListField(ReferenceField(Faces, PULL), required=False)
	DoneCount = IntField(min_value=0, default=0)
	ToDoCount = IntField(min_value=0)
	StartDate = DateTimeField(default=datetime.datetime.utcnow(), required=False)
	FinishDate = DateTimeField(required=False)
	JobCompleted = BooleanField(required=False, default=False)
	meta = {
		'indexes': [
			'FriendlyName'
		],
		'collection': 'JobFaces'
	}


class JobFilesFaces(DynamicDocument):
	FileJobID = UUIDField(required=True, primary_key=True, default=uuid.uuid4())
	WorkingFolder = StringField(required=True)
	FaceFolder = StringField(required=True, default=u"Face\\")
	FacelessFolder = StringField(required=True, default=u"NoFace\\")
	JobStatus = ReferenceField(JobStatus, DO_NOTHING, required=False)
	meta = {
		'indexes': [
			'WorkingFolder'
		],
		'collection': 'JobFilesFaces'
	}
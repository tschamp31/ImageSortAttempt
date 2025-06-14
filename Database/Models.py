import datetime
import uuid

from sqlalchemy import Column, UUID, Table, PrimaryKeyConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.mssql import *

Base = declarative_base()

# Define the Author model
Faces = Table(
	'Faces',
	Base.metadata,
	Column('FaceID', UUID, primary_key=True, default=uuid.uuid4),
	Column('Image', IMAGE , default=datetime.datetime.utcnow),
	Column('FriendlyName', DATETIME2, default=datetime.datetime.utcnow),
	Column('Tags', NVARCHAR(500), nullable=True),
	PrimaryKeyConstraint("FaceID", mssql_clustered=True),
)

VGGFaces = Table(
	'VGGFaces',
	Base.metadata,
	Column('VGGFaceID', UUID, primary_key=True, default=uuid.uuid4),
	Column('VGGEmbedding', IMAGE, default=datetime.datetime.utcnow),
)

Facenet = Table(
	'Facenet',
	Base.metadata,
	Column('FacenetID', UUID, primary_key=True, default=uuid.uuid4),
	Column('FacenetEmbedding', IMAGE, default=datetime.datetime.utcnow),
)

Facenet512 = Table(
	'Facenet512',
	Base.metadata,
	Column('Facenet512ID', UUID, primary_key=True, default=uuid.uuid4),
	Column('Facenet512Embedding', IMAGE, default=datetime.datetime.utcnow),
)

OpenFace = Table(
	'OpenFace',
	Base.metadata,
	Column('OpenFaceID', UUID, primary_key=True, default=uuid.uuid4),
	Column('OpenFaceEmbedding', IMAGE, default=datetime.datetime.utcnow),
)
DeepFace = Table(
	'DeepFace',
	Base.metadata,
	Column('DeepFaceID', UUID, primary_key=True, default=uuid.uuid4),
	Column('DeepFaceEmbedding', IMAGE, default=datetime.datetime.utcnow),
)
Dlib = Table(
	'Dlib',
	Base.metadata,
	Column('DlibID', UUID, primary_key=True, default=uuid.uuid4),
	Column('DlibEmbedding', IMAGE, default=datetime.datetime.utcnow),
)
ArcFace = Table(
	'ArcFace',
	Base.metadata,
	Column('ArcFaceID', UUID, primary_key=True, default=uuid.uuid4),
	Column('ArcFaceEmbedding', IMAGE, default=datetime.datetime.utcnow),
)
SFace = Table(
	'SFace',
	Base.metadata,
	Column('SFaceID', UUID, primary_key=True, default=uuid.uuid4),
	Column('SFaceEmbedding', IMAGE, default=datetime.datetime.utcnow),
)
GhostFaceNet = Table(
	'GhostFaceNet',
	Base.metadata,
	Column('GhostFaceNetID', UUID, primary_key=True, default=uuid.uuid4),
	Column('GhostFaceNetEmbedding', IMAGE, default=datetime.datetime.utcnow),
)

FaceEmbedding = Table(
	'FaceEmbedding',
	Base.metadata,
	Column('FaceID', UUID, primary_key=True, nullable=False),
	Column('VGGFaceID', UUID, nullable=True),
	Column('FacenetID', UUID, nullable=True),
	Column('Facenet512ID', UUID, nullable=True),
	Column('OpenFaceID', UUID, nullable=True),
	Column('DeepFaceID', UUID, nullable=True),
	Column('DlibID', UUID, nullable=True),
	Column('ArcFaceID', UUID, nullable=True),
	Column('SFaceID', UUID, nullable=True),
	Column('GhostFaceNetID', UUID, nullable=True),
	PrimaryKeyConstraint("FaceID", mssql_clustered=True),
)

Files = Table(
	'Files',
	Base.metadata,
	Column('FileID', UUID, primary_key=True, default=uuid.uuid4),
	Column('FileName', NVARCHAR, nullable=False),
	Column('FilePath', NVARCHAR, nullable=False),
)

FaceFileRelationship = Table(
	'FaceFileRelationship',
	Base.metadata,
	Column('FileID', UUID, nullable=False),
	Column('FaceID', UUID, nullable=False),
)

FileProcessingJobs = Table (
	'FileProcessingJobs',
	Base.metadata,
Column('FileProcessingJobID', UUID, default=uuid.uuid4),
	Column('FileID', UUID, nullable=False),
	Column('Processed', BIT, default=False),
	Column('DateCreated', DATETIME2, default=datetime.datetime.utcnow),
	Column('DateProcessed', DATETIME2, nullable=True),
	PrimaryKeyConstraint("FileProcessingJobID", mssql_clustered=True),
)

FolderProcessingJobs = Table(
	'FolderProcessingJobs',
	Base.metadata,
Column('FolderProcessingJobID', UUID, default=uuid.uuid4),
	Column('Processed', BIT, default=False),
	Column('DateCreated', DATETIME2, default=datetime.datetime.utcnow),
	Column('DateProcessed', DATETIME2, nullable=True),
	PrimaryKeyConstraint("FileProcessingJobID", mssql_clustered=True),
)

FaceProcessingJobs = Table(
	'FaceProcessingJobs',
	Base.metadata,
Column('FaceProcessingJobID', UUID, default=uuid.uuid4),
	Column('FaceID', UUID, nullable=False),
	Column('EmbeddingType',TINYINT, nullable=False),
	Column('Processed', BIT, default=False),
	Column('DateCreated', DATETIME2, default=datetime.datetime.utcnow),
	Column('DateProcessed', DATETIME2, nullable=True),
	PrimaryKeyConstraint("FileProcessingJobID", mssql_clustered=True),
)

# class FaceGroups(DynamicDocument):
# 	FaceGroupID = UUIDField(required=True, primary_key=True, default=uuid.uuid4())
# 	IncludedFaces = ListField(ReferenceField(Faces, PULL), required=True)
# 	ExcludedFaces = ListField(ReferenceField(Faces, PULL), required=False)
# 	FriendlyName = StringField(required=False)
# 	LikenessThreshold = FloatField(required=True, default=70.0)
# 	Tags = ListField(StringField())
# 	DateCreated = DateTimeField(default=datetime.datetime.utcnow(), required=True)
# 	DateUpdated = DateTimeField(required=False)
# 	meta = {
# 		'collection': 'FaceGroups'
# 	}
#
# class FaceToFaceRelation(DynamicDocument):
# 	FaceToFaceID = UUIDField(required=True, primary_key=True, default=uuid.uuid4())
# 	ReferenceFace = ReferenceField(Faces, PULL, required=True, unique=True)
# 	LikeFaces = ListField(ReferenceField(Faces, PULL), required=False)
# 	UnLikeFaces = ListField(ReferenceField(Faces, PULL), required=False)
# 	LikenessThreshold = FloatField(required=True, default=70.0)
# 	meta = {
# 		'indexes': [
# 			'ReferenceFace'
# 		],
# 		'collection': 'FaceToFaceRelation'
# 	}
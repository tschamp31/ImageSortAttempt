import timeit
from functools import reduce

from deepface import *
from pymongo import MongoClient
from gridfs import GridFS
from multiprocessing import Pool, Queue, Pipe, JoinableQueue
import logging.config
import filetype
import cupy as np
import yaml
import os

from Database import *
from Tasks import GPUManager, DBManager
from Utils import FacialLikeness

with open("logConfig.yaml", 'r') as stream:
	# noinspection PyyamlLoad
	logging_config = yaml.load(stream, Loader=yaml.loader.UnsafeLoader)
	logging.config.dictConfig(logging_config)

log = logging.getLogger()

# deepFaceLog = logging.getLogger("DeepFace")
# deepFaceLog
# logging.basicConfig(filename='./Logs/FaceLogs.log', level=logging.DEBUG, format="%(asctime)s | %(levelname)-8s | %(processName)s | %(message)s")
# tf.get_logger().setLevel(logging.DEBUG)
# log.setLevel(logging.INFO)

working_path = u"H:\\Everything"
input_directory = u""
face_directory = u"Sorted\\"
no_face_directory = u"NoFace\\"
input_path = os.path.join(working_path, input_directory)
face_path = os.path.join(working_path, face_directory)
no_face_path = os.path.join(working_path, no_face_directory)

# Using to track stopping points
items_total = 0
items_processed = 0
items_unprocessed = 0
faces_job = None


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

acceptable_files = [
	'jpg',
	'jpeg',
	'png',
	'apng',
	'jpx',
	'tif',
	'heic',
	'webp'
]

# path1 = Path("D:\\PhonePhotos\\Facebook\\")
# path2 = Path("D:\\PhonePhotos\\Screenshots\\")
# progress_bar = tqdm(total=sum(1 for _ in path1.rglob('*')) + sum(1 for _ in path2.rglob('*')))
gpu_max_processes = 8
db_max_processes = 2
db_work_queue = JoinableQueue()
gpu_work_queue = JoinableQueue()
db_instruction_queue = Queue()
gpu_instruction_queue = Queue()
# progress_bar = tqdm(total=Faces.objects.count())

def chunks(l, n):
	"""Yield n number of striped chunks from l."""
	for i in range(0, n):
		yield l[i::n]

def send_worker_instructions(instruction, queue, cpu_max):
	for i in range(cpu_max):
		queue.put(instruction)


def send_end_queue_signal(queue, cpu_max):
	for i in range(cpu_max):
		queue.put(None)


def main(job_type):
	gpu_worker_pool = Pool(gpu_max_processes, GPUManager, (gpu_work_queue, gpu_instruction_queue, db_work_queue, input_path, face_path, no_face_path))
	db_worker_pool = Pool(db_max_processes, DBManager, (db_work_queue, "test"))
	send_worker_instructions(job_type, gpu_instruction_queue, gpu_max_processes)
	gpu_instruction_queue.close()
	match job_type:
		case "ProcessFiles":
			if not os.path.isdir(input_path):
				raise IOError("Input Path not found: " + input_path)
			if not os.path.exists(face_path):
				os.makedirs(face_path)
			if not os.path.exists(no_face_path):
				os.makedirs(no_face_path)

			for root, dirs, files in os.walk(input_path):
				for file in files:
					file_path = os.path.join(root, file)
					kind = filetype.guess(file_path)
					if kind is not None and kind.extension.lower() in acceptable_files and "Sorted" not in file_path:
						gpu_work_queue.put(file_path)
		case "CompareFaces":
			full_faces_table = Faces.objects[900:].all().allow_disk_use(True)
			for chunk in full_faces_table:
				gpu_work_queue.put(chunk)
		case "ExtractEmbeddings":
			for face in Faces.objects.all():
				gpu_work_queue.put(face)

	send_end_queue_signal(gpu_work_queue, gpu_max_processes)
	gpu_work_queue.close()
	gpu_work_queue.join_thread()
	gpu_worker_pool.close()
	gpu_worker_pool.join()
	send_end_queue_signal(db_work_queue, db_max_processes)
	db_work_queue.close()
	db_work_queue.join_thread()
	db_worker_pool.close()
	db_worker_pool.join()
	exit()

# def on_exit(sig, func=None):
# 	if faces_job is not None:
# 		faces_job.DoneCount = items_processed
# 		faces_job.save()
# 	print("Saving Faces....")
# 	time.sleep(10)  # so you can see the message before program exits

def run_benchmark():
	embedding_set_1 = FacialEmbeddings.objects[1:2].scalar('VGGFace')[0]
	embedding_set_2 = FacialEmbeddings.objects[5:6].scalar('VGGFace')[0]
	dist_setting = "euclidean_v1"
	print("Distance Metric: {} - AvgTime: {}μs ".format(dist_setting,timeit.timeit('FacialLikeness.verify(embedding_set_1, embedding_set_2, distance_metric=dist_setting)', globals=globals(), number=1000)))
	dist_setting = "euclidean_v2"
	print("Distance Metric: {} - AvgTime: {}μs ".format(dist_setting,timeit.timeit('FacialLikeness.verify(embedding_set_1, embedding_set_2, distance_metric=dist_setting)', globals=globals(), number=1000)))
	dist_setting = "cosine_v1"
	print("Distance Metric: {} - AvgTime: {}μs ".format(dist_setting,timeit.timeit('FacialLikeness.verify(embedding_set_1, embedding_set_2, distance_metric=dist_setting)', globals=globals(), number=1000)))
	dist_setting = "cosine_v2"
	print("Distance Metric: {} - AvgTime: {}μs ".format(dist_setting,timeit.timeit('FacialLikeness.verify(embedding_set_1, embedding_set_2, distance_metric=dist_setting)', globals=globals(), number=1000)))


# Function to update metadata for all documents
def update_metadata_for_all_documents():
	# Connect to MongoDB
	client = MongoClient("mongodb://ImageSort:ImageSort123!!@192.168.1.8:27017/ImageSort?authSource=admin&readPreference=primary")
	db = client['ImageSort']
	gridfs = GridFS(db)

	log.info("We in dis bitch")
	for file_info in db.images.files.find():
		file_id = file_info['_id']
		db.images.files.update_one({'_id': file_id}, {'$unset': {"metadata": ""}})

def like_faces_chunks(l):
	"""Yield n number of striped chunks from l."""
	for i in range(0, l.count()):
		yield l[i].LikeFaces

def unlike_faces_chunks(l):
	"""Yield n number of striped chunks from l."""
	for i in range(0, l.count()):
		yield l[i].UnLikeFaces

def age_demographic_test():
	for root, dirs, files in os.walk(".\\Brit"):
		for file in files:
			file_path = os.path.join(root, file)
			log.info("{}".format(file_path))
			demographies = DeepFace.analyze(
				img_path=file_path,
				detector_backend=backends[5],
				enforce_detection=False
			)
			log.info("{}".format(demographies))

if __name__ == '__main__':
	print("Starting....")
	age_demographic_test()
	#main("CompareFaces")

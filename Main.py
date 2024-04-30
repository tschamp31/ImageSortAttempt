import timeit
from multiprocessing import Pool, Queue, Pipe, JoinableQueue
import logging.config
import filetype
import numpy as np
import yaml
import os

from Database import *
from Tasks.FaceManager import FaceManager
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

working_path = u"H:\Everything"
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
max_processes = 8
work_queue = JoinableQueue()
instruction_queue = Queue()
# progress_bar = tqdm(total=Faces.objects.count())


def send_worker_instructions(instruction):
	for i in range(max_processes):
		instruction_queue.put(instruction)


def send_end_queue_signal():
	for i in range(max_processes):
		work_queue.put(None)


def main(job_type):
	worker_pool = Pool(max_processes, FaceManager, (work_queue, instruction_queue, input_path, face_path, no_face_path))
	send_worker_instructions(job_type)
	instruction_queue.close()
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
						work_queue.put(file_path)
		case "CompareFaces":
			for face in Faces.objects.all():
				work_queue.put(face)
		case "ExtractEmbeddings":
			for face in Faces.objects.all():
				work_queue.put(face)

	send_end_queue_signal()
	work_queue.close()
	work_queue.join_thread()
	worker_pool.close()
	worker_pool.join()
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


if __name__ == '__main__':
	print("Starting....")
	main("CompareFaces")

import shutil
import pandas as pd

def copy_directory(src, dest):
	try:
		shutil.copytree(src, dest)
		return 1
	except shutil.Error as e:
		print("Couldn't copy directory. Error: %s" % e)
		return 2
	except OSError as e:
		print("Couldn't copy directory. Error: %s" % e)
		return 3
	
def extract_dataset(source, destination, csv_file, set_type, sample_count=-1):
	csv = pd.read_csv(csv_file, delimiter=",")
	classes = ['Swiping Up', 'Swiping Down', 'Swiping Right', 'Swiping Left', 'Doing other things']

	rows = []

	for current in classes:
		count = 0
		for index, video_id in enumerate(csv["video_id"]):
			if count == sample_count and sample_count != -1:
				break

			if csv["label"][index] == current:
				result = copy_directory(source + str(video_id), destination + str(video_id))
				if result == 1:
					rows.append([video_id, current])
					count += 1

	df = pd.DataFrame(data=rows, columns=["VideoId", "Label"])
	df.set_index("VideoId", inplace=True)
	df.to_csv(f"{set_type}.csv")

def extract_testset(source, destination, csv_file, set_type):
	csv = pd.read_csv(csv_file, delimiter=",")
	rows = []


	for video_id in csv["id"]:
		result = copy_directory(source + str(video_id), destination + str(video_id))
		if result == 1:
			rows.append([video_id])

	df = pd.DataFrame(data=rows, columns=["Id"])
	df.set_index("Id", inplace=True)
	df.to_csv(f"{set_type}.csv")
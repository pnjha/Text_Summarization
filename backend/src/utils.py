from packages import *

def load_data(data_path):
	with open(data_path, 'r') as fp:
		return json.load(fp)

def save_data(data_path,data):
	with open(data_path, 'w') as fp:
		json.dump(data, fp, indent=4, sort_keys=True)

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def show_plot(points, file_path):
    plt.plot(points)
    plt.savefig(file_path)
    plt.show()
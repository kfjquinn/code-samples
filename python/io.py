import mmap

def read_file(file_name, line_fn):
	with open(file_name, mode='r', encoding='utf-8') as handle:
		map_file = mmap.mmap(handle.fileno(), 0, prot=mmap.PROT_READ)
		examples = list(map(line_fn, iter(map_file.readline, b"")))
	return examples
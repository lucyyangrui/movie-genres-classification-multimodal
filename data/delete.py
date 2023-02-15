import os

times = 0

nums = os.listdir('./pic-videos')

for num in nums:
	if not '.zip' in num:
		new_path = os.path.join('./pic-videos', num, 'data')
		files = os.listdir(new_path)
		for name in files:
			if '.mp4' in name or 'stills' in name or 'wav' in name:
				os.remove(os.path.join(new_path, name))
				# print('remove: ', os.path.join(new_path, name))
		times += 1
		if times % 100 == 0:
			print('move: %d / %d' % (times, len(nums)))


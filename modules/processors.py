from threading import Thread, Event
from threading import Lock as Thread_lock
from multiprocessing import Semaphore, Process
import random
import string
import time
import torch
from .conveyors import Line, Switcher, Router
from .shared_attr import ShmValue

from multiprocessing import Queue


class Module:
	def __init__(self, input_line: Line, output_line: Line = None,
	             control_line: Line = None, report_line: Line = None,
	             run_num_thread=1, receive_num_thread=1, send_num_thread=1,
	             receive_buffer_size=8, send_buffer_size=8,
	             name: str = None):

		if name is not None:
			self.__name = name
		else:
			self.__name = '(unnamed)' + type(self).__name__

		assert type(self).run != Module.run, \
			f'you haven\'t rewrite "run" for Module {self.__name}'
		assert type(self).run.__code__.co_argcount == 2, \
			f'{self.__name}.run has abnormal number of parameters'
		# assert isinstance(input_line, Line), \
		# 	f'Module {self.__name} must have one input_line'

		self.input_line = input_line
		self.output_line = output_line
		self.control_line = control_line
		self.report_line = report_line

		self.__input_buffer = []
		self.__input_buffer_empty_semaphore = Semaphore(receive_buffer_size)
		self.__input_buffer_ready_semaphore = Semaphore(0)
		self.__input_buffer_lock = Thread_lock()

		self.__output_buffer = []
		self.__output_buffer_empty_semaphore = Semaphore(send_buffer_size)
		self.__output_buffer_ready_semaphore = Semaphore(0)
		self.__output_buffer_lock = Thread_lock()

		self.__run_num_thread = run_num_thread
		self.__receive_num_thread = receive_num_thread
		self.__send_num_thread = send_num_thread

		self.__num_thread =\
			self.__receive_num_thread \
			+ self.__run_num_thread \
			+ (self.__send_num_thread if self.output_line is not None else 0) \
			+ (1 if self.control_line is not None else 0)


		self.__started_thread_semaphore = Semaphore(0)
		self._terminate_line = Queue()
		self._terminate_report_line = Queue()
		self._terminate_flag =False
		self.__terminated_thread_semaphore = Semaphore(0)

		self.__pid = None

		self.cluster_mode = False
		self.cluster_batch_data_flag = None

		self.status_line = None

	def launch(self):
		p = Process(target=self.__launch_p)
		p.start()
		self.__pid = p.pid
		for i in range(self.__num_thread):
			self.__started_thread_semaphore.acquire()

	def __launch_p(self):
		self.awake()
		for i in range(self.__receive_num_thread):
			Thread(target=self.__receive_data_t).start()

		if self.output_line is not None:
			for i in range(self.__send_num_thread):
				Thread(target=self.__send_data_t).start()

		for i in range(self.__run_num_thread):
			Thread(target=self.__run_t).start()

		if self.control_line is not None:
			Thread(target=self.__control_t).start()


		if self.status_line is not None:
			num_threads = {'receive_num_thread': self.__receive_num_thread,
			               'run_num_thread': self.__run_num_thread,
			               'send_num_thread': self.__send_num_thread if self.output_line is not None else 0,
			               }

			device_count = torch.cuda.device_count()
			gpu_memory_used = {}
			for i in range(device_count):
				memory = torch.cuda.memory_allocated(device=i) / 1024 / 8
				if memory != 0:
					gpu_memory_used['cuda:%d' % i] = memory

			status_package = {'name': self.__name,
			                  'type': 'Module',
			                  'status': 'launched',
			                  'gpu_memory_used': gpu_memory_used,
			                  'num_threads': num_threads
			                  }
			self.status_line.put(status_package)

		self._terminate_line.get()
		print(f'{self.__name} terminating')
		self.__terminate()


	def __terminate(self):
		self._terminate_flag = True

		for line in [self.input_line, self.output_line, self.control_line]:
			if line is not None:
				line.release()

		for i in range(self.__num_thread * 256):
			self.__input_buffer_empty_semaphore.release()
			self.__input_buffer_ready_semaphore.release()
			self.__output_buffer_empty_semaphore.release()
			self.__output_buffer_ready_semaphore.release()


		for i in range(self.__num_thread):
			self.__terminated_thread_semaphore.acquire()


		self.on_terminate()

		self._terminate_report_line.put({'terminated': True})

	# def __receive_terminate_t(self):
	# 	self._terminate_line.get()
	# 	print(f'{self.__name} terminating')
	# 	self.__terminate()

	def terminate(self):
		self._terminate_line.put({})
		self._terminate_report_line.get()

	def __receive_data_t(self):
		self.__started_thread_semaphore.release()
		while not self._terminate_flag:
			data = self.input_line.get()
			if data is None:
				break

			self.__input_buffer_empty_semaphore.acquire()
			if self._terminate_flag:
				break

			self.__input_buffer_lock.acquire()
			self.__input_buffer.append(data)
			self.__input_buffer_lock.release()

			self.__input_buffer_ready_semaphore.release()

		self.__terminated_thread_semaphore.release()

	def __send_data_t(self):
		self.__started_thread_semaphore.release()

		while not self._terminate_flag:
			ra = self.__output_buffer_ready_semaphore.acquire(timeout=0.5)
			if self._terminate_flag:
				break
			elif not ra:
				continue

			ra = self.__output_buffer_lock.acquire(timeout=0.5)
			if self._terminate_flag:
				break
			elif not ra:
				continue

			data = self.__output_buffer[0]
			del self.__output_buffer[0]
			self.__output_buffer_lock.release()

			self.__output_buffer_empty_semaphore.release()
			self.output_line.put(data)
		self.__terminated_thread_semaphore.release()

	def __control_t(self):
		self.__started_thread_semaphore.release()

		while not self._terminate_flag:
			ctl_package = self.control_line.get()
			if self._terminate_flag:
				break
			imperative(self, ctl_package)

		self.__terminated_thread_semaphore.release()

	def __run_t(self):
		self.__started_thread_semaphore.release()
		while not self._terminate_flag:
			input_data = self.__get_data()
			if self._terminate_flag:
				break

			if self.status_line is not None:
				if not self.cluster_mode:
					status_package = {'name': self.__name,
					                  'type': 'Module',
					                  'status': 'run_begin'}
					self.status_line.put(status_package)

			run_time_0 = time.time()

			if self.cluster_mode:
				input_batch_data = input_data[self.cluster_batch_data_flag]
				del input_data[self.cluster_batch_data_flag]
				output_batch_data = self.run(input_batch_data)
				run_time_1 = time.time()

				output_data = input_data
				output_data.update({self.cluster_batch_data_flag: output_batch_data})
				self.__put_data(output_data)
			else:
				output_data = self.run(input_data)
				run_time_1 = time.time()
				self.put_data(output_data)
			if self.status_line is not None:
				if not self.cluster_mode:
					status_package = {'name': self.__name,
					                  'type': 'Module',
					                  'status': 'run_end',
					                  'run_time_cost': run_time_1 - run_time_0}
					self.status_line.put(status_package)
		self.__terminated_thread_semaphore.release()

	def __get_data(self):
		self.__input_buffer_ready_semaphore.acquire()
		self.__input_buffer_lock.acquire()
		if any(self.__input_buffer):
			data = self.__input_buffer[0]
			del self.__input_buffer[0]
			self.__input_buffer_lock.release()
		else:
			self.__input_buffer_lock.release()
			return None

		self.__input_buffer_empty_semaphore.release()
		return data

	def __put_data(self, data_package):
		if data_package is None:
			return
		assert isinstance(data_package, dict), '模块间传输的数据包必须为dict格式\n'
		# assert self.output_line is not None, '没有output_line的Module不能使用put_data接口\n'
		self.__output_buffer_empty_semaphore.acquire()

		self.__output_buffer_lock.acquire()
		self.__output_buffer.append(data_package)
		self.__output_buffer_lock.release()

		self.__output_buffer_ready_semaphore.release()

	def put_data(self, data_package):
		assert not self.cluster_mode, '在Cluster中的创建Module不能使用put_data接口\n'
		self.__put_data(data_package)


	def report(self, report_package):
		assert self.report_line is not None, \
			f"failed to report {report_package}, {self.__name} does not have report_line"
		assert isinstance(report_package, dict), \
			f'failed to report {report_package}, report_package must be dict'
		assert ('target' in report_package), \
			f'failed to report {report_package}, missing target'
		self.report_line.put(report_package)

	def awake(self):
		pass

	def run(self, input_package: dict) -> (dict, None):
		pass

	def on_terminate(self):
		pass


class Cluster:
	def __init__(self, input_line: Line, output_line: Line,
	             batch_data_flag: str,
	             receive_num_thread: int = 1, send_num_thread: int = 1,
	             name=None):

		if name is not None:
			self.__name = name
		else:
			self.__name = '(unnamed)' + type(self).__name__

		assert type(self).build_modules != Cluster.build_modules, \
			f'you haven\'t rewrite "build_modules" for Cluster {self.__name}'

		self.input_line = input_line
		self.output_line = output_line

		self.modules = []
		self.modules_input_line = None
		self.modules_output_line = None

		self.__batch_data_flag = batch_data_flag
		self.__batch_registry = {}
		self.__batch_registry_lock = Thread_lock()

		self.__receive_num_thread = receive_num_thread
		self.__send_num_thread = send_num_thread

		self.__num_thread = receive_num_thread + send_num_thread

		self.__started_thread_semaphore = Semaphore(0)
		self.__terminated_thread_semaphore = Semaphore(0)

		self._terminate_flag = ShmValue(bool)
		self._terminate_flag.set(False)

		self.__pid = None

		self.__num_modules = 0
		self.status_line = None
		self.__modules_status_line = None

	def __receive_t(self):
		self.__started_thread_semaphore.release()

		while not self._terminate_flag.value():
			input_package = self.input_line.get()
			if input_package is None:
				continue

			assert self.__batch_data_flag in input_package, \
				f'{self.__name} received package without batch_data_flag: {self.__batch_data_flag}'
			assert isinstance(input_package[self.__batch_data_flag], list), \
				f'value of {self.__batch_data_flag} must be list'

			if not any(input_package[self.__batch_data_flag]):
				self.output_line.put(input_package)
				continue

			batch_packages = input_package[self.__batch_data_flag]
			input_package[self.__batch_data_flag] = None

			batch_size = len(batch_packages)
			if batch_size == 0:
				self.output_line.put(input_package)
			else:
				batch_token = ''.join(random.sample(string.ascii_letters + string.digits, 8))
				while batch_token in self.__batch_registry:
					batch_token = ''.join(random.sample(string.ascii_letters + string.digits, 8))

				assert batch_token not in self.__batch_registry, \
					f'{self.__name} received repeated batch_token: {batch_token}'

				self.__batch_registry[batch_token] = {}
				self.__batch_registry[batch_token]['batch_size'] = batch_size
				self.__batch_registry[batch_token]['finished_cnt'] = 0
				self.__batch_registry[batch_token]['batch_begin_time'] = time.time()
				self.__batch_registry[batch_token]['single_data_cost_time_cnt'] = 0

				self.__batch_registry[batch_token]['result'] = {}
				self.__batch_registry[batch_token]['result'].update(input_package)
				self.__batch_registry[batch_token]['result'][self.__batch_data_flag] = [None] * batch_size

				for i in range(batch_size):
					package = {'__batch_token': batch_token,
					           '__data_index': i,
					           '__send_time': time.time(),
					           self.__batch_data_flag: batch_packages[i]}
					self.modules_input_line.put(package)

				if self.status_line is not None:
					status = {'name': self.__name,
					          'type': 'Cluster',
					          'status': 'run_begin'}
					self.status_line.put(status)

		self.__terminated_thread_semaphore.release()

	def __send_t(self):
		self.__started_thread_semaphore.release()

		while not self._terminate_flag.value():
			modules_output_package = self.modules_output_line.get()
			if modules_output_package is None:
				continue
			assert self.__batch_data_flag in modules_output_package
			batch_token = modules_output_package['__batch_token']
			data_index = modules_output_package['__data_index']
			batch_data = modules_output_package[self.__batch_data_flag]
			assert isinstance(batch_data, dict)

			output_package = None

			self.__batch_registry_lock.acquire()

			self.__batch_registry[batch_token]['result'][self.__batch_data_flag][data_index] = batch_data
			self.__batch_registry[batch_token]['finished_cnt'] += 1
			self.__batch_registry[batch_token]['single_data_cost_time_cnt'] += \
				time.time() - modules_output_package['__send_time']
			if self.__batch_registry[batch_token]['finished_cnt'] == self.__batch_registry[batch_token]['batch_size']:
				output_package = self.__batch_registry[batch_token]['result']
				del self.__batch_registry[batch_token]['result']

			self.__batch_registry_lock.release()

			if output_package is not None:
				batch_run_time_cost = time.time() - self.__batch_registry[batch_token]['batch_begin_time']
				single_data_run_time_cost = \
					self.__batch_registry[batch_token]['single_data_cost_time_cnt'] \
					/ self.__batch_registry[batch_token]['batch_size']
				self.output_line.put(output_package)

				if self.status_line is not None:
					status = {'name': self.__name,
					          'type': 'Cluster',
					          'status': 'run_end',
					          'batch_run_time_cost': batch_run_time_cost,
					          'single_data_run_time_cost': single_data_run_time_cost}
					self.status_line.put(status)

		self.__terminated_thread_semaphore.release()

	def __launch_p(self):
		for i in range(self.__receive_num_thread):
			Thread(target=self.__receive_t).start()
		for i in range(self.__send_num_thread):
			Thread(target=self.__send_t).start()

	def launch(self):
		assert not self._terminate_flag.value(), \
			f'failed to launch {self.__name}, can\'t launch a terminated Cluster'
		self.build_modules()

		assert any(self.modules), \
			f'failed to build_modules for {self.__name}, there is no module built for Cluster'
		for module in self.modules:
			assert isinstance(module, Module), \
				f'failed to build_modules for {self.__name}, {type(module).__name__} is not Module'
			assert isinstance(module, type(self.modules[0])), \
				f'failed to build_modules for {self.__name}, all module in one Cluster must have same type'
			assert isinstance(module.input_line, Line), \
				f'failed to build_modules for {self.__name}, modules in Cluster must have input_line'
			assert isinstance(module.output_line, Line), \
				f'failed to build_modules for {self.__name}, modules in Cluster must have output_line'
			assert module.input_line == self.modules_input_line, \
				'failed to build_modules for {self.__name}, modules in Cluster must share one single input_line'
			assert module.output_line == self.modules_output_line, \
				'failed to build_modules for {self.__name}, modules in Cluster must share one single output_line'
			self.__num_modules += 1

		if self.status_line is not None:
			self.__modules_status_line = Line()

		for module in self.modules:
			module.cluster_mode = True
			module.cluster_batch_data_flag = self.__batch_data_flag
			if self.__modules_status_line is not None:
				module.status_line = self.__modules_status_line
			module.launch()

		p = Process(target=self.__launch_p)
		p.start()
		self.__pid = p.pid
		for i in range(self.__num_thread):
			self.__started_thread_semaphore.acquire()

		if self.status_line is not None:
			status_package = {'name': self.__name,
			                  'type': 'Cluster',
			                  'status': 'launched',
			                  'num_threads': {'receive_num_thread': self.__receive_num_thread,
			                                  'send_num_thread': self.__send_num_thread
			                                  },
			                  'num_modules': self.__num_modules,
			                  'type_modules': type(self.modules[0]).__name__,
			                  'modules_status': []
			                  }
			for i in range(self.__num_modules):
				status = self.__modules_status_line.get()
				status_package['modules_status'].append(status)
			self.status_line.put(status_package)

	def terminate(self):
		assert self.__pid is not None, f'try to terminate an un-launched Cluster {self.__name}'
		self._terminate_flag.set(True)
		for module in self.modules:
			module.terminate()

		self.input_line.release()
		self.output_line.release()

		for i in range(self.__num_thread):
			self.__terminated_thread_semaphore.acquire()

		self._terminate_flag.release()

	def build_modules(self):
		pass


class Dispatcher:
	def __init__(self, task_token_flag: str):

		self.__name = type(self).__name__

		assert type(self).build_processors != Dispatcher.build_processors, \
			f'you haven\'t rewrite "build_processors" for Dispatcher {self.__name}'

		self.__task_token_flag = task_token_flag
		self.__statistics_length = 500
		# self.__reboot()
		self.__launched = False

	def __reboot(self):
		self.processors = []
		self.conveyors = []
		self.head_line = None
		self.report_line = None

		self.__task_registry = {}

		self._terminate_flag = False
		self.__started_thread_semaphore = Semaphore(0)
		self.__terminated_thread_semaphore = Semaphore(0)

		self.__status_line = Line()
		self.__launched_semaphore = Semaphore(0)

		# self.__statistics = {'Dispatcher': {'type': 'Dispatcher',
		#                                     'task_time_costs': [],
		#                                     'received_task_cnt': 0}}
		self.__statistics = {
			'Dispatcher': {
				'type': 'Dispatcher',
				'task_time_costs': [],
				'received_task_cnt': 0,
				'run_time_costs': [] },
			'text_recognition_module': {
				'type': 'Text Recognition',
				'task_time_costs': [],
				'received_package_cnt': 0,
				'run_time_costs': [] },
			'text_detection_module': {
				'type': 'Text Recognition',
				'task_time_costs': [],
				'received_package_cnt': 0,
				'run_time_costs': [] },
			'text_refine_module': {
				'received_package_cnt': 0,
				'run_time_costs': []}
		}
		

	def __receive_reports_t(self):
		self.__started_thread_semaphore.release()

		while (not self._terminate_flag) or (not self.report_line.empty()):
			report_package = self.report_line.get()
			if self._terminate_flag:
				break
			if report_package is None:
				continue
			imperative(self, report_package)
		self.__terminated_thread_semaphore.release()

	def __receive_status_t(self):
		self.__started_thread_semaphore.release()
		self.__launched_semaphore.acquire()
		while (not self._terminate_flag) or (not self.__status_line.empty()):
			status = self.__status_line.get()
			if status is None:
				continue
			if status['status'] == 'run_begin':
				self.__statistics[status['name']]['received_package_cnt'] += 1

			elif status['status'] == 'run_end':

				if status['type'] == 'Module':
					run_time_costs = self.__statistics[status['name']]['run_time_costs']
					run_time_costs.append(status['run_time_cost'])
					if len(run_time_costs) >= self.__statistics_length:
						del run_time_costs[0]
				elif status['type'] == 'Cluster':
					batch_run_time_costs = self.__statistics[status['name']]['batch_run_time_costs']
					batch_run_time_costs.append(status['batch_run_time_cost'])
					if len(batch_run_time_costs) >= self.__statistics_length:
						del batch_run_time_costs[0]
		self.__terminated_thread_semaphore.release()

	def statistics(self):
		return self.__statistics

	def statistics_log(self):
		log = '*' * 32 + '\n'
		for name, statistic in self.__statistics.items():
			type_ = statistic['type']
			if type_ == 'Module':
				received_package_cnt = statistic["received_package_cnt"]
				log += f'Module 【{name}】\n' \
				       + f'received_package_cnt: {received_package_cnt}\n' \
				       + f'run_num_threads: {statistic["run_num_threads"]}\n'
				if received_package_cnt > 0:
					run_time_costs = statistic['run_time_costs']
					mean_run_time_cost = sum(run_time_costs) / len(run_time_costs)
					log += f'mean_run_time_cost: {mean_run_time_cost:.8f}s\n'
				log += '*' * 32 + '\n'
			elif type_ == 'Cluster':
				received_package_cnt = statistic["received_package_cnt"]
				log += f'Cluster 【{name}】\n' \
				       + f'received_package_cnt: {received_package_cnt}\n' \
				       + f'num_modules: {statistic["num_modules"]}'
				if received_package_cnt > 0:
					batch_run_time_costs = statistic['batch_run_time_costs']
					mean_batch_run_time_costs = sum(batch_run_time_costs) / len(batch_run_time_costs)
					log += f'mean_batch_run_time_costs: {mean_batch_run_time_costs:.8f}s\n'
				log += '*' * 32 + '\n'
		task_time_costs = self.__statistics['Dispatcher']['task_time_costs']
		mean_task_time_cost = sum(task_time_costs) / len(task_time_costs)
		received_task_cnt = self.__statistics['Dispatcher']['received_task_cnt']
		log += '【Dispatcher】\n' \
		       + f'received_task_cnt: {received_task_cnt}\n' \
		       + f'mean_task_time_cost: {mean_task_time_cost:.8f}s\n'

		return log

	def finish(self, result: dict):
		assert self.__task_token_flag in result, \
			f'missing task_toke_flag {self.__task_token_flag} in result {result}'
		task_token = result[self.__task_token_flag]
		self.__task_registry[task_token]['result'] = result
		self.__task_registry[task_token]['semaphore'].release()

	def launch_t(self, processor_tmp):
		processor_tmp.launch()

	def launch(self):
		assert not self.__launched, 'Dispatcher has already launched'
		self.__launched = True
		self.__reboot()
		self.build_processors()

		assert self.head_line is not None, \
			f'failed to launch {self.__name}, {self.__name}.head_line is not assigned in {self.__name}.build_processors'
		assert self.report_line is not None, \
			f'failed to launch {self.__name}, {self.__name}.report_line is not assigned in {self.__name}.build_processors'

		Thread(target=self.__receive_reports_t).start()
		Thread(target=self.__receive_status_t).start()
		for i in range(2):
			self.__started_thread_semaphore.acquire()

		for conveyor in self.conveyors:
			assert isinstance(conveyor, (Switcher, Router)), \
				f'failed to launch {self.__name}, {type(conveyor).__name__} is not Switcher or Router'
			conveyor.status_line = self.__status_line
			conveyor.launch()

		for processor in self.processors:
			assert isinstance(processor, (Module, Cluster)),\
				f'failed to launch {self.__name}, {type(processor).__name__} is not Module or Cluster'
			processor.status_line = self.__status_line
			Thread(target=self.launch_t, args=(processor,)).start()
			time.sleep(0.2)
		for i in range(len(self.processors)):
			status = self.__status_line.get()
			if status['status'] == 'launched':
				if status['type'] == 'Module':
					log = \
						'*' * 32 + '\n' \
						+ 'launched Module: ' + log_module_launch_status(status) \
						+ '*' * 32 + '\n'

					self.__statistics[status['name']] = {'type': 'Module',
					                                     'received_package_cnt': 0,
					                                     'run_num_threads': status['num_threads']['run_num_thread'],
					                                     'run_time_costs': []}
				else:  # Cluster
					log = \
						'*' * 32 + '\n' \
						+ f'launched Cluster 【{status["name"]}】\n' \
						+ 'num_threads:' + ', '.join([f'{k}: {v}' for k, v in status['num_threads'].items()]) + '\n' \
						+ f'modules: 【{status["type_modules"]}】 * {status["num_modules"]}\n'
					for module_status in status['modules_status']:
						log += log_module_launch_status(module_status, indent=1)
					log += '*' * 32 + '\n'
					self.__statistics[status['name']] = {'type': 'Cluster',
					                                     'received_package_cnt': 0,
					                                     'num_modules': len(status['modules_status']),
					                                     'batch_run_time_costs': []}

				print(log)

		self.__launched_semaphore.release()

	def terminate(self):
		for c in self.conveyors:
			c.terminate()
		for p in self.processors:
			p.terminate()
		self._terminate_flag = True
		self.head_line.release()
		for key in self.__task_registry.keys():
			self.__task_registry[key]['result'] = None
			self.__task_registry[key]['semaphore'].release()
		# while not self.report_line.empty():
		# 	time.sleep(0.01)
		self.report_line.release()
		# while not self.__status_line.empty():
		# 	time.sleep(0.01)
		self.__status_line.release()
		for i in range(2):
			self.__terminated_thread_semaphore.acquire()
		self.report_line.release()
		self.__status_line.release()
		self.__launched = False

	def dispatch(self, input_data_package):
     # 下面是输入
		# input_data = {
        #         'timestamp': time.time(),
        #         'raw_img': img,
        #         'bbox_list': bbox_list[idx],
        #         'type_list': construction_type_list[idx]
        #     }
		assert self.__task_token_flag in input_data_package
		if self._terminate_flag:
			return None

		task_token = input_data_package[self.__task_token_flag]
		assert task_token not in self.__task_registry, \
			f'repeated task_token {task_token}'
		self.__task_registry[task_token] = {}
		self.__task_registry[task_token]['semaphore'] = Semaphore(0)
		self.__task_registry[task_token]['begin_time'] = time.time()
		self.head_line.put(input_data_package)
  		# 这一行将输入数据包 input_data_package 放入了 head_line 队列。通常，这样的队列
 	 	# 被用于进程间或线程间的通信。在这个队列中的任务将被另外的工作器（worker）线程或进程取出并执行实际的处理工作。
    	# 完成任务的工作器通常会释放 Semaphore
		self.__statistics['Dispatcher']['received_task_cnt'] += 1

		self.__task_registry[task_token]['semaphore'].acquire()
		# 这行代码是等待任务完成的地方。Semaphore 的 acquire 方法会阻塞，直到任务被工作器处理并且调用了 Semaphore 的 release 方法。

		task_time_cost = time.time() - self.__task_registry[task_token]['begin_time']
		task_time_costs = self.__statistics['Dispatcher']['task_time_costs']
		task_time_costs.append(task_time_cost)
		if len(task_time_costs) >= self.__statistics_length:
			del task_time_costs[0]

		res = self.__task_registry[task_token]['result']
		del self.__task_registry[task_token]
		return res

	def build_processors(self):
		pass


def log_module_launch_status(status, indent=0):
	log = \
		'\t' * indent + f'【{status["name"]}】\n' \
		+ ('\t' * indent
		   + ('gpu_memory_used: ' + ', '.join([f'"{k}"' for k in status['gpu_memory_used'].keys()]) + '\n')
		   if any(status['gpu_memory_used'])
		   else '') \
		+ '\t' * indent + 'num_threads: ' + ', '.join([f'{k}: {v}' for k, v in status['num_threads'].items()]) + '\n'
	return log


def imperative(instance, ctl_package):
	assert 'target' in ctl_package
	target = ctl_package['target']
	args = ctl_package.get('args', dict)

	assert isinstance(target, str), \
		f'failed to imperative {type(instance).__name__}.{target},' \
		f' target must be str. ' \
		f'imperative_package: {str(ctl_package)}'
	assert isinstance(args, dict), \
		f'failed to imperative {type(instance).__name__}.{target}, ' \
		f'args must dict. ' \
		f'imperative_package: {str(ctl_package)}'
	assert (target in dir(instance)) and hasattr(getattr(instance, target), '__call__'), \
		f'failed to imperative {type(instance).__name__}.{target}, ' \
		f'{type(instance).__name__} does not have function {target} ' \
		f'imperative_package: {str(ctl_package)}'

	getattr(instance, target)(**args)


class DemoComponent:
	def __init__(self, run_time=0.5, use_gpu=False, device_id=0):
		self.use_gpu = use_gpu
		if use_gpu:
			self.__tensor = torch.zeros((1,), dtype=torch.int8)
			self.__tensor = self.__tensor.to(torch.device('cuda:%d' % device_id))

		self.__run_time = run_time

	def inference(self, v):
		if self.use_gpu:
			self.__tensor[0] += 1

		time.sleep(self.__run_time)
		return v+1

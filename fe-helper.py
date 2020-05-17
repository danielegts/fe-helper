import argparse
import csv
import logging
import multiprocessing
import os
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from argparse import RawTextHelpFormatter
from collections import Counter
from multiprocessing.pool import Pool

import numpy
import pandas as pd
import xmlschema

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
	'%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
# logger.parent = None
logger.addHandler(handler)


# logger.propagate = False


class XmlFeaturesAnalyzer:
	def __init__(self, schema, root_key, encoding="ISO-8859-1"):
		'''
		Initialize feature analyzer

		:param schema: path of the xsd schema file
		:param root_key: root key of schema
		:param encoding: encoding of xml file
		'''
		self.encoding = encoding

		self.xs = xmlschema.XMLSchema(schema)

		el = self.xs.elements[root_key]

		self.unique_elements, self.repeating_elements, self.types = self.analyze_elements(el, "", [], [], {})

		# delete empty keys
		self.unique_elements = [key for key in self.unique_elements if key != '']
		self.repeating_elements = [key for key in self.repeating_elements if key != '']

		logger.info("Analyzer initialized")

	def analyze_elements(self, elem, tags_path, unique_elements, repeating_elements, types):
		'''
		Recursively explore XML schema hierarchy to obtain information on each element

		:param elem: initial element to start exploration
		:param tags_path: parents attributes concatenation of specified element
		:param unique_elements: list of elements appearing only once in a XML with this schema
		:param repeating_elements: list of elements appearing more than one time in a XML with this schema
		:param types: type of each element (only for simple type elements, e.g. string, int etc.)
		:return: unique_elements, repeated_elements, types
		'''
		children_count = sum(1 for el in elem.iterchildren())

		# for repeating elements max_occurs return None
		if elem.max_occurs == None:
			repeating_elements += [tags_path]
		else:
			unique_elements += [tags_path]

		if children_count == 0:
			# to get the type we need to use two different attributes
			type_id = elem.type.id
			base_type = elem.type.base_type

			if base_type is not None:
				base_type_id = base_type.id

				if base_type_id == "string" or base_type_id == "normalizedString":
					types[tags_path] = "string"
				if base_type_id == "decimal":
					types[tags_path] = "decimal"
				if base_type_id == "integer":
					types[tags_path] = "integer"

			if type_id == "date":
				types[tags_path] = "date"

			return unique_elements, repeating_elements, types

		else:
			for child in elem.iterchildren():
				new_path = tags_path + "/" + child.local_name

				unique_elements, repeating_elements, types = self.analyze_elements(child, new_path, unique_elements,
																				   repeating_elements, types)

			return unique_elements, repeating_elements, types

	def analyze(self, xml_path):
		'''
		Scan a single XML file to get information on element values

		:param xml_path: path of xml file
		:return: values distribution and number of occurrences of each element in the file
		'''
		values = {}
		occurrences = {}

		try:
			parser = ET.XMLParser(encoding=self.encoding)
			tree = ET.parse(xml_path, parser=parser)

			root = tree.getroot()

			# if we deal with unique elements we just take the value or track if not present
			for key in self.unique_elements + self.repeating_elements:
				repeated = list(filter(key.startswith, self.repeating_elements)) != []

				if repeated:
					# if we deal with sequence elements we count how many times they appear
					els = root.findall("." + key)
					if els is not None:
						values[key] = len(els)
					else:
						values[key] = 0
				else:
					el = root.find("." + key)
					if el is not None:
						values[key] = el.text
					else:
						values[key] = None

				occurrences[key] = len(root.findall("." + key))


		except Exception as e:
			logger.error(xml_path + "\n" + str(e))
			pass

		return values, occurrences


def explore(path, analyzer, debug=False, n_pools=multiprocessing.cpu_count()):
	'''
	Process each XML file in the specified path

	:param path: target path containing XML files to be processed
	:param analyzer: XML analyzer
	:param debug: test on small selection of XML files
	:return: values distribution of each element (dictionary), maximum number of occurrences of each element in a file,
			 number of processed xml files
	'''
	logger.info("Starting analyzing xml files using %d cpu cores", n_pools)
	p = Pool(n_pools)

	xml_files = [os.path.abspath(os.path.join(path, p)) for p in os.listdir(path) if os.path.splitext(p)[1] == ".XML"]

	if debug:
		xml_files = xml_files[:100]

	logger.info('Processing %d files', len(xml_files))

	comp_res = p.map(analyzer.analyze, xml_files, chunksize=1000)
	result = [x[0] for x in comp_res if x[0] is not None]
	occurrences = [x[1] for x in comp_res if x[0] is not None]

	values = {
		k: Counter([d.get(k) for d in result]) for k in set().union(*result)
	}

	max_occurrences = {
		k: max([d.get(k) for d in occurrences]) for k in set().union(*result)
	}

	return values, max_occurrences, len(xml_files)


def get_elements_report(types, values, max_occurrences, n_files, repeating_elements):
	'''
	Get a pandas representation of elements (only simple type ones, e.g. string, decimal, date etc.)

	:param types: type of each element
	:param values: elements values distributions
	:param max_occurrences: maximum number of occurrences of each element
	:param n_files: number of processed files
	:param repeating_elements: list of elements appearing more than one time in a XML with this schema
	:return: info for each element in a pandas dataframe
	'''

	dtypes = numpy.dtype([
		('key', str),
		('type', str),
		('none_%', float),
		('multiple_occurrence', bool),
		('num_occrs', int)
	])

	data = numpy.empty(0, dtype=dtypes)
	report = pd.DataFrame(data)

	for k in types.keys():
		maybe_multiple = list(filter(k.startswith, repeating_elements)) != []

		if maybe_multiple:
			none_percentage = values[k][0] / n_files
		else:
			none_percentage = values[k][None] / n_files

		k_type = types[k]

		report.loc[len(report)] = [k, k_type, none_percentage, maybe_multiple, max_occurrences[k]]

	return report


def main():
	if args.verbose:
		level = logging.DEBUG
	else:
		level = logging.ERROR

	logger.setLevel(level)

	xml_fa = XmlFeaturesAnalyzer(args.schema_path, args.root_element)

	if args.poolsize:
		values, max_occurrences, n_files = explore(args.xmls_path, xml_fa, debug=args.debug, n_pools=args.poolsize)
	else:
		values, max_occurrences, n_files = explore(args.xmls_path, xml_fa, debug=args.debug)

	report = get_elements_report(xml_fa.types, values, max_occurrences, n_files, xml_fa.repeating_elements)
	report.to_csv('report.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)

	return report, values


if __name__ == '__main__':
	try:
		start_time = time.time()

		parser = argparse.ArgumentParser(
			description="XML features explorer",
			formatter_class=RawTextHelpFormatter)

		parser.add_argument("-v", "--verbose", action="store_true")
		parser.add_argument("-d", "--debug", action="store_true")
		parser.add_argument('-p', '--poolsize', type=int, dest="poolsize", default=4,
							help='Thread pool size (do not set more than the amount of cpu cores)')
		parser.add_argument("schema_path", type=str, help="Schema files path")
		parser.add_argument("xmls_path", type=str, help="Target XML files path")
		parser.add_argument("root_element", type=str, help="Element to start exploration")
		args = parser.parse_args()

		if not len(sys.argv) > 2:
			parser.error('missing argument')

		main()
		if args.verbose:
			time_taken = time.time() - start_time
			print('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(time_taken))))

		sys.exit(0)
	except KeyboardInterrupt as e:  # Ctrl-C
		raise e
	except SystemExit as e:  # sys.exit()
		raise e
	except Exception as e:
		print('ERROR, UNEXPECTED EXCEPTION')
		print(str(e))
		traceback.print_exc()
		os._exit(1)

if __name__ == '__main__':
	main()

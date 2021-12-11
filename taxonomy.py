import pandas as pd
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types


person_sex = {'U': 0, 'M': 1, 'F': 2}

bodily_injury = {'Unknown': -1, 'Does Not Apply': 0, 'Entire Body': 1, 'Head': 2, 'Neck': 3, 'Chest': 4, 'Back': 5,
					'Abdomen - Pelvis': 6, 'Arm': 7, 'Leg': 8, 'Face': 9, 'Eye': 10, 'Shoulder - Upper Arm': 11,
					'Elbow-Lower-Arm-Hand': 12, 'Hip-Upper Leg': 13, 'Knee-Lower Leg Foot': 14}

safety_equipment = {'Unknown': -1, 'None': 0, 'Air Bag Deployed': 1, 'Child Restraint': 2, 'Harness': 3, 'Helmet': 4, 'Lap Belt': 5, 
					'Pads': 6, 'Stoppers': 7, 'Other': 8}


def get_encoding(string, encoder):
	encoding = -2

	if isinstance(string, str) is False:
		return encoding

	for key, value in encoder.items():
		if key in string:
			if key == 'Arm' or key == 'Leg':
				continue
			elif encoding != -2 and encoder == safety_equipment:
				encoding = int(str(encoding) + str(value))
			elif encoding != -2 and encoder == bodily_injury:
				break
			else:
				encoding = value

	if encoding == -2:
		raise ValueError("Encoding didn't work.")

	return encoding


def create_encoding_columns(data):
	data['SAFETY_EQUIPMENT_ENCODING'] = data.apply(lambda row: get_encoding(row['SAFETY_EQUIPMENT'], safety_equipment), axis=1)
	data['BODILY_INJURY_ENCODING'] = data.apply(lambda row: get_encoding(row['BODILY_INJURY'], bodily_injury), axis=1)
	data['PERSON_SEX_ENCODING'] = data.apply(lambda row: get_encoding(row['PERSON_SEX'], person_sex), axis=1)


def print_unique_symbolic_values(data):
	variables = get_variable_types(data)['Symbolic']
	if [] == variables:
		raise ValueError('There are no symbolic variables.')

	if 'GbCity' in variables:
		variables.remove('GbCity')

	if 'date' in variables:
		variables.remove('date')

	if 'CRASH_DATE' in variables:
		variables.remove('CRASH_DATE')

	if 'CRASH_TIME' in variables:
		variables.remove('CRASH_TIME')

	if 'PERSON_ID' in variables:
		variables.remove('PERSON_ID')

	data = data.drop(columns=[col for col in data if col not in variables])

	for col in data:
		print('\n' + col)
		print(data[col].unique())
	print()


def main():
	register_matplotlib_converters()

	# 'Air Quality'
	#data = pd.read_csv('data/air_quality_tabular.csv', index_col='FID', parse_dates=True, infer_datetime_format=True)
	#print_unique_symbolic_values(data)


	# 'NYC Collisions'
	data = pd.read_csv('data/NYC_collisions_tabular.csv', index_col='COLLISION_ID', parse_dates=True, infer_datetime_format=True)
	#print_unique_symbolic_values(data)
	create_encoding_columns(data)
	print(data['SAFETY_EQUIPMENT_ENCODING'])
	print(data['BODILY_INJURY_ENCODING'])
	print(data['PERSON_SEX_ENCODING'])



if __name__ == '__main__':
	main()
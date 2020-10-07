import csv
import os

class splitter:

    def __init__(self, data_folder, delimiter):
        self.data_path = data_folder
        self.delimiter = delimiter

        if not os.path.exists(self.data_path + f"/CURATED"):
            os.makedirs(self.data_path + f"/CURATED")
        if not os.path.exists(self.data_path + f"/RAW"):
            os.makedirs(self.data_path + f"/RAW")
        if not os.path.exists(self.data_path + f"/OUTPUT"):
            os.makedirs(self.data_path + f"/OUTPUT")


    def file_splitter(self, file_name, column_to_split_on, output_folder):

        csv.field_size_limit(10000000)
        if not os.path.exists(self.data_path + f"/CURATED/{output_folder}"):
            os.makedirs(self.data_path + f"/CURATED/{output_folder}")

        with open(self.data_path + "/RAW/" + file_name, encoding = 'utf-8') as file :
            file_dict = csv.DictReader(file, delimiter = self.delimiter)

            already_opened_files = {}

            for row in file_dict:
                file_column = row[column_to_split_on]
                file_column = file_column.replace('\\', '.')
                file_column = file_column.replace('/', '.')

                if file_column not in already_opened_files:
                    out_file = open(self.data_path + f"/CURATED/{output_folder}/{file_column}.csv", 'w', encoding='utf-8')
                    dict_writer = csv.DictWriter(out_file, fieldnames=file_dict.fieldnames)
                    dict_writer.writeheader()
                    already_opened_files[file_column] = out_file, dict_writer

                already_opened_files[file_column][1].writerow(row)

            for output, _ in already_opened_files.values():
                output.close()

        print(f"Split fichier {file_name} terminé")

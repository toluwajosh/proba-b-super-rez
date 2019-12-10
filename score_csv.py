"""
Get baseline score from csv
"""
import csv


class BaseScore(object):
    def __init__(self, csv_file="./data/norm.csv"):
        with open("./data/norm.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=" ")
            scores_dict = {}
            for row in csv_reader:
                scores_dict[row[0]] = float(row[1])
        self.scores_dict = scores_dict

    def __getitem__(self, image_set_name):
        return self.scores_dict[image_set_name]

    def mean(self):
        total = 0
        for score in self.scores_dict.values():
            total+=score
        return total/len(self.scores_dict.values())


if __name__ == "__main__":
    base_scores = BaseScore()
    print(base_scores["imgset1449"])
    print("base score mean: ", base_scores.mean())
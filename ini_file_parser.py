import csv
import os


def file_parser(input_file):
    output_file = input_file.split(".")[0] + ".csv"
    prices = []
    stocks = 0
    pop = 0
    with open(input_file, "r") as file:

        for index, line in enumerate(file):
            if index == 0:
                stocks, pop = line.strip(" \n").replace('\t', ' ').split(None, 1)
            else:
                prices.extend(line.strip(" \n").replace('\t', ' ').split())

    with open(output_file, "w") as file:

        for stock in range(int(stocks)):
            for j in range(int(pop)+1):
                file.write(prices[j + (int(pop)+1)*stock].strip(" ") + "\t")
            file.write("\n")


if __name__ == "__main__":
    for file in os.listdir("stocks"):
        if file.endswith(".txt"):
            file_parser("stocks/" + file)

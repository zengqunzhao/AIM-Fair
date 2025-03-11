# '''
# Sampling from CelebA to construct a biased dataset.
# '''

import shutil
import os
import pandas as pd
import pdb
import csv
import argparse
from fractions import Fraction

parser = argparse.ArgumentParser()
parser.add_argument('--data-type', type=str, default="train", help="train, val, test")
parser.add_argument('--target-attribute', type=str, default="Smiling", help="Smiling, Blond_Hair, Black_Hair, Male, Young")
parser.add_argument('--sensitive-attribute', type=str, default="Male", help="Male, Young")
parser.add_argument('--bias-degree', type=str, default="1/9", help="1/9, 2/8, 3/7...")
parser.add_argument('--number-train-data', type=int, default=20000)
args = parser.parse_args()

img_path = "./img_align_celeba/"
attr_path = './list_attr_celeba.csv'


def main():

    attr_target_name = args.target_attribute
    attr_sensitive_name = args.sensitive_attribute
    bias_degree =  float(Fraction(args.bias_degree))
    num_train_total = args.number_train_data

    list_selected_img_name= []

    if args.data_type == "val":
        saved_csv_path = f"./annotations/CelebA_Val.csv"
        for i in range(162771, 182638):
            list_selected_img_name.append(str(i).zfill(6)+".jpg")
            
    elif args.data_type == "test":
        saved_csv_path = f"./annotations/CelebA_Test.csv"
        for i in range(182638, 202600):
            list_selected_img_name.append(str(i).zfill(6)+".jpg")
            
    elif args.data_type == "train":
        saved_csv_path = f"./annotations/CelebA_Train_{attr_target_name}_{attr_sensitive_name}_{num_train_total}_{bias_degree/(bias_degree+1)/2:.2f}.csv"
        df = pd.read_csv(attr_path)
        ratio_ys = [bias_degree/(bias_degree+1)/2, 1/(bias_degree+1)/2, 1/(bias_degree+1)/2, bias_degree/(bias_degree+1)/2]
        print(ratio_ys)
        num_train_ys_total = [round(ratio_ys[i]*num_train_total) for i in range(len(ratio_ys))]
        print(num_train_ys_total)
        count_train_ys = [0, 0, 0, 0]
        for i in range(162770):
            if(os.path.exists(img_path+str(i+1).zfill(6)+".jpg")):
                y = int((df.loc[i][attr_target_name]+1)/2)
                s = int((df.loc[i][attr_sensitive_name]+1)/2)
                ys = y*2+s
                if count_train_ys[ys] < num_train_ys_total[ys]:
                    list_selected_img_name.append(df.loc[i]['image_id'])
                    count_train_ys[ys] += 1
                if count_train_ys == num_train_ys_total:
                    print('count finished')
                    break
        if count_train_ys != num_train_ys_total:
            print(count_train_ys)
            print('error: training data is NOT enough')
            for i in range(162770):
                y = int((df.loc[i][attr_target_name]+1)/2)
                s = int((df.loc[i][attr_sensitive_name]+1)/2)
                ys = y*2+s
                if (count_train_ys[ys] < num_train_ys_total[ys]) & (df.loc[i]['image_id'] not in list_selected_img_name):
                    list_selected_img_name.append(df.loc[i]['image_id'])
                    count_train_ys[ys] += 1
                if count_train_ys == num_train_ys_total:
                    print('count finished')
                    break
            if count_train_ys != num_train_ys_total:
                print(count_train_ys)
                print('error: training data is NOT enough')

    print(saved_csv_path)

    selected_row = []
    with open(attr_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for index, line_ in enumerate(reader):
            if index == 0: # Add the head into the selected list
                selected_row.append(line_)
            if line_[0] in list_selected_img_name:
                selected_row.append(line_)
    with open(saved_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in selected_row:
            writer.writerow(row)


if __name__ == "__main__":
    main()

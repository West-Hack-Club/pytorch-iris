
import pandas as pd
import numpy as np
import re

csvpath = 'C:/PythonProjects/irispytorch/data/'

class DataProcess:
    
    def __init__(self, path, newfilepath) -> None:
        self.path = path
        self.newfilepath = newfilepath

    def getLastColumn(self):
        original = open(self.path, "r")
        processed_file = original.readlines()
        open(self.newfilepath, "w", encoding="utf-8")

        for i in range(len(processed_file)):
            str1 = processed_file[i].replace("Class_", "")
            str1 = re.sub(r'^.*,', '', str1)
            f = open(self.newfilepath, "a", encoding="utf-8")
            f.write(str1)
            f.close()

    def getFeatures(self):
        original = open(self.path, "r")
        processed_file = original.readlines()
        open(self.newfilepath, "w", encoding="utf-8")

        for i in range(len(processed_file)):
            str1 = processed_file[i]
            str1 = re.sub(r',Class_.*$', '', str1)
            str1 = re.sub(r',target.*$', '', str1)
            str1 = re.sub(r'^(.+?),', '', str1)
            f = open(self.newfilepath, "a", encoding="utf-8")
            f.write(str1)
            f.close()

class DataProcess2(DataProcess):
    def __init__(self, path, newfilepath) -> None:
        super().__init__(path, newfilepath)
        

    def fileSize(self):
        readFile = open(self.path, "r")
        fileLength = readFile.readlines()
        print(len(fileLength))



class TextLoading:
    
    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        loadedText = np.genfromtxt(self.filepath, delimiter=',',skip_header=1)
        return loadedText


features = TextLoading(csvpath + "test.csv")

mx = features.load()

x = mx[:100, :10]

print(mx.shape)

tx = mx.transpose()

np.matmul(mx,tx)



data1 = DataProcess(csvpath + "train.csv", csvpath + "getdata.csv")

data1.getFeatures()

data2 = DataProcess2(csvpath + "train.csv", csvpath + "datacls.csv")

data2.fileSize()

data2.getLastColumn()
#Just Classes - Old

train = open(csvpath + "train.csv", "r")

classcsv = train.readlines()

open(csvpath + "trainclass.csv", "w", encoding="utf-8")

for i in range(len(classcsv)):
    str1 = classcsv[i].replace("Class_", "")
    str1 = re.sub(r'^.*,', '', str1)
    f = open(csvpath + "trainclass.csv", "a", encoding="utf-8")
    f.write(str1)
    f.close()

#-------------------------------------------------------------------------------------------------------------

def classcsv(path, newfilepath):
    train = open(path, "r")

    classcsv = train.readlines()

    open(newfilepath, "w", encoding="utf-8")

    for i in range(len(classcsv)):
        str1 = classcsv[i].replace("Class_", "")
        str1 = re.sub(r'^.*,', '', str1)
        f = open(newfilepath, "a", encoding="utf-8")
        f.write(str1)
        f.close()

classcsv('C:/PythonProjects/code/data/train.csv', 'C:/PythonProjects/code/data/classtrain.csv')


#Data other than ID and Class

train = open(csvpath + "train.csv", "r")

classcsv = train.readlines()

open(csvpath + "traindata.csv", "w", encoding="utf-8")

for i in range(len(classcsv)):
    str1 = classcsv[i]
    str1 = re.sub(r',Class_.*$', '', str1)
    #str1 = str1.replace(",Class", "")
    str1 = re.sub(r'^(.+?),', '', str1)
    f = open(csvpath + "traindata.csv", "a", encoding="utf-8")
    f.write(str1)
    f.close()

#-------------------------------------------------------------------------------------------------------------

def noclasscsv(path, newfilepath):
    train = open(path, "r")

    classcsv = train.readlines()

    open(newfilepath, "w", encoding="utf-8")

    for i in range(len(classcsv)):
        str1 = classcsv[i]
        str1 = re.sub(r',Class_.*$', '', str1)
        str1 = re.sub(r'^(.+?),', '', str1)
        f = open(newfilepath, "a", encoding="utf-8")
        f.write(str1)
        f.close()

noclasscsv('C:/PythonProjects/code/data/train.csv', 'C:/PythonProjects/code/data/newtrain.csv')



def clstobin(path, newfilepath):
    train = open(path, "r")

    classcsv = train.readlines()

    open(newfilepath, "w", encoding="utf-8")

    for i in range(len(classcsv)):
        str1 = classcsv[i].replace("Class_", "")
        str1 = re.sub(r'^.*,', '', str1)
        if str1 == "1\n":
            str1 = "1,0,0,0,0,0,0,0,0\n"
        elif str1 == "2\n":
            str1 = "0,1,0,0,0,0,0,0,0\n"
        elif str1 == "3\n":
            str1 = "0,0,1,0,0,0,0,0,0\n"
        elif str1 == "4\n":
            str1 = "0,0,0,1,0,0,0,0,0\n"
        elif str1 == "5\n":
            str1 = "0,0,0,0,1,0,0,0,0\n"
        elif str1 == "6\n":
            str1 = "0,0,0,0,0,1,0,0,0\n"
        elif str1 == "7\n":
            str1 = "0,0,0,0,0,0,1,0,0\n"
        elif str1 == "8\n":
            str1 = "0,0,0,0,0,0,0,1,0\n"
        elif str1 == "9\n":
            str1 = "0,0,0,0,0,0,0,0,1\n"
        else:
            str1 = ""

        f = open(newfilepath, "a", encoding="utf-8")
        f.write(str1)
        f.close()

clstobin('C:/PythonProjects/irispytorch/data/train.csv', 'C:/PythonProjects/irispytorch/data/clstobin.csv')


def clsround(path, newfilepath):
    train = open(path, "r")

    classcsv = train.readlines()

    open(newfilepath, "w", encoding="utf-8")

    for i in range(len(classcsv)):
        str1 = classcsv[i].replace("Class_", "")
        str1 = re.sub(r'^.*,', '', str1)
        if str1 == "1\n" or str1 == "2\n" or str1 == "3\n" or str1 == "4\n" or str1 == "5\n":
            str1 = "1\n"
        elif str1 == "6\n" or str1 == "7\n" or str1 == "8\n" or str1 == "9\n" or str1 == "10\n":
            str1 = "0\n"


        f = open(newfilepath, "a", encoding="utf-8")
        f.write(str1)
        f.close()

clsround('C:/PythonProjects/irispytorch/data/train.csv', 'C:/PythonProjects/irispytorch/data/clsround.csv')

'''
str1 = 'abd,cde '

str2 = re.sub(r'.*,', '', str1)


print(str2)

testcsv = np.genfromtxt(csvpath + 'test.csv',delimiter=',',skip_header=1)

samplesubmission = np.genfromtxt(csvpath + 'sample_submission.csv',delimiter=',',skip_header=1)



traincsv = np.genfromtxt(csvpath + 'train.csv',delimiter=',',skip_header=1)

traincsv.shape

traincsv[1][0]

traincsv[2][76]

'''
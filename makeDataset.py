import os, shutil

rawDataset = 'train' # Where raw data is, need to be an existing path
targetDirt ='trainTest' # target folder, will be create by program
# change the value to adjust size
trainSetSize = 11000
validationSize = 750
testSize = 750
os.mkdir('trainResult')
os.mkdir(targetDirt)

# Initialize folders
train_dir = os.path.join(targetDirt, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(targetDirt, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(targetDirt, 'test')
os.mkdir(test_dir)

trainCats = os.path.join(train_dir, 'cats')
os.mkdir(trainCats)

trainDogs = os.path.join(train_dir, 'dogs')
os.mkdir(trainDogs)

validationCats = os.path.join(validation_dir, 'cats')
os.mkdir(validationCats)

validationDogs = os.path.join(validation_dir, 'dogs')
os.mkdir(validationDogs)

testCats = os.path.join(test_dir, 'cats')
os.mkdir(testCats)

testDogs = os.path.join(test_dir, 'dogs')
os.mkdir(testDogs)


trainSetCat = ['cat.{}.jpg'.format(i) for i in range(trainSetSize)]
for tsc in trainSetCat:
    shutil.copyfile(os.path.join(rawDataset, tsc), os.path.join(trainCats, tsc))

valiSetCat = ['cat.{}.jpg'.format(i) for i in range(trainSetSize, trainSetSize+validationSize)]
for vsc in valiSetCat:
    shutil.copyfile(os.path.join(rawDataset, vsc), os.path.join(validationCats, vsc))

testSetCat = ['cat.{}.jpg'.format(i) for i in range(trainSetSize+validationSize, trainSetSize+validationSize+testSize)]
for ttsc in testSetCat:
    shutil.copyfile(os.path.join(rawDataset, ttsc), os.path.join(testCats, ttsc))

trainSetDog = ['dog.{}.jpg'.format(i) for i in range(trainSetSize)]
for tsd in trainSetDog:
    shutil.copyfile(os.path.join(rawDataset, tsd), os.path.join(trainDogs, tsd))

valiSetDog = ['dog.{}.jpg'.format(i) for i in range(trainSetSize, trainSetSize+validationSize)]
for vsd in valiSetDog:
    shutil.copyfile(os.path.join(rawDataset, vsd),  os.path.join(validationDogs, vsd))

testSetDog = ['dog.{}.jpg'.format(i) for i in range(trainSetSize+validationSize, trainSetSize+validationSize+testSize)]
for ttsd in testSetDog:
    shutil.copyfile(os.path.join(rawDataset, ttsd), os.path.join(testDogs, ttsd))

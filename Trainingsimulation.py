print('Setting Up')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utlis import *
from sklearn .model_selection import train_test_split
path='dataset'
data=importDataInfo(path)

data=balanceData(data,display=False)

imagepath, steering=loadData(path,data)
print(imagepath[0],steering[0])

xTrain,xVal,yTrain,yVal=train_test_split(imagepath, steering, test_size=0.2, random_state=5)
print('Total training images', len(xTrain))
print('Total validation images', len(xVal))

model=createModel()
model.summary()


history = model.fit(batchGen(xTrain, yTrain, 100, 1), steps_per_epoch=300, epochs=50,
                    validation_data=batchGen(xVal, yVal , 100, 0), validation_steps=200)

model.save('model.h5')
print('model saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0, 1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
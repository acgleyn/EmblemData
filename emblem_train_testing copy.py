from ultralytics import YOLO

model = YOLO('yolov8m.pt')

model.train(data='emblem_test3HY.yaml', epochs= 50)

'''
test1 = YOLO('./runs/detect/train2/weights/best.pt' )
test2 = YOLO('./best.pt') #emblems 2.0 train 
results = test2.predict(source ='./Veloster.JPG', save = True)
'''
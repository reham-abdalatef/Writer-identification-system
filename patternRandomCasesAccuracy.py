import cv2
import glob
from featuresModule import lbp
import time
import random
from testingModule import test
from preprocessingModule import crop

folders = ['000','013','016','017','019','025','026','037','058','059','060','061','062','063','064','085','087','088','089','090','092','093','094','107','108','109','110','111','112','113','114','117','118','123','124','125','126','127','128','129','130','131','132','133','150','151','152','153','154','155','158','173','174','181','193','199','202','203','204','205','206','207','208','209','212','213','217','239','241','242','243','246','247','248','272','273','274','285','286','287','288','289','291','292','293','294','296','297','298','299','300','315','330','332','333','334','335','336','337','338','339','340','341','342','343','344','345','346','347','348','349','350','351','352','353','354','355','357','384','385','386','387','388','389','390','391','392','393','415','454','455','456','458','495','496','497','498','544','545','546','547','548','549','550','551','552','555','567','582','583','584','585','586','587','588','634','635','670','671']
random.seed(103) # same seed = same tests every time!
numTests = 500
correct = 0

for testId in range (0,numTests):
    start = time.time()
    print("=================test case",testId+1,"======================") 
    a = random.randint(0,len(folders)-1)
    b = random.randint(0,len(folders)-1)
    while a==b:
        b=random.randint(0,len(folders)-1)
    c = random.randint(0,len(folders)-1)
    while a==c or b==c :
        c=random.randint(0,len(folders)-1)
    expected = random.randint(1,3)
    folderNames = [folders[a],folders[b],folders[c]]
    features=[]
    label=[]
    testImage = []
    for i in range(0,3):
        images = [cv2.imread(file) for file in glob.glob("new_form/"+folderNames[i]+ "/*.PNG")]
        for j in range(0,2):
            features.append(lbp(crop(images[j])))
            label.append(i+1)
        if i+1 == expected:
            testImage = images[2]
    
    predict = test(crop(testImage),features,label)
    if predict == expected:
        print("Success!")
        correct += 1 
    if predict != expected:
        print("Fail :(")
        print("expected =",expected,"predicted =",predict,"case#",testId+1,file=open("newWrongCases.txt", "a"))
        print(folderNames,file=open("newWrongCases.txt", "a"))
    end = time.time()
    print("Time taken=",end - start)
    
accuracy = 100.0*correct/numTests
print("accuracy = " +str(accuracy) + "%")
    
    
    
    
    
    
    
    
    
    
    
    

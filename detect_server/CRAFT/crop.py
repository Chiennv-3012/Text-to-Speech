from PIL import Image
import os.path
import re
import codecs

def crop(inFolderImage, inFolderText):
    dirsImage = os.listdir(inFolderImage)
    dirsText = os.listdir(inFolderText)
    
    outFolder = "./resultImage/"
    if not os.path.isdir(outFolder):
        os.mkdir(outFolder)

    for itemImage in dirsImage:
        fullpathImage = os.path.join(inFolderImage,itemImage)         
        pathResult = os.path.join(outFolder,itemImage)
        
        if os.path.isfile(fullpathImage):
            imageFileName, imageFileExtension = os.path.splitext(fullpathImage)
            imageName = imageFileName.split("/")
            for itemText in dirsText:
                fullpathText = os.path.join(inFolderText, itemText)
                if os.path.isfile(fullpathText):
                    textFileName, textFileExtension = os.path.splitext(fullpathText)
                    textname = textFileName.split("/")
                    if (textname[len(textname) - 1] == imageName[len(imageName) - 1]):
                        f = open(fullpathText, "r")
                        i = 1
                        for str in f:
                            im = Image.open(fullpathImage)
                            p,s = os.path.splitext(pathResult)
                            d = map(int, re.findall(r'\d+', str))
                            imCrop = im.crop((d))      #corrected
                            imCrop.save(p + 'Cropped' + '% s' %i + '.png', "PNG", quality=100)
                            i += 1
    return outFolder

def crop2(inFolderImage, inFolderText):
    dirsImage = os.listdir(inFolderImage)
    dirsText = os.listdir(inFolderText)
    
    outFolder = "./resultImage/"
    if not os.path.isdir(outFolder):
        os.mkdir(outFolder)

    for itemImage in dirsImage:
        fullpathImage = os.path.join(inFolderImage,itemImage)         
        pathResult = os.path.join(outFolder,itemImage)
        
        if os.path.isfile(fullpathImage):
            imageFileName, imageFileExtension = os.path.splitext(fullpathImage)
            imageName = imageFileName.split("/")
            for itemText in dirsText:
                fullpathText = os.path.join(inFolderText, itemText)
                if os.path.isfile(fullpathText):
                    textFileName, textFileExtension = os.path.splitext(fullpathText)
                    textname = textFileName.split("/")
                    if (textname[len(textname) - 1] == imageName[len(imageName) - 1]):
                        f = open(fullpathText, "r")
                        i = 1
                        for str in f:
                            im = Image.open(fullpathImage)
                            p,s = os.path.splitext(pathResult)
                            d = map(int, re.findall(r'\d+', str))
                            imCrop = im.crop((d))      
                            imCrop.save(p + 'Cropped' + '% s' %i + '.png', "PNG", quality=100)
                            i += 1
    return outFolder

inFolderImage = '../static/uploads/'
inFolderText = '../result/'
crop2(inFolderImage, inFolderText)
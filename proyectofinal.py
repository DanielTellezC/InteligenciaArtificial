import numpy as np
import math
import cv2
import glob

imagesNormal = [cv2.imread(file) for file in glob.glob("NORMAL_VISION/*.jpeg")]
imagesNeumonia = [cv2.imread(file) for file in glob.glob("PNEUMONIA_VISION/*.jpeg")]

if __name__ == '__main__':

    ###Las imagenes se convierten a escala de grises para que sea más fácil el procesamiento###
    imagesNormalGrises = [cv2.cvtColor(file, cv2.COLOR_RGB2GRAY) for file in imagesNormal]
    imagesNeumoniaGrises = [cv2.cvtColor(file, cv2.COLOR_RGB2GRAY) for file in imagesNeumonia]

    ### Se hace una ecualización de imagenes para resaltar las partes afectadas#
    imagesNormalEcualizadas = [cv2.equalizeHist(file) for file in imagesNormalGrises]
    imagesNeumoniaEcualizadas =  [cv2.equalizeHist(file) for file in imagesNeumoniaGrises]
    ### Se saca la media de cada imagen ###
    imagesNormalMedia = [np.mean(file) for file in imagesNormalEcualizadas]
    imagesNeumoniaMedia = [np.mean(file) for file in imagesNeumoniaEcualizadas]

    cv2.imshow("Ecualizada",imagesNormalEcualizadas[0])
    cv2.imshow("normal", imagesNormalGrises[0])

    print(imagesNormalMedia[0])
    print(imagesNormalGrises[0].shape)
    print(imagesNormal[0].shape)
    print(imagesNeumonia[0].shape)
    print(len(imagesNormal))
    print(len(imagesNeumonia))

    cv2.waitKey(0)
    cv2.destroyAllWindows()